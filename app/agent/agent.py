"""The investigation agent.

  1. search    hybrid search on the question (no LLM)
  2. gate      best dense similarity < GATE_MIN_SIM  ->  abstain, no LLM money spent
  3. plan      LLM splits the question into <= 3 sub-queries (structured JSON); each is searched
  4. gather    tool-calling loop, <= MAX_ROUNDS rounds (search_documents, get_document, compare_sources)
  5. answer    LLM writes JSON {status, answer, claims[citations], conflicts, missing} from the evidence
  6. validate  citations must point at retrieved chunks; status is downgraded if claims fall away

Failures degrade instead of crashing: a failed plan falls back to the question itself, a failed tool
round ends gathering early, and only a failed final answer returns status "error".
"""
import json
import logging
from dataclasses import asdict, dataclass, field

from pydantic import ValidationError

from app.agent.schema import ANSWER_SCHEMA, PLAN_SCHEMA, Answer, Plan
from app.agent.tools import TOOLS, Evidence, EvidenceStore, Retriever, Toolbox, blocks
from app.agent.validator import NO_EVIDENCE, validate
from app.llm.client import LLMClient, LLMError
from app.llm import prompts

log = logging.getLogger(__name__)

GATE_MIN_SIM = 0.35        # all 300 answerable eval questions have top-1 similarity >= 0.43 (measured in M6)
FIRST_K = 6
SUB_K = 4
MAX_ROUNDS = 4
MAX_CALLS_PER_ROUND = 3
MAX_EVIDENCE = 24          # chunks sent to the final answer call


@dataclass
class AgentResult:
    question: str
    status: str            # answered | partial | insufficient_evidence | conflicting | error
    answer: str
    claims: list[dict] = field(default_factory=list)
    conflicts: list[dict] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)   # the cited chunks, for the UI
    sub_queries: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    validation_issues: list[str] = field(default_factory=list)
    evidence_count: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class Agent:
    def __init__(self, llm: LLMClient, retriever: Retriever, gate_min_sim: float = GATE_MIN_SIM,
                 max_rounds: int = MAX_ROUNDS, agentic: bool = True, first_k: int = FIRST_K):
        """agentic=False is the single-shot RAG baseline for the evaluation:
        one search (first_k chunks) and one answer call, no planning and no tools."""
        self.llm, self.retriever = llm, retriever
        self.gate_min_sim, self.max_rounds = gate_min_sim, max_rounds
        self.agentic, self.first_k = agentic, first_k

    def run(self, question: str) -> AgentResult:
        r = AgentResult(question=question, status="error", answer="")
        store = EvidenceStore()

        # 1-2. first search and evidence gate
        first = self.retriever.search(question, self.first_k)
        store.add(first)
        best = max((e.dense_sim or 0.0 for e in first), default=0.0)
        r.trace.append(f"search: {len(first)} chunks, best similarity {best:.3f}")
        if best < self.gate_min_sim:
            r.trace.append(f"gate: best similarity below {self.gate_min_sim}, abstaining without LLM")
            r.status, r.answer = "insufficient_evidence", NO_EVIDENCE
            r.missing = ["No document in the index is relevant enough to this question."]
            return r

        if self.agentic:
            # 3. plan and search each sub-query
            r.sub_queries = self._plan(question, r)
            for sq in r.sub_queries:
                if sq.strip().lower() != question.strip().lower():
                    new = store.add(self.retriever.search(sq, SUB_K))
                    r.trace.append(f"sub-query '{sq}': {len(new)} new chunks")
            # 4. tool loop
            self._gather(question, store, r)

        # 5-6. answer and validate
        evidence = self._select_evidence(store)
        r.evidence_count = len(evidence)
        try:
            answer = self._answer(question, evidence, r)
        except LLMError as e:
            r.status, r.answer = "error", f"The answer could not be generated: {e}"
            r.trace.append(f"answer failed: {e}")
            return r
        checked, issues = validate(answer, {e.chunk_id for e in evidence})
        r.validation_issues = issues
        r.status, r.answer, r.missing = checked.status, checked.answer, checked.missing
        r.claims = [c.model_dump() for c in checked.claims]
        r.conflicts = [c.model_dump() for c in checked.conflicts]
        cited = list(dict.fromkeys(i for c in checked.claims + checked.conflicts for i in c.citations))
        r.sources = [{"chunk_id": e.chunk_id, "document_id": e.document_id, "title": e.title, "text": e.text}
                     for i in cited if (e := store.get(i))]
        return r

    def _chat(self, r: AgentResult, *args, **kwargs):
        resp = self.llm.chat(*args, **kwargs)
        r.llm_calls += 1
        r.input_tokens += resp.input_tokens
        r.output_tokens += resp.output_tokens
        r.cost_usd += resp.cost
        return resp

    def _plan(self, question: str, r: AgentResult) -> list[str]:
        try:
            resp = self._chat(r, [{"role": "system", "content": prompts.PLANNER},
                                  {"role": "user", "content": question}],
                              json_schema=PLAN_SCHEMA, max_tokens=200)
            subs = [s.strip() for s in Plan.model_validate_json(resp.content or "").sub_queries if s.strip()]
            if subs:
                r.trace.append(f"plan: {subs}")
                return subs[:3]
        except (LLMError, ValidationError) as e:
            r.trace.append(f"plan failed ({type(e).__name__}): using the question as the only query")
        return [question]

    def _gather(self, question: str, store: EvidenceStore, r: AgentResult) -> None:
        toolbox = Toolbox(self.retriever, store)
        messages = [{"role": "system", "content": prompts.AGENT},
                    {"role": "user", "content": f"Question: {question}\n\nEvidence so far:\n{blocks(store.items())}"}]
        for round_no in range(1, self.max_rounds + 1):
            try:
                resp = self._chat(r, messages, tools=TOOLS, max_tokens=300)
            except LLMError as e:
                r.trace.append(f"tool round {round_no} failed ({e}): answering with the evidence so far")
                return
            if not resp.tool_calls:
                r.trace.append(f"tool round {round_no}: no more tools needed")
                return
            messages.append({"role": "assistant", "content": resp.content, "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": _json(c.arguments)}}
                for c in resp.tool_calls]})
            for i, call in enumerate(resp.tool_calls):
                # the API needs a tool message for EVERY call id, even the ones we skip
                out = toolbox.run(call) if i < MAX_CALLS_PER_ROUND else "skipped: too many calls in one round"
                messages.append({"role": "tool", "tool_call_id": call.id, "content": out})
                r.trace.append(f"tool round {round_no}: {call.name}({_short(call.arguments)})")
        r.trace.append(f"stopped after {self.max_rounds} tool rounds")

    @staticmethod
    def _select_evidence(store: EvidenceStore) -> list[Evidence]:
        items = store.items()
        if len(items) <= MAX_EVIDENCE:
            return items
        half = MAX_EVIDENCE // 2  # keep the first searches AND the latest (multi-hop) findings
        return items[:half] + items[-half:]

    def _answer(self, question: str, evidence: list[Evidence], r: AgentResult) -> Answer:
        messages = [{"role": "system", "content": prompts.ANSWER},
                    {"role": "user", "content": f"Question: {question}\n\nEvidence:\n{blocks(evidence)}"}]
        for attempt in range(2):
            resp = self._chat(r, messages, json_schema=ANSWER_SCHEMA, max_tokens=700)
            try:
                return Answer.model_validate_json(resp.content or "")
            except ValidationError:
                r.trace.append(f"answer attempt {attempt + 1}: invalid JSON")
                messages.append({"role": "user", "content": "Your output did not match the required JSON schema. "
                                                            "Return only valid JSON."})
        raise LLMError("the model did not return a valid answer")


def _json(d: dict) -> str:
    return json.dumps(d, ensure_ascii=False)


def _short(d: dict) -> str:
    return ", ".join(f"{k}={str(v)[:50]}" for k, v in d.items())
