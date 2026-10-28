"""The three tools the agent can call, their JSON definitions, and the evidence store.

Every chunk a tool returns goes into the EvidenceStore. Only chunks in the store may be cited
(the validator enforces this), so the model cannot cite something it never saw.
"""
import json
import re
from dataclasses import dataclass
from typing import Protocol

from app.llm.client import ToolCall
from app.llm.prompts import evidence_block

SEARCH_K = 5
DOC_CHUNKS = 6


@dataclass
class Evidence:
    chunk_id: int
    document_id: int
    title: str
    text: str
    dense_sim: float | None = None


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Evidence]: ...
    def get_document(self, document_id: int, limit: int) -> list[Evidence]: ...


class EvidenceStore:
    """Insertion-ordered set of evidence chunks."""

    def __init__(self):
        self._items: dict[int, Evidence] = {}

    def add(self, items: list[Evidence]) -> list[Evidence]:
        new = [e for e in items if e.chunk_id not in self._items]
        for e in new:
            self._items[e.chunk_id] = e
        return new

    def ids(self) -> set[int]:
        return set(self._items)

    def get(self, chunk_id: int) -> Evidence | None:
        return self._items.get(chunk_id)

    def items(self) -> list[Evidence]:
        return list(self._items.values())

    def __len__(self):
        return len(self._items)


def blocks(items: list[Evidence]) -> str:
    return "\n\n".join(evidence_block(e.chunk_id, e.title, e.text) for e in items) or "(no results)"


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required, "additionalProperties": False}}}


TOOLS = [
    _fn("search_documents", "Hybrid (semantic + keyword) search over all documents. Returns the best chunks.",
        {"query": {"type": "string", "description": "A precise search query"}}, ["query"]),
    _fn("get_document", "Read the chunks of one document by its id (document_id from a search result).",
        {"document_id": {"type": "integer"}}, ["document_id"]),
    _fn("compare_sources", "Put 2 to 6 already retrieved chunks side by side and list the numbers and dates "
        "each one states, to check whether sources agree.",
        {"chunk_ids": {"type": "array", "items": {"type": "integer"}},
         "aspect": {"type": "string", "description": "What to compare, e.g. 'release year'"}},
        ["chunk_ids", "aspect"]),
]

_VALUE = re.compile(r"\b(?:\d{1,2} (?:January|February|March|April|May|June|July|August|September|October|"
                    r"November|December) \d{4}|(?:January|February|March|April|May|June|July|August|"
                    r"September|October|November|December) \d{1,2}, \d{4}|\d[\d,.]*%?)")


class Toolbox:
    def __init__(self, retriever: Retriever, store: EvidenceStore):
        self.retriever, self.store = retriever, store

    def run(self, call: ToolCall) -> str:
        """Execute one tool call. Bad arguments return an error string for the model, never an exception."""
        a = call.arguments
        try:
            if call.name == "search_documents":
                query = str(a["query"]).strip()[:300]
                if not query:
                    return "error: empty query"
                found = self.retriever.search(query, SEARCH_K)
                self.store.add(found)
                return blocks(found)
            if call.name == "get_document":
                found = self.retriever.get_document(int(a["document_id"]), DOC_CHUNKS)
                self.store.add(found)
                return blocks(found)
            if call.name == "compare_sources":
                return self._compare([int(i) for i in a["chunk_ids"]][:6], str(a.get("aspect", "")))
            return f"error: unknown tool {call.name}"
        except (KeyError, TypeError, ValueError) as e:
            return f"error: bad arguments for {call.name} ({type(e).__name__})"

    def _compare(self, ids: list[int], aspect: str) -> str:
        known = [self.store.get(i) for i in ids if self.store.get(i)]
        if len(known) < 2:
            return "error: compare_sources needs at least 2 chunk_ids that were already retrieved"
        rows = [{"chunk_id": e.chunk_id, "title": e.title, "values_found": list(dict.fromkeys(_VALUE.findall(e.text)))[:15]}
                for e in known]
        return json.dumps({"aspect": aspect, "sources": rows}, ensure_ascii=False) + "\n\n" + blocks(known)
