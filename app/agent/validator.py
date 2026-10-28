"""Citation validator: the model may only cite chunks we actually retrieved.

Rules:
- citation ids not in the retrieved set are removed;
- a claim left with no valid citation is removed (no uncited claims reach the user);
- a conflict needs at least 2 valid citations (a disagreement needs two sources);
- the status is downgraded to match what survived.
"""
from app.agent.schema import Answer, Claim, Conflict

NO_EVIDENCE = "The documents do not contain enough evidence to answer this question."


def validate(answer: Answer, retrieved: set[int]) -> tuple[Answer, list[str]]:
    issues: list[str] = []
    claims = []
    for c in answer.claims:
        valid = [i for i in dict.fromkeys(c.citations) if i in retrieved]
        if len(valid) < len(set(c.citations)):
            issues.append(f"removed unknown citation(s) {sorted(set(c.citations) - set(valid))}")
        if valid:
            claims.append(Claim(text=c.text, citations=valid))
        else:
            issues.append(f"dropped uncited claim: {c.text[:60]}")

    conflicts = []
    for c in answer.conflicts:
        valid = [i for i in dict.fromkeys(c.citations) if i in retrieved]
        if len(valid) >= 2:
            conflicts.append(Conflict(description=c.description, citations=valid))
        else:
            issues.append(f"dropped conflict without two valid sources: {c.description[:60]}")

    status, text = answer.status, answer.answer
    if status == "insufficient_evidence":
        claims, conflicts = [], []  # an abstention carries no claims
        text = text.strip() or NO_EVIDENCE  # the model often leaves "answer" empty when it abstains
    elif not claims and not conflicts:
        status, text = "insufficient_evidence", NO_EVIDENCE
        issues.append("no supported claims left: abstaining")
    elif status == "conflicting" and not conflicts:
        status = "answered" if claims else "insufficient_evidence"
        issues.append("conflict not supported by two sources: status changed")
    elif status == "answered" and len(claims) < len(answer.claims):
        status = "partial"
        issues.append("some claims were unsupported: status changed to partial")

    return Answer(status=status, answer=text, claims=claims, conflicts=conflicts, missing=answer.missing), issues
