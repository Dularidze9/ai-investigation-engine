from app.agent.schema import Answer, Claim, Conflict
from app.agent.validator import NO_EVIDENCE, validate

RETRIEVED = {1, 2, 3}


def ans(status="answered", claims=(), conflicts=()):
    return Answer(status=status, answer="x", claims=list(claims), conflicts=list(conflicts), missing=[])


def test_valid_answer_passes_unchanged():
    out, issues = validate(ans(claims=[Claim(text="a", citations=[1, 2])]), RETRIEVED)
    assert out.status == "answered" and out.claims[0].citations == [1, 2] and issues == []


def test_unknown_citation_is_removed_but_claim_kept():
    out, issues = validate(ans(claims=[Claim(text="a", citations=[1, 99])]), RETRIEVED)
    assert out.claims[0].citations == [1] and any("99" in i for i in issues)


def test_claim_with_only_invented_citations_is_dropped_and_status_downgraded():
    out, _ = validate(ans(claims=[Claim(text="ok", citations=[1]), Claim(text="made up", citations=[42])]), RETRIEVED)
    assert [c.text for c in out.claims] == ["ok"] and out.status == "partial"


def test_no_supported_claims_means_abstain():
    out, _ = validate(ans(claims=[Claim(text="made up", citations=[42]), Claim(text="uncited", citations=[])]),
                      RETRIEVED)
    assert out.status == "insufficient_evidence" and out.claims == [] and out.answer == NO_EVIDENCE


def test_conflict_needs_two_valid_sources():
    out, _ = validate(ans(status="conflicting", claims=[Claim(text="a", citations=[1])],
                          conflicts=[Conflict(description="dates differ", citations=[1, 77])]), RETRIEVED)
    assert out.conflicts == [] and out.status == "answered"


def test_supported_conflict_is_kept():
    out, _ = validate(ans(status="conflicting", conflicts=[Conflict(description="d", citations=[1, 3])]), RETRIEVED)
    assert out.status == "conflicting" and out.conflicts[0].citations == [1, 3]


def test_conflicting_with_empty_answer_gets_the_conflict_text():
    a = Answer(status="conflicting", answer="", claims=[],
               conflicts=[Conflict(description="2h40m vs 3h10m.", citations=[1, 2])], missing=[])
    out, _ = validate(a, RETRIEVED)
    assert out.answer == "Sources disagree: 2h40m vs 3h10m."


def test_abstention_never_carries_claims():
    out, _ = validate(ans(status="insufficient_evidence", claims=[Claim(text="a", citations=[1])]), RETRIEVED)
    assert out.status == "insufficient_evidence" and out.claims == []


def test_abstention_with_empty_answer_gets_a_message():
    a = Answer(status="insufficient_evidence", answer="", claims=[], conflicts=[], missing=["revenue"])
    out, _ = validate(a, RETRIEVED)
    assert out.answer == NO_EVIDENCE and out.missing == ["revenue"]


def test_duplicate_citations_collapse():
    out, _ = validate(ans(claims=[Claim(text="a", citations=[2, 2, 1])]), RETRIEVED)
    assert out.claims[0].citations == [2, 1]
