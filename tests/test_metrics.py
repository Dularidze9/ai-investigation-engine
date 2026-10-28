import pytest

from eval.metrics import all_gold_at_k, dedupe, recall_at_k, reciprocal_rank, score_question, summarize

GOLD = {"A", "B"}


def test_dedupe_keeps_best_rank():
    assert dedupe(["x", "A", "x", "B", "A"]) == ["x", "A", "B"]


def test_recall_at_k():
    ranked = ["x", "A", "y", "z", "w", "B"]
    assert recall_at_k(ranked, GOLD, 5) == 0.5   # only A in top 5
    assert recall_at_k(ranked, GOLD, 6) == 1.0
    assert recall_at_k([], GOLD, 10) == 0.0


def test_all_gold_needs_every_gold_document():
    assert all_gold_at_k(["A", "B"], GOLD, 10) == 1.0
    assert all_gold_at_k(["A", "x"], GOLD, 10) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "y", "B"], GOLD) == pytest.approx(1 / 3)
    assert reciprocal_rank(["A"], GOLD) == 1.0
    assert reciprocal_rank(["x", "y"], GOLD) == 0.0


def test_summarize_is_the_mean():
    rows = [score_question(["A", "B"], GOLD), score_question(["x"], GOLD)]
    s = summarize(rows)
    assert s["recall@10"] == pytest.approx(0.5)
    assert s["mrr"] == pytest.approx(0.5)
    assert s["all_gold@10"] == pytest.approx(0.5)
