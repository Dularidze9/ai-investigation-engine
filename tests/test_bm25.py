"""BM25 tests. The first test is the hand-computed example from docs/bm25_rrf_walkthrough.md."""
import math

import pytest

from app.retrieval.bm25 import CorpusStats, idf, rank, score

# Tiny corpus, already tokenized. N = 3, total length 3 + 4 + 5 = 12, avgdl = 4.
DOCS = {
    1: ["payment", "outage", "march"],
    2: ["payment", "gateway", "payment", "retry"],
    3: ["deploy", "ledger", "service", "march", "rollback"],
}
DF = {"payment": 2, "outage": 1, "march": 2, "gateway": 1, "retry": 1,
      "deploy": 1, "ledger": 1, "service": 1, "rollback": 1}
STATS = CorpusStats(n_docs=3, avgdl=4.0)
QUERY = ["payment", "march"]


def test_hand_computed_example():
    # idf(payment) = idf(march) = ln((3 - 2 + 0.5) / (2 + 0.5) + 1) = ln(1.6) = 0.470004
    assert idf(2, 3) == pytest.approx(math.log(1.6))
    result = rank(QUERY, DOCS, DF, STATS)
    assert [doc for doc, _ in result] == [1, 2, 3]
    scores = dict(result)
    assert scores[1] == pytest.approx(1.047097, abs=1e-6)  # both terms, short doc
    assert scores[2] == pytest.approx(0.646255, abs=1e-6)  # "payment" twice, no "march"
    assert scores[3] == pytest.approx(0.426395, abs=1e-6)  # "march" once, long doc


def test_idf_simplified_form_is_the_same():
    for n, df in [(3, 2), (66889, 1), (66889, 30000), (10, 10)]:
        assert idf(df, n) == pytest.approx(math.log((n + 1) / (df + 0.5)))


def test_idf_is_positive_even_for_very_common_terms():
    assert idf(df=3, n_docs=3) > 0  # term in every document still adds a little, never subtracts


def test_rare_term_beats_common_term():
    assert idf(1, 1000) > idf(500, 1000)


def test_term_frequency_saturates():
    one = score(["x"], ["x", "y", "y", "y"], {"x": 1}, STATS)
    two = score(["x"], ["x", "x", "y", "y"], {"x": 1}, STATS)
    four = score(["x"], ["x", "x", "x", "x"], {"x": 1}, STATS)
    assert one < two < four
    assert two - one > four - two  # diminishing returns from k1
    assert four < 2.2 * idf(1, 3)   # bounded by (k1 + 1) * idf


def test_longer_document_scores_lower_for_same_tf():
    short = score(["x"], ["x", "y"], {"x": 1}, STATS)
    long = score(["x"], ["x"] + ["y"] * 9, {"x": 1}, STATS)
    assert short > long


def test_b_zero_turns_off_length_normalization():
    short = score(["x"], ["x", "y"], {"x": 1}, STATS, b=0)
    long = score(["x"], ["x"] + ["y"] * 9, {"x": 1}, STATS, b=0)
    assert short == pytest.approx(long)


def test_repeated_query_term_counts_once():
    assert score(["payment", "payment"], DOCS[1], DF, STATS) == score(["payment"], DOCS[1], DF, STATS)


def test_no_match_empty_query_and_empty_doc():
    assert rank(["nothing"], DOCS, DF, STATS) == []
    assert rank([], DOCS, DF, STATS) == []
    assert score(QUERY, [], DF, STATS) == 0.0


def test_top_k_and_deterministic_ties():
    docs = {7: ["a"], 3: ["a"], 5: ["a"]}
    assert [d for d, _ in rank(["a"], docs, {"a": 3}, CorpusStats(3, 1.0), top_k=2)] == [3, 5]
