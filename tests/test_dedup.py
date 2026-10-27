import numpy as np

from app.ingestion.dedup import content_hash, exact_dedup, jaccard, near_duplicates


def test_hash_ignores_case_and_whitespace():
    assert content_hash("Hello   World\n") == content_hash("hello world")
    assert content_hash("hello world") != content_hash("hello there")


def test_exact_dedup_keeps_first():
    keep, dropped = exact_dedup(["a b", "c d", "A  B", "c d"])
    assert keep == [0, 1]
    assert dropped == {2: 0, 3: 1}


def test_jaccard_identical_and_disjoint():
    assert jaccard("a b c d e", "a b c d e") == 1.0
    assert jaccard("a b c d e", "v w x y z") == 0.0


def test_near_duplicate_needs_both_cosine_and_jaccard():
    base = "the payment gateway failed after the march deploy of the ledger service today"
    near = base + " again"          # shingles almost identical
    other = "completely different text about cooking pasta with tomatoes and basil leaves"
    texts = [base, near, other, base.replace("payment", "billing").replace("march", "april")]
    v = np.array([[1.0, 0.0], [0.999, 0.01], [0.0, 1.0], [0.999, 0.02]])
    dropped = near_duplicates(v, texts)
    assert dropped[1] == 0                  # near-duplicate of 0
    assert 2 not in dropped                 # different vector
    assert 3 not in dropped                 # close vector but Jaccard too low


def test_near_duplicates_empty_when_all_distinct():
    v = np.eye(3)
    assert near_duplicates(v, ["a b c d", "e f g h", "i j k l"]) == {}
