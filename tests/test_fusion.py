"""RRF and MMR tests. The first test is the hand-computed example from docs/bm25_rrf_walkthrough.md."""
import pytest

from app.retrieval.fusion import cosine, mmr, rrf


def test_hand_computed_rrf_example():
    dense = [3, 1, 2]   # doc 3 is the best semantic match
    bm25 = [1, 2, 3]    # doc 1 is the best keyword match
    fused = rrf([dense, bm25], k=60)
    assert [d for d, _ in fused] == [1, 3, 2]
    s = dict(fused)
    assert s[1] == pytest.approx(1 / 62 + 1 / 61)  # 0.032522
    assert s[3] == pytest.approx(1 / 61 + 1 / 63)  # 0.032266
    assert s[2] == pytest.approx(1 / 63 + 1 / 62)  # 0.032002


def test_found_by_both_beats_top_of_one_list():
    dense = [4, 1, 2]   # doc 4: rank 1 in dense only
    bm25 = [1, 2, 3]
    fused = dict(rrf([dense, bm25]))
    assert fused[2] > fused[4]  # rank 3 + rank 2 in two lists beats rank 1 in one list


def test_weights():
    fused = rrf([[1, 2], [2, 1]], weights=[2.0, 1.0])
    assert fused[0][0] == 1  # the dense list counts double


def test_ties_keep_first_seen_order_and_duplicates_count_once():
    assert [d for d, _ in rrf([[5, 6], [6, 5]])] == [5, 6]
    assert dict(rrf([[1, 1, 2]]))[1] == pytest.approx(1 / 61)


def test_rrf_rejects_wrong_number_of_weights():
    with pytest.raises(ValueError):
        rrf([[1], [2]], weights=[1.0])


def test_cosine():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([0, 0], [1, 1]) == 0.0


VECS = {1: [1.0, 0.0], 2: [0.99, 0.14], 3: [0.0, 1.0]}  # 1 and 2 almost identical, 3 different
REL = {1: 1.0, 2: 0.95, 3: 0.6}


def test_mmr_lambda_one_is_relevance_order():
    assert mmr(REL, VECS, k=3, lam=1.0) == [1, 2, 3]


def test_mmr_promotes_a_different_document():
    # after picking 1, doc 2 is nearly a copy of it, so doc 3 is preferred
    assert mmr(REL, VECS, k=2, lam=0.5) == [1, 3]


def test_mmr_k_larger_than_candidates():
    assert sorted(mmr(REL, VECS, k=10)) == [1, 2, 3]
