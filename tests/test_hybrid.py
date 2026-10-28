import pytest

from app.retrieval.hybrid import fuse

DENSE = [(1, 0.9), (2, 0.8), (3, 0.7)]
KEYWORD = [(3, 12.0), (4, 9.0), (1, 5.0)]


def test_dense_mode_uses_only_the_dense_list_even_when_both_are_given():
    assert [c for c, _ in fuse(DENSE, KEYWORD, "dense")] == [1, 2, 3]


def test_bm25_mode_uses_only_the_keyword_list_even_when_both_are_given():
    # regression: this used to return the dense list whenever both were populated
    assert [c for c, _ in fuse(DENSE, KEYWORD, "bm25")] == [3, 4, 1]


def test_hybrid_mode_fuses_both():
    ids = [c for c, _ in fuse(DENSE, KEYWORD, "hybrid")]
    assert set(ids) == {1, 2, 3, 4}
    assert ids[:2] == [1, 3]  # found by both lists


def test_unknown_mode():
    with pytest.raises(ValueError):
        fuse(DENSE, KEYWORD, "nope")
