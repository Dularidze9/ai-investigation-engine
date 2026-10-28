import pytest

from eval.answer_metrics import citation_scores, exact_match, f1, lenient, must_mention_recall, normalize


def test_normalize_drops_case_punctuation_and_articles():
    assert normalize("The  Lorax!") == "lorax"


def test_normalize_unicode_quotes_and_accents():
    # regression: a curly apostrophe made a correct answer count as wrong
    assert exact_match("Queen In-hyun’s Man", "Queen In-hyun's Man") == 1.0
    assert normalize("Béla Bartók") == "bela bartok"


def test_exact_match_and_yes_no():
    assert exact_match("the Lorax", "The Lorax") == 1.0
    assert exact_match("Yes. Both were American.", "yes") == 1.0
    assert exact_match("No, they were not.", "yes") == 0.0


def test_f1_partial_overlap():
    # pred tokens: new york city ; gold: greenwich village new york city -> P=3/3, R=3/5
    assert f1("New York City.", "Greenwich Village, New York City") == pytest.approx(2 * 1 * 0.6 / 1.6)
    assert f1("Paris", "London") == 0.0


def test_lenient_contains():
    assert lenient("It was directed by Scott Derrickson.", "Scott Derrickson") == 1.0
    assert lenient("New York City", "Greenwich Village, New York City") == 0.0


def test_must_mention_recall():
    assert must_mention_recall("CHG-1052 and CHG-1047 hit payment-gateway", ["chg-1052", "chg-1047", "fraud-check"]) \
        == pytest.approx(2 / 3)
    assert must_mention_recall("anything", []) == 1.0


def test_citation_scores():
    assert citation_scores(["A", "B", "C"], ["A", "D"]) == (pytest.approx(1 / 3), 0.5)
    assert citation_scores([], ["A"]) == (None, 0.0)
    assert citation_scores(["A"], []) == (0.0, None)
