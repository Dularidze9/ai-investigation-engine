"""Answer-quality metrics.

EM / F1: the standard SQuAD/HotpotQA definitions after normalization (lowercase, no punctuation,
no articles). Yes/no questions compare only the first word of the prediction.
Lenient match: the normalized gold answer appears inside the normalized prediction (models tend to
answer in a full phrase, which EM punishes even when the answer is right).
must_mention: for the long synthetic answers, the share of required terms present.
Citation precision/recall are computed on document titles of the cited chunks.
"""
import re
import string
import unicodedata
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(s: str) -> str:
    # Unicode first: curly quotes and accents ("In-hyun’s", "Bartók") must match their ASCII forms.
    s = s.translate(str.maketrans("‘’“”–—", "''\"\"--"))
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _yes_no(pred: str) -> str:
    words = normalize(pred).split()
    return words[0] if words else ""


def exact_match(pred: str, gold: str) -> float:
    if normalize(gold) in ("yes", "no"):
        return float(_yes_no(pred) == normalize(gold))
    return float(normalize(pred) == normalize(gold))


def f1(pred: str, gold: str) -> float:
    if normalize(gold) in ("yes", "no"):
        return exact_match(pred, gold)
    p, g = normalize(pred).split(), normalize(gold).split()
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def lenient(pred: str, gold: str) -> float:
    if normalize(gold) in ("yes", "no"):
        return exact_match(pred, gold)
    return float(bool(normalize(gold)) and normalize(gold) in normalize(pred))


def must_mention_recall(text: str, terms: list[str]) -> float:
    if not terms:
        return 1.0
    t = text.lower()
    return sum(term.lower() in t for term in terms) / len(terms)


def citation_scores(cited_titles: list[str], gold_titles: list[str]) -> tuple[float | None, float | None]:
    """(precision, recall) over distinct document titles; None when undefined."""
    cited, gold = set(cited_titles), set(gold_titles)
    precision = len(cited & gold) / len(cited) if cited else None
    recall = len(cited & gold) / len(gold) if gold else None
    return precision, recall
