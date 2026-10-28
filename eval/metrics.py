"""Retrieval metrics at document (title) level, so chunking strategies A and B are comparable.

ranked: titles in rank order, best first, each title listed once (see dedupe).
gold:   the set of supporting-fact titles for the question.
"""


def dedupe(titles: list[str]) -> list[str]:
    """Keep each title once, at its best (first) rank. Several chunks of one document count as one hit."""
    return list(dict.fromkeys(titles))


def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """Fraction of gold documents found in the top k."""
    return len(gold & set(ranked[:k])) / len(gold)


def all_gold_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    """1.0 if EVERY gold document is in the top k. The multi-hop success rate: the answer needs both."""
    return 1.0 if gold <= set(ranked[:k]) else 0.0


def reciprocal_rank(ranked: list[str], gold: set[str]) -> float:
    """1 / rank of the first gold document, 0 if none is found."""
    for i, title in enumerate(ranked, start=1):
        if title in gold:
            return 1.0 / i
    return 0.0


def score_question(ranked: list[str], gold: set[str]) -> dict[str, float]:
    return {
        "recall@5": recall_at_k(ranked, gold, 5),
        "recall@10": recall_at_k(ranked, gold, 10),
        "all_gold@10": all_gold_at_k(ranked, gold, 10),
        "mrr": reciprocal_rank(ranked, gold),
    }


def summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    """Mean of every metric over the questions."""
    return {m: sum(r[m] for r in rows) / len(rows) for m in rows[0]}
