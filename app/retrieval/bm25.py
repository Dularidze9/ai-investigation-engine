"""Okapi BM25, written by hand. This is our own ranking component (no search library).

For a query q and a document (chunk) d:

    score(q, d) = sum over unique query terms t that occur in d of
                  IDF(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))

    IDF(t) = ln( (N - df + 0.5) / (df + 0.5) + 1 )      which simplifies to  ln( (N + 1) / (df + 0.5) )

    tf    = how many times t occurs in d
    dl    = length of d in tokens;  avgdl = average length over the corpus
    N     = number of chunks;       df = number of chunks that contain t
    k1    = term-frequency saturation (1.2): the 2nd, 3rd... occurrence adds less and less
    b     = length normalization (0.75): long chunks are penalized, b=0 turns that off

N, avgdl and df come from our own tables (corpus_stats, term_stats). Terms must come from
app.retrieval.tokenize.tokenize, the same function that built those tables.
"""
import math
from collections import Counter
from dataclasses import dataclass

K1 = 1.2
B = 0.75


@dataclass(frozen=True)
class CorpusStats:
    n_docs: int     # N
    avgdl: float    # average document length in tokens


def idf(df: int, n_docs: int) -> float:
    """Rare terms get a high weight, common terms a low one. The +1 keeps it positive."""
    return math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)


def score(query_terms: list[str], doc_terms: list[str], df: dict[str, int],
          stats: CorpusStats, k1: float = K1, b: float = B) -> float:
    if not doc_terms:
        return 0.0
    tf = Counter(doc_terms)
    # The part of the denominator that depends only on document length: computed once per document.
    length_norm = k1 * (1 - b + b * len(doc_terms) / stats.avgdl)
    total = 0.0
    for term in dict.fromkeys(query_terms):  # unique terms, original order
        f = tf.get(term, 0)
        if f == 0:
            continue
        total += idf(df.get(term, 0), stats.n_docs) * f * (k1 + 1) / (f + length_norm)
    return total


def rank(query_terms: list[str], docs: dict[int, list[str]], df: dict[str, int],
         stats: CorpusStats, top_k: int | None = None,
         k1: float = K1, b: float = B) -> list[tuple[int, float]]:
    """Score every candidate and return [(doc_id, score)] best first. Zero scores are dropped.
    Ties are broken by the smaller doc id so results are deterministic."""
    scored = [(doc_id, score(query_terms, terms, df, stats, k1, b)) for doc_id, terms in docs.items()]
    scored = [s for s in scored if s[1] > 0]
    scored.sort(key=lambda s: (-s[1], s[0]))
    return scored[:top_k] if top_k is not None else scored
