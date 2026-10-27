"""OpenAI embeddings with retry/backoff. Vectors are 512-d (text-embedding-3-small, shortened) and unit-normalized."""
import logging
import time

import numpy as np
from openai import APIConnectionError, APIStatusError, RateLimitError

log = logging.getLogger(__name__)

MODEL = "text-embedding-3-small"
DIMS = 512
PRICE_PER_TOKEN = 0.02 / 1_000_000


def embed_batch(client, texts: list[str], retries: int = 5, sleep=time.sleep) -> tuple[np.ndarray, int]:
    """Embed texts; returns (float32 array [n, DIMS], tokens used). Retries 429, connection errors and 5xx."""
    for attempt in range(retries):
        try:
            resp = client.embeddings.create(model=MODEL, input=texts, dimensions=DIMS)
            vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            return vecs, resp.usage.total_tokens
        except (RateLimitError, APIConnectionError) as e:
            err = e
        except APIStatusError as e:
            if e.status_code < 500:
                raise
            err = e
        wait = 2 ** attempt
        log.warning("embedding call failed (%s), retry %d in %ds", type(err).__name__, attempt + 1, wait)
        sleep(wait)
    raise err
