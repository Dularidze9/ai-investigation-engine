"""Turn data/corpus/paragraphs.jsonl into chunks (strategy A) in a deterministic order."""
import json
from dataclasses import dataclass
from pathlib import Path

from app.ingestion.chunking import chunk_a


@dataclass(frozen=True)
class Chunk:
    doc_hash: str
    title: str
    index: int
    text: str

    @property
    def key(self) -> str:
        return f"{self.doc_hash}:{self.index}"


def load_chunks(path: Path) -> list[Chunk]:
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            for i, text in enumerate(chunk_a(p["title"], p["text"])):
                chunks.append(Chunk(p["hash"], p["title"], i, text))
    return chunks


def load_cached_vectors(chunks: list[Chunk], cache_dir: Path, shard: int = 256):
    """Read the embedding shards written by scripts/embed_corpus.py. Returns float32 array [n, 512]."""
    import numpy as np

    vecs = np.zeros((len(chunks), 512), dtype=np.float32)
    for s, start in enumerate(range(0, len(chunks), shard)):
        part = chunks[start:start + shard]
        with np.load(cache_dir / f"shard_{s:05d}.npz") as z:
            assert list(z["keys"]) == [c.key for c in part], f"shard {s} does not match chunks"
            vecs[start:start + len(part)] = z["vecs"].astype(np.float32)
    return vecs
