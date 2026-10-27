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
