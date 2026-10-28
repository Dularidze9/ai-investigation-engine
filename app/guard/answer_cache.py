"""Answer cache in Postgres: the same question asked twice costs nothing the second time.

The key is the normalized question. Every document add/update/delete clears the cache (clear()),
because a cached answer may cite a chunk that changed or no longer exists.
"""
import hashlib
import json
import re

VERSION = "v1"  # bump when the agent or prompts change, so old answers are not served


def key(question: str) -> str:
    norm = re.sub(r"\s+", " ", question.strip().lower())
    return hashlib.sha256(f"{VERSION}|{norm}".encode()).hexdigest()


def get(conn, question: str) -> dict | None:
    row = conn.execute("SELECT answer FROM answer_cache WHERE key = %s", (key(question),)).fetchone()
    return row[0] if row else None


def put(conn, question: str, answer: dict) -> None:
    conn.execute("INSERT INTO answer_cache (key, answer) VALUES (%s, %s::jsonb) "
                 "ON CONFLICT (key) DO UPDATE SET answer = EXCLUDED.answer, created_at = now()",
                 (key(question), json.dumps(answer, ensure_ascii=False)))
    conn.commit()


def clear(conn) -> None:
    conn.execute("DELETE FROM answer_cache")
