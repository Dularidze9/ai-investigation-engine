"""Postgres connection pool (Neon, pooled URL). Opened lazily so the app can start while Neon is asleep."""
import logging
import time

from psycopg_pool import ConnectionPool

from app.config import get_settings

log = logging.getLogger(__name__)
_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        s = get_settings()
        if not s.database_url:
            raise RuntimeError("DATABASE_URL is not set")
        _pool = ConnectionPool(
            s.database_url,
            min_size=0,
            max_size=4,
            open=False,
            timeout=s.db_connect_timeout,
            kwargs={"connect_timeout": s.db_connect_timeout},
        )
        _pool.open()
    return _pool


def check_db(retries: int = 3) -> dict:
    """Ping the database and report the pgvector version. Retries to survive a Neon cold start."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with get_pool().connection() as conn:
                row = conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
            return {"db": "ok", "pgvector": row[0] if row else None}
        except Exception as e:  # noqa: BLE001 - any connection failure is retried then reported
            last = e
            log.warning("db check attempt %d failed: %s", attempt + 1, type(e).__name__)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"database unreachable: {type(last).__name__}")
