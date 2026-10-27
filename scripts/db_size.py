"""Report real database size per table and per index (Neon free limit is 512 MB; our target <= 350 MB)."""
import os
import sys

import psycopg
from dotenv import load_dotenv

load_dotenv()
url = os.environ.get("DATABASE_URL")
if not url:
    sys.exit("DATABASE_URL is not set")

with psycopg.connect(url, connect_timeout=30) as conn:
    print("TABLES (heap + toast + indexes):")
    for name, total, heap in conn.execute(
        "SELECT relname, pg_total_relation_size(oid), pg_relation_size(oid) "
        "FROM pg_class WHERE relkind='r' AND relnamespace='public'::regnamespace ORDER BY 2 DESC"):
        print(f"  {name:15s} total {total / 1e6:8.1f} MB   heap {heap / 1e6:8.1f} MB")
    print("INDEXES:")
    for name, size in conn.execute(
        "SELECT indexrelid::regclass::text, pg_relation_size(indexrelid) FROM pg_index "
        "WHERE indrelid IN (SELECT oid FROM pg_class WHERE relnamespace='public'::regnamespace) ORDER BY 2 DESC"):
        print(f"  {name:30s} {size / 1e6:8.1f} MB")
    total = conn.execute("SELECT pg_database_size(current_database())").fetchone()[0] / 1e6
    rows = {t: conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            for t in ("documents", "chunks", "term_stats")}
    print("ROWS:", rows)
    print(f"DATABASE TOTAL: {total:.1f} MB  (target <= 350 MB, hard free-tier limit 512 MB)")
    print("WITHIN TARGET:", total <= 350)
