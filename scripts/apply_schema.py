"""M1: apply db/schema.sql to the database in DATABASE_URL and report pgvector/halfvec support."""
import os
import pathlib
import sys

import psycopg
from dotenv import load_dotenv

load_dotenv()
url = os.environ.get("DATABASE_URL")
print("DATABASE_URL set:", bool(url))
if not url:
    sys.exit(1)

sql = (pathlib.Path(__file__).parent.parent / "db" / "schema.sql").read_text(encoding="utf-8")
with psycopg.connect(url, connect_timeout=30, autocommit=True) as conn:
    conn.execute(sql)
    ver = conn.execute("SELECT extversion FROM pg_extension WHERE extname='vector'").fetchone()[0]
    print("pgvector version:", ver)
    conn.execute("SELECT '[1,2,3]'::halfvec(3) <=> '[1,2,4]'::halfvec(3)")
    print("halfvec works: True")
    tables = [r[0] for r in conn.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY 1")]
    print("tables:", tables)
