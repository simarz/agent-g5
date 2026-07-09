"""Throwaway exploration: read a Spider database's schema, then run its gold query."""

import sqlite3
from pathlib import Path

DB_ROOT = Path("spider_data") / "spider_data" / "database"
db_id = "department_management"
db_path = DB_ROOT / db_id / f"{db_id}.sqlite"
print("db_path:", db_path, "| exists:", db_path.exists(), "\n")

# read-only connection: the same call the verifier will need for untrusted SQL
conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

# --- Step 1: the schema lives in the built-in sqlite_master table ---
rows = conn.execute(
    "SELECT name, sql FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'"
).fetchall()

print(f"=== {len(rows)} tables ===\n")
for name, create_sql in rows:
    print(create_sql, "\n")

# --- Step 2: execute the gold query for this example ---
gold_sql = "SELECT count(*) FROM head WHERE age  >  56"
result = conn.execute(gold_sql).fetchall()
print("=== gold query ===")
print(gold_sql)
print("result:", result)

conn.close()
