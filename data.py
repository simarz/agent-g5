from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache
import sqlite3
from datasets import load_dataset

DB_ROOT = Path("spider_data") / "spider_data" / "database"

@dataclass(frozen=True)
class Problem:
    question: str
    db_id: str
    db_path: Path
    schema: list[str]   # the CREATE TABLE strings
    gold_sql: str

def db_path_for(db_id: str) -> Path:
    return DB_ROOT / db_id / f"{db_id}.sqlite"

@lru_cache(maxsize=None)
def schema_for(db_id: str) -> tuple[str, ...]:
    p = db_path_for(db_id)
    conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    rows = conn.execute(
    "SELECT sql FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence' AND sql IS NOT NULL").fetchall()
    conn.close()
    return tuple(row[0] for row in rows)


def load_problems(split: str = "train") -> list[Problem]:
    dataset = load_dataset("xlangai/spider", split=split)
    problems = []
    for ex in dataset:
        db_id = ex["db_id"]
        problems.append(Problem(
            question=ex["question"],
            db_id=db_id,
            db_path= db_path_for(db_id),      
            schema= list(schema_for(db_id)),
            gold_sql=ex["query"]     # ← mind the field-name mismatch
        ))
    return problems

    

