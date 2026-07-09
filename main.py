from datasets import load_dataset
from data import load_problems
from data import Problem

def main():
    probs = load_problems("validation")
    print(len(probs))          # expect 1034
    p = probs[0]
    print(p.question)
    print(p.gold_sql)
    print(p.db_id, p.db_path.exists())


if __name__ == "__main__":
    main()