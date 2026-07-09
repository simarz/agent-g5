from engine import generate
from datasets import load_dataset
import random

def main():
    test_data = load_dataset("openai/gsm8k", "main", splti="train")
    question_number = random.randrange(len(test_data))
    q = test_data[question_number]["question"]
    print(generate(q))

if __name__ == "__main__":
    main()