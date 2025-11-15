import random
import csv
from pathlib import Path

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_addition_dataset(n, path):
    rows = []
    for _ in range(n):
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        prompt = f"What is {a} + {b}? Answer:"
        answer = str(a + b)
        rows.append({"prompt": prompt, "answer": answer})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "answer"])
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    generate_addition_dataset(4000, DATA_DIR / "addition_train.csv")
    generate_addition_dataset(800, DATA_DIR / "addition_test.csv")