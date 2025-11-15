Got it, good, then Phase 1 starts **after** “repo + tree exist.”

### High level checklist for Phase 1

1. Lock the task and prompt format.
2. Build the two digit addition dataset.
3. Confirm GPT-2 small runs on that dataset.
4. Implement a baseline accuracy script.
5. Save first plots and notes into the repo.

Think of Phase 1 as: **“I can feed GPT-2 my dataset and get a number for how good it is.”** Nothing interpretability yet.

---

## Phase 1: Foundations / Baseline

### Objective

Get a clean, reproducible baseline pipeline for two digit addition using GPT-2 small. By the end, you should be able to run a single command or notebook and see:

* Test accuracy
* Example model outputs
* A basic error pattern summary

After that, every interpretability step has something solid to hook into.

---

### Phase 1.1: Lock the task

In `PROJECT_PLAN.md` or a short `reports/notes.md`, write this explicitly:

* Task: two digit addition
* Input template:
  `"What is {a} + {b}? Answer:"`
* Output:
  `"{a + b}"` (no punctuation, just digits)
* Range:

  * `a, b ∈ [10, 99]`
* Eval: exact match on the numeric string

This sounds trivial, but it forces you not to wander later.

**Deliverable:** 3–5 lines in your notes describing this. That is your spec.

---

### Phase 1.2: Dataset generation

File: `src/data_generation.py`

Core elements:

```python
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
```

Run it once from VS Code terminal:

```bash
python src/data_generation.py
```

**Deliverable:**

* `data/raw/addition_train.csv`
* `data/raw/addition_test.csv`

You can quickly inspect in a notebook:

```python
import pandas as pd
df = pd.read_csv("data/raw/addition_train.csv")
df.head()
```

---

### Phase 1.3: Baseline evaluation script

File: `src/eval_baseline.py`
Notebook: `notebooks/02_baseline_eval.ipynb` to drive it and visualize.

Skeleton logic:

1. Load GPT-2 small via `HookedTransformer` (or HF if you prefer at first).
2. Read `addition_test.csv`.
3. For each row:

   * Feed `prompt` to the model.
   * Generate a short completion, for example `max_new_tokens=4`.
   * Extract the digits at the end as the predicted answer.
4. Compare to `answer`, count correct, compute accuracy.

Pseudocode:

```python
from transformer_lens import HookedTransformer
import torch
import pandas as pd
import re

model = HookedTransformer.from_pretrained("gpt2-small", device="cuda" if torch.cuda.is_available() else "cpu")

def predict_answer(prompt):
    toks = model.to_tokens(prompt, prepend_bos=True)
    logits = model(toks)
    # greedy decode a few tokens
    out = model.to_string(toks[0])
    # you can switch to generate() pattern if needed
    # but for first pass you can try direct logits argmax on final position
    # and gradually refine
    return out

def extract_digits(text):
    m = re.findall(r"\d+", text)
    return m[-1] if m else None

def eval_file(path):
    df = pd.read_csv(path)
    correct = 0
    total = 0
    for _, row in df.iterrows():
        prompt = row["prompt"]
        gold = str(row["answer"])
        pred_text = model.generate(...)  # you will fill this
        pred = extract_digits(pred_text)
        if pred == gold:
            correct += 1
        total += 1
    return correct / total
```

You will need to tune the generation call. That is part of Phase 1: getting a reasonable but not perfect baseline.

**Deliverables:**

* A script that prints something like:
  `Baseline accuracy: 0.12` or `12.3%`
* A notebook that:

  * Shows a small table of prompts, gold answers, predictions
  * Prints the final accuracy

---

### Phase 1.4: Log your findings

In `reports/notes.md`, log:

* Date
* Git commit hash
* Dataset spec (range, size)
* Model version (`gpt2-small`)
* Baseline accuracy
* Any obvious weird behavior

Something short like:

> 2025-11-14
> Commit: `abc1234`
> Task: 2 digit addition, a,b in [10, 99], 4000 train, 800 test
> Model: gpt2-small via HookedTransformer
> Baseline accuracy: 14.5 percent
> Notes: Model often answers with near sums, sometimes repeats last number, etc.

Future you will thank present you for this.

---

## Phase 1 exit criteria

Phase 1 is complete when all of these are true:

* [ ] `data/raw/addition_train.csv` and `data/raw/addition_test.csv` exist and look correct
* [ ] You can run `python src/eval_baseline.py` (or a notebook cell) and get a baseline accuracy
* [ ] At least one notebook shows:

  * A few example predictions
  * The overall accuracy
* [ ] `reports/notes.md` has a dated entry with those results

You already have the repo and tree, so your immediate next move is:

> Implement `src/data_generation.py`, generate the train/test CSVs, and confirm in a notebook that they look right.

Once that is locked in, we promote you to Phase 2, which is where we start touching activations and actually hunting circuits.

# Activate source .venv/bin/activate