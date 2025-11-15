### Quick checklist of what I’ll do here

1. Judge your current markdown for clarity and impact.
2. Point out the main problems (duplication, rough edges, stray junk).
3. Give you a **clean, final** `PROJECT_PLAN.md` that merges everything coherently.
4. Make sure it is directly usable in a GitHub repo with no edits.
5. End with one concrete next step for you.

---

## Is your current version “good”?

Conceptually: yes.
Structurally: messy.

You have two overlapping documents mashed together:

* A high level **“Mechanistic Interpretability Research Project Guide”**
* A deep, stepwise **“Mechanistic Interpretability Project Plan / Circuit Discovery in Small LMs”**

Plus:

* Some duplicated promises (two “checklist of what you will get” sections)
* A stray `::contentReference[oaicite:0]{index=0}` artifact
* Mixed citation styles and some placeholder reference text

So the ideas are strong, but you want **one clean, focused file** that:

* Starts with the big picture
* Specifies one concrete research problem
* Gives a step by step execution plan and timeline
* Ends with a clear checklist

I merged and cleaned everything for you below.

---

## Final `PROJECT_PLAN.md` (drop this into your repo)

````markdown
# Mechanistic Interpretability Project Plan  
## Circuit Discovery in Small Language Models

Author: Raja Hussain  
Status: Draft  
Last updated: TODO

---

## 0. Quick Summary

This project is a mechanistic interpretability study on small and mid-scale open models such as GPT-2, Qwen-3B, and DeepSeek-VL2. The core objective is to:

- Identify a specific computation (for example two digit addition) inside a transformer model  
- Localize it to a small set of layers and heads or features  
- Demonstrate **causal** evidence via ablations and activation patching  
- Compare behavior across at least one additional model  
- Produce a publishable report and a clean, reproducible GitHub repo

All experiments should be runnable on a laptop plus Google Colab.

---

## 1. Project Overview and Motivation

Mechanistic interpretability is about reverse-engineering neural networks and understanding how specific neurons, heads, or circuits implement computations. For AGI relevance, this matters because:

- It helps us understand **how** models form internal representations and perform reasoning  
- It is foundational for **alignment and safety**, since we cannot control what we do not understand  
- It is unusually accessible to undergraduates with modest compute, especially on small models

**Target models:**

- GPT-2 / GPT-2 small (primary)  
- Qwen-3B (or similar small Qwen variant)  
- DeepSeek-VL2 text backbone or another small DeepSeek language model  

**High level goals:**

- Run all experiments on a laptop plus Colab  
- Discover and validate nontrivial circuits or features  
- Achieve workshop-level results  
- Frame findings around AGI-relevant themes: alignment, internal representations, emergent reasoning

---

## 2. Requirements

### 2.1 Technical Prerequisites

- Comfortable with Python and basic data structures  
- Willing to learn PyTorch fundamentals  
- Comfortable with Jupyter/Colab notebooks  
- Basic understanding of transformers (attention, layers, tokens)  
- Familiarity with VS Code and GitHub (clone, commit, push, branches)

### 2.2 Core Tools and Accounts

- **VS Code** for local development  
- **GitHub** for version control and sharing  
- **Google Colab** for free GPU compute  
- **Hugging Face Hub** for models and (optional) datasets  
- **Mechanistic interpretability libraries**:  
  - `transformer_lens` (HookedTransformer, activation caching, patching)  
  - `sae-lens` (sparse autoencoders, feature analysis, optional)  
  - `acdc` or similar tooling for automated circuit discovery (optional, later)

---

## 3. Research Problem

### 3.1 Core Research Question

> How does a small transformer model (GPT-2 small) implement two digit addition, and can we identify and causally validate specific heads and layers that form an “addition circuit”

Later, this can be extended to other structured tasks (bracket matching, simple logic, sentence boundaries) and to other models (Qwen-3B, DeepSeek variants).

### 3.2 Hypothesis

- There exists a small set of attention heads and layers whose activations are:
  - Predictive of correct addition behavior  
  - Causally necessary in the sense that ablating or patching them reliably changes the model’s answer  

A secondary hypothesis:

- Analogous circuits or structural patterns will appear in at least one other small model (Qwen-3B or a small DeepSeek model), even if not at identical layer and head indices.

---

## 4. Success Criteria and Impact

This project counts as **impactful** if, by the end, you have:

1. A public GitHub repo with:  
   - Clean code for data generation, evaluation, activation collection, and patching  
   - Notebooks with figures and analysis  
   - A clear `README` and this `PROJECT_PLAN`  

2. A written report (8 to 12 pages) that includes:  
   - A precise task definition and dataset  
   - Methods for baseline evaluation, ablations, and patching  
   - At least one nontrivial, well-supported mechanistic finding  

3. At least one concrete mechanistic claim, for example:  
   - “Attention head 5.3 in GPT-2 small is causally involved in two digit addition for this prompt family. Patching its activations from correct runs fixes 30 to 50 percent of previously incorrect runs on the test subset.”

4. A clear story for applications:  
   - “I used mechanistic interpretability tools to discover and causally validate a reasoning circuit in GPT-2, then partially replicated the pattern in another model.”

---

## 5. Repository Structure

Create a GitHub repo, for example `mechanistic-interpretability-gpt2`, with this structure:

```text
mechanistic-interpretability-gpt2/
  README.md
  PROJECT_PLAN.md
  requirements.txt

  data/
    raw/
    processed/

  notebooks/
    01_sanity_checks.ipynb
    02_baseline_eval.ipynb
    03_activation_exploration.ipynb
    04_activation_patching.ipynb

  src/
    __init__.py
    data_generation.py
    eval_baseline.py
    activations.py
    patching.py
    analysis.py
    plots.py

  reports/
    figures/
    draft_paper.md
    notes.md
````

---

## 6. Environment and Tooling Setup

### 6.1 Requirements File

Initial `requirements.txt`:

```text
torch
transformers
transformer-lens
datasets
numpy
pandas
matplotlib
einops
jupyter
```

Add `sae-lens` and any other libraries later as needed.

### 6.2 Local Setup (VS Code)

1. Create the repo on GitHub.

2. Clone locally and open in VS Code.

3. Create a virtual environment and install requirements:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # on macOS/Linux
   # .venv\Scripts\activate   # on Windows

   pip install -r requirements.txt
   ```

4. Confirm imports in a small test script or notebook:

   ```python
   import torch
   from transformer_lens import HookedTransformer
   from transformers import AutoModelForCausalLM, AutoTokenizer
   ```

5. Commit and push.

### 6.3 Colab Workflow

* In a Colab notebook:

  ```python
  !git clone https://github.com/<your-username>/mechanistic-interpretability-gpt2.git
  %cd mechanistic-interpretability-gpt2
  !pip install -r requirements.txt
  ```

* Use notebooks in `notebooks/` for GPU-based experiments, but keep core logic in `src/` and commit back to GitHub regularly.

---

## 7. Step by Step Execution Plan

### Step 0: Model and Prompt Sanity Checks

Notebook: `notebooks/01_sanity_checks.ipynb`

1. Load GPT-2 small:

   ```python
   from transformer_lens import HookedTransformer
   model = HookedTransformer.from_pretrained("gpt2-small")
   ```

2. Try simple prompts:

   * `"What is 23 + 54 Answer:"`
   * `"What is 10 + 11 Answer:"`
   * `"What is 99 + 1 Answer:"`

3. Inspect whether the model sometimes answers correctly and what style of answer it gives.

**Success:** GPT-2 small is loaded successfully and you have a basic feel for how it behaves on arithmetic.

---

### Step 1: Dataset Creation

Script: `src/data_generation.py`
Notebook: `notebooks/02_baseline_eval.ipynb` (for quick inspection)

Goals:

* Generate a dataset of two digit addition problems and answers.
* Save train and test splits to `data/raw/`.

Design:

* Prompt format: `"What is {a} + {b}? Answer:"`
* Answer format: `"{a + b}"` (string)
* Ranges:

  * `a, b` in `[10, 99]` for base experiments
* Sizes:

  * Train: 2,000 to 5,000 examples
  * Test: 500 to 1,000 examples

**Success:** You can load the dataset in a notebook, print a few examples, and clearly see the train/test split.

---

### Step 2: Baseline Behavior Evaluation

Script: `src/eval_baseline.py`
Notebook: `notebooks/02_baseline_eval.ipynb`

Tasks:

1. For each test example:

   * Tokenize the prompt with `model.to_tokens` or HF tokenizer
   * Generate a short completion
   * Extract the predicted answer (digits at the end of the completion)

2. Compute:

   * Overall accuracy
   * Accuracy by sum bucket (for example 20–50, 50–100, 100–150, 150–198)

3. Log common error patterns.

**Output:**

* A table and possibly plots of accuracy by bucket.
* Written notes about biases and mistakes.

**Success:** You have a reproducible baseline evaluation with saved metrics and at least one figure.

---

### Step 3: Activation Collection

Script: `src/activations.py`
Notebook: `notebooks/03_activation_exploration.ipynb`

Using TransformerLens:

1. Load GPT-2 small as a `HookedTransformer`.

2. For a batch of prompts, run:

   ```python
   tokens = model.to_tokens(prompts, prepend_bos=True)
   logits, cache = model.run_with_cache(tokens)
   ```

3. Collect activations for:

   * Residual streams at selected layers
   * Attention head outputs at selected layers

4. For each head, compute simple statistics separately for correct vs incorrect examples (for example mean activation magnitude at specific positions).

**Success:** You can produce per-head statistics and plots that highlight a few heads with strong differences between correct and incorrect runs.

---

### Step 4: Circuit Discovery via Ablation and Patching

Script: `src/patching.py`
Notebook: `notebooks/04_activation_patching.ipynb`

Techniques:

1. **Ablation:**

   * Zero out or randomize a specific head or layer during the forward pass.
   * Measure the change in accuracy on the test set.
   * Rank heads by how much they hurt performance when ablated.

2. **Activation Patching:**

   * Select a “clean” example where the model answers correctly and a “corrupted” example where it fails.
   * Run the model on both, cache activations.
   * For a given head and layer, replace the corrupted activations with clean ones and rerun to see if the answer flips.

Goal:

* Identify a small set of heads and layers that:

  * Strongly reduce accuracy when ablated
  * Often fix wrong answers when patched

**Success:** You have quantitative evidence that a handful of heads are strongly implicated in two digit addition.

---

### Step 5: Causal Validation at Scale

Extend patching experiments:

1. Define a protocol for generating many clean/corrupt pairs with similar structure.
2. For each candidate head:

   * Run patching across many pairs
   * Compute the fraction of previously incorrect answers that become correct
3. Compare candidate heads and rank them by patch success rate.

**Target:** A statement like:

* “Head 5.3, when patched from clean runs, corrects 40 percent of previously incorrect answers on this evaluation subset, while other tested heads rarely exceed 5 percent.”

This is your primary mechanistic claim.

---

### Step 6: Cross Model Comparison

Once the GPT-2 pipeline is working:

1. Select a second model:

   * For example a small Qwen model (around 1–3B) or a small DeepSeek language model from Hugging Face.

2. Repeat a shortened version of Steps 2–5:

   * Baseline accuracy
   * Coarse ablation screening
   * A few targeted patching experiments

3. Compare:

   * Which layers and heads appear important
   * Whether the “shape” of the circuit is similar (for example middle layers vs late layers, number of influential heads)

**Success:** You can say something concrete about whether the second model uses a similar or different structure for addition.

---

### Step 7: Robustness Checks and Extensions

If time allows:

* Vary the prompt wording and see whether the same heads remain critical
* Test sums slightly outside the train range (for example up to 199)
* Explore whether the same heads matter for subtraction or increment tasks

These robustness checks strengthen your claims, but are not required for a first workshop-level result.

---

## 8. Timeline and Weekly Milestones (10–12 Weeks)

Assuming 6–8 focused hours per week.

* **Week 1**

  * Repo created, environment set up
  * GPT-2 small loaded and simple prompts tested

* **Week 2**

  * Dataset generation implemented
  * Train/test splits saved and inspected

* **Week 3**

  * Baseline evaluation script working
  * Accuracy metrics and initial plots generated

* **Week 4**

  * Activation collection and basic statistics per head
  * Shortlist of candidate layers and heads

* **Week 5**

  * Ablation experiments implemented and run
  * Top heads identified by performance drop

* **Week 6**

  * Activation patching pipeline implemented
  * Initial patching experiments on hand-picked examples

* **Week 7**

  * Systematic patching at scale and summary statistics
  * Core mechanistic claim drafted in `reports/draft_paper.md`

* **Week 8**

  * Cross model comparison on second model (Qwen or DeepSeek)
  * Shortened ablation and patching experiments run

* **Week 9**

  * Results section drafted: baseline, ablation, patching, cross-model comparison
  * Figures and tables organized into `reports/figures/`

* **Week 10–11**

  * Introduction, related work, and methods written
  * Code cleaned and repo structure finalized

* **Week 12**

  * Final editing of report
  * README updated with a clear project summary and usage instructions

---

## 9. Documentation, Publication, and Communication

### 9.1 Artifacts to Produce

By the end you should have:

* A public GitHub repo with:

  * Clean `README` and `PROJECT_PLAN`
  * Scripts and notebooks that reproduce key results
  * Figures stored under `reports/figures/`

* A written report with:

  * Motivation and background
  * Task and dataset description
  * Detailed methods for ablations and patching
  * Results with plots and tables
  * Discussion of limitations and future work

### 9.2 How to Use This Project for Your Career

* Link the repo and report in research and transfer applications
* When talking to faculty or interviewers, describe:

  * The concrete task
  * The methods you used (activation caching, ablation, patching)
  * The main mechanistic finding
  * Any cross model comparison result

---

## 10. Execution Rules and Checklist

### 10.1 Weekly Execution Rules

* Every week must end with **something that runs** committed to the repo
* Do not keep important code only in Colab
* Maintain `reports/notes.md` as a running experiment log
* Prioritize:

  1. End-to-end pipeline working
  2. One strong causal result
  3. Extensions and polishing

### 10.2 Closing Checklist

* [ ] Repo initialized on GitHub and cloned in VS Code
* [ ] Environment and imports working locally and in Colab
* [ ] Dataset generated and split into train/test
* [ ] Baseline evaluation implemented and plotted
* [ ] Activations collected and candidate heads identified
* [ ] Ablation pipeline implemented and run
* [ ] Activation patching implemented with solid causal results
* [ ] Cross model comparison completed at least at a coarse level
* [ ] Report drafted and GitHub repo cleaned up for public viewing

---

## 11. Key Resources

* Mechanistic interpretability quickstart guides and open problems
* TransformerLens documentation and tutorials
* Sparse autoencoder tooling (SAE-Lens)
* “Awesome Mechanistic Interpretability” style paper lists and repos

Keep this file updated as you refine the problem or add models.


