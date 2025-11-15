# Mechanistic Interpretability Research Project Guide

This guide provides a **clear, actionable roadmap** for launching a mechanistic interpretability research project on small/mid-scale open-source models using VSCode, GitHub, Google Colab, and Hugging Face.

***

## 1. Project Overview
Mechanistic interpretability aims to reverse-engineer neural networks, uncovering how specific neurons, layers, or circuits implement computations. Your goals are:
- Study GPT-2, Qwen-3B, and DeepSeek-VL2
- Run all experiments on a laptop or Colab
- Discover and validate new circuits/features
- Achieve workshop-publishable results
- Address AGI-relevant questions: alignment, internal representations, emergent reasoning

***

## 2. Requirements
### **Technical Prerequisites:**
- Familiarity with Python, PyTorch, and Jupyter/Colab
- Basic neural networks and transformer background
- Experience with VSCode and GitHub workflows

### **Core Tools:**
- **VSCode**: Code and document locally
- **GitHub**: Version control, collaboration, sharing results
- **Google Colab**: Free GPU/TPU compute for running notebooks
- **Hugging Face Hub**: Source of models and datasets
- **TransformerLens, SAE-Lens, ACDC**: For mechanistic analysis

***

## 3. Project Structure
Create a GitHub repository (e.g. `mech-interp-research`) like:
```
mech-interp-research/
├─ notebooks/           # Colab & Jupyter notebooks for experiments
├─ scripts/             # Supporting Python scripts (e.g., for data handling)
├─ data/                # Sample data or rerun outputs
├─ results/             # Figures, circuit diagrams, tables, evaluation results
├─ README.md            # Project overview, setup guide, milestones
└─ requirements.txt     # All library dependencies
```

***

## 4. Step-by-Step Action Plan

### **Step 1: Background Reading & Problem Selection**
- Read the [Mechanistic Interpretability Quickstart Guide], [Starter Templates], and skim the "200 Open Problems" list to choose a concrete, tractable research question (e.g., "How does GPT-2 represent arithmetic?", "What circuits implement sentence boundary detection in Qwen-3B?").[1][2]
- Skim recent reviews and thesis papers for ideas and best practices.[3][5][7][8]

### **Step 2: Set Up Your Environment**
- Clone your new repo to VSCode. Install Python and Docker (optional).
- Add core libraries to `requirements.txt`:
  - `transformers`, `datasets`, `transformer_lens`, `sae-lens`, `torch`, `acdc`
- Create a first `notebooks/` experiment file (or use a Colab template).

### **Step 3: Initial Experiments**
- Load a small model (GPT-2 or Qwen-3B) from Hugging Face in Colab:
  ```python
  from transformers import AutoModelForCausalLM, AutoTokenizer
  model = AutoModelForCausalLM.from_pretrained("gpt2")
  tokenizer = AutoTokenizer.from_pretrained("gpt2")
  ```
- Run a mechanistic analysis tool (e.g. TransformerLens) to extract activations, attention maps:
  ```python
  from transformer_lens import HookedTransformer
  hooked_model = HookedTransformer.from_pretrained("gpt2")
  # Explore circuits with built-in visualization tools
  ```
- Try out SAE-Lens/Sparse Autoencoders for feature analysis.
- Always document code/workflows as you go in Markdown cells or `README.md`.

### **Step 4: Project Execution & Discovery**
1. **Circuit Discovery:** Try basic circuit tracing, interventions, and ablations. Look for meaningful input-output patterns.
2. **Feature Analysis:** Use sparse autoencoders to identify interpretable neurons/features and test which inputs activate them.
3. **Ablation Studies:** Temporarily disable certain layers/components to check behavioral changes.
4. **Generalization:** Test whether your discoveries hold on other checkpoints or datasets.
5. **Result Logging:** Save your results/plots to `results/`; write brief summaries in README or a dedicated `REPORT.md`.

### **Step 5: Evaluation and Validation**
- Check the **faithfulness** and **completeness** of your circuit explanations (does your circuit capture the model's behavior? Is anything missing?)[10]
- Set up reproducibility: create scripts/notebooks others can easily run.
- Compare your findings to open problems or previous literature.

### **Step 6: Documentation & Publication**
- Clean up your repo: organize code, add clear instructions, and polish `README.md` with key findings and next steps.
- Write up results in a workshop-ready format (use Hackathon/ML Safety workshop templates, include figures and code snippets).
- Optionally, share on Hugging Face, submit to an open workshop, or draft an arXiv preprint.

***

## 5. Example Milestone Timeline
| Week | Milestone Description                                |
|------|-----------------------------------------------------|
| 1    | Learn basics, read guides, pick research question   |
| 2-3  | Experiment setup; replicate toy mechanistic analyses|
| 4-5  | Discover circuits/features, log findings            |
| 6    | Run ablation/intervention experiments, generalize   |
| 7    | Evaluate results, refine research question          |
| 8    | Write up, share on GitHub, submit to workshop       |

***

## 6. Key Tips for Impact
- **Start small**: Choose a focused, tractable problem—don't chase scale early.
- **Reproducibility matters**: Document everything (code, environment, results).
- **Prioritize clarity**: Clean repo layout and Markdown documentation are highly valued.
- **Engage the community**: Share progress, ask for feedback, connect with interpretability researchers (e.g., on Alignment Forum, LessWrong).
- **Aim for AGI relevance**: Frame your findings in terms of safety, alignment, representations, or reasoning.

***

## 7. Handy Resources
- [Mechanistic Interpretability Quickstart Guide][2]
- [Interpretability Starter Templates][1]
- [TransformerLens Documentation](https://docs.transformerlens.org/)
- [Sparse Autoencoder Tools (SAE-Lens)](https://github.com/jacobcd52/sae-lens)
- [200 Concrete Open Problems in MI](https://www.lesswrong.com/s/5gMf6mWD6tvmYoitJ)
- [Awesome Mechanistic Interpretability](https://github.com/itsqyh/Awesome-LMMs-Mechanistic-Interpretability)

***

## 8. Closing Checklist
- [ ] Project initialized on GitHub, integrated with VSCode
- [ ] Models, code, Colab notebook working
- [ ] Research question chosen, relevant papers/guides read
- [ ] Core experiment run (feature/circuit discovery)
- [ ] Ablation/intervention results analyzed
- [ ] Results documented, repo cleaned up
- [ ] Writeup ready for publication/workshop sharing

***

If you want help scoping a specific problem or want a detailed example pipeline, let me know your chosen model/problem! I can tailor this for your course level or publication target.

[1](https://github.com/apartresearch/interpretability-starter)
[2](https://www.alignmentforum.org/posts/jLAvJt8wuSFySN975/mechanistic-interpretability-quickstart-guide)
[3](https://dspace.mit.edu/bitstream/handle/1721.1/156787/liao-iliao-meng-eecs-2024-thesis.pdf?sequence=1&isAllowed=y)
[4](https://www.lesswrong.com/posts/jP9KDyMkchuv6tHwm/how-to-become-a-mechanistic-interpretability-researcher)
[5](https://arxiv.org/html/2404.14082v1)
[6](https://proceedings.neurips.cc/paper_files/paper/2024/file/90e73f3cf1a6c84c723a2e8b7fb2b2c1-Paper-Conference.pdf)
[7](https://leonardbereska.github.io/blog/2024/mechinterpreview/)
[8](https://dspace.mit.edu/bitstream/handle/1721.1/156748/lad-vedang-meng-eecs-2024-thesis.pdf?sequence=1&isAllowed=y)
[9](https://www.neelnanda.io/mechanistic-interpretability/getting-started-old)
[10](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/final-projects/SachalSohanSrivastavaMalick.pdf)

### Checklist of what you will get in this answer

1. A single `PROJECT_PLAN.md` you can drop into your repo.
2. Clear goals and success criteria so you know what “impactful” means.
3. Exact tools and accounts you need, tied to VS Code + GitHub + Colab + Hugging Face.
4. Step by step execution plan from setup to publishable results.
5. A 10–12 week timeline with milestones.
6. Risks, assumptions, and dependencies called out explicitly.

You can copy everything below into a file named `PROJECT_PLAN.md`.

````markdown
# Mechanistic Interpretability Project Plan  
## Circuit Discovery in Small Language Models

Author: Raja Hussain  
Status: Draft  
Last updated: TODO

---

## 1. Vision and Impact

### 1.1 Core Goal

Build and publish a mechanistic interpretability study that:

- Identifies a specific computation inside a small language model (for example two digit addition).  
- Localizes this computation to a small set of layers and heads.  
- Demonstrates **causal** evidence that these components implement the behavior.  
- Compares behavior across at least one other small model (for example Qwen-3B or a small DeepSeek variant).  

This should be:

- **AGI relevant**: contributes to understanding how reasoning circuits arise in transformers.  
- **Publishable**: suitable for a workshop, student venue, or serious research portfolio.  
- **Executable on student hardware**: laptop plus Google Colab.  

### 1.2 What “Impactful” Means

This project counts as impactful if, by the end, you have:

1. A public GitHub repo with clean code, experiments, and plots.  
2. A written report (8 to 12 pages) with clear methods and results.  
3. At least one nontrivial finding that is not already in the literature, such as:  
   - A previously undocumented circuit for a structured task in GPT-2 small.  
   - Evidence that different models reuse a similar structural pattern for the same task.  
4. A clear story you can tell in applications:  
   - “I isolated and causally tested a reasoning circuit in GPT-2 and replicated patterns in another model.”

---

## 2. Research Problem

### 2.1 Candidate Research Question

> How does a small transformer model (GPT-2 small) implement two digit addition, and can we identify and causally validate specific heads and layers that form an addition circuit

You can refine this later to a different structured task (indirect object identification, bracket matching, simple logic), but start with addition because it is simple and quantifiable.

### 2.2 Hypothesis

- There exist a small number of attention heads and layers whose activations are both:  
  - Predictive of correct addition behavior.  
  - Causally necessary for correct outputs, in the sense that patching or ablating them reliably changes the answer.

---

## 3. Assumptions, Dependencies, and Risks

### 3.1 Assumptions

- You have consistent access to:  
  - A laptop with VS Code, git, and Python.  
  - Stable internet access.  
- You are comfortable with basic Python and can learn PyTorch as needed.  
- You can use Google Colab for GPU access.  

### 3.2 Dependencies

- **Accounts**:  
  - GitHub (for code and version control).  
  - Hugging Face (for models and possibly dataset hosting).  
  - Google account for Colab.  
- **Core libraries**:  
  - `torch`, `transformers`, `transformer-lens`, `numpy`, `pandas`, `matplotlib`.  

### 3.3 Risks and Mitigations

- **Risk 1: Model or task is too hard.**  
  - Mitigation: start with very small tasks and few digit ranges, then scale gradually.  
- **Risk 2: Activation patterns are noisy and hard to interpret.**  
  - Mitigation: use simple structured prompts and many repeated trials. Focus on relative comparisons between correct vs incorrect runs.  
- **Risk 3: Time slip and unfinished project.**  
  - Mitigation: enforce weekly milestones, even if results are imperfect. Negative or partial results are still valuable if documented.  

---

## 4. Technical Stack

### 4.1 Tools

- **Editor**: VS Code.  
- **Version control**: git + GitHub.  
- **Compute**:  
  - Local CPU for development and small runs.  
  - Google Colab GPU for heavier experiments.  
- **Models** (via Hugging Face or direct support in TransformerLens):  
  - `gpt2` or `gpt2-small` as the primary model.  
  - A second small model, for example a Qwen small or DeepSeek text model, for cross model comparison.  

### 4.2 Libraries (initial `requirements.txt`)

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
````

You can add more only when needed.

---

## 5. Repository Structure

Target layout:

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
    02_activation_exploration.ipynb
    03_activation_patching.ipynb

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
```

---

## 6. Step by Step Plan

### Step 0: Repo and Environment Setup

1. Create GitHub repo `mechanistic-interpretability-gpt2`.
2. Clone into VS Code.
3. Add `requirements.txt` with the list above.
4. Create folders as in the structure section.
5. Create and activate a virtual environment locally, install requirements, and confirm imports.
6. Commit and push.

**Success criteria**: you can `import torch`, `import transformers`, `from transformer_lens import HookedTransformer` without errors.

---

### Step 1: Model and Prompt Sanity Checks

Create `notebooks/01_sanity_checks.ipynb`.

Tasks:

1. Load GPT-2 small using either `transformers` or `HookedTransformer.from_pretrained("gpt2-small")`.
2. Run simple prompts and print outputs:

   * “What is 23 + 54 Answer:”
   * “What is 10 + 11 Answer:”
   * “What is 99 + 1 Answer:”
3. Inspect whether the model ever answers correctly for simple sums.

**Output**:

* A few printed examples and short notes in the notebook about how the model behaves.

---

### Step 2: Dataset Creation

Create `src/data_generation.py`.

Goals:

* Generate a dataset of two digit addition prompts and answers.
* Save to disk in `data/raw/addition.jsonl` or `data/raw/addition.csv`.

Key design decisions:

* Input format: `"What is {a} + {b}? Answer:"`
* Answer format: `"37"` etc.
* Dataset sizes:

  * Train: 2,000 to 5,000 examples.
  * Test: 500 to 1,000 examples.
* Avoid leaking test examples into any in-context examples later.

**Success criteria**:

* You can load the dataset and see a small sample in a notebook.
* There is a clear split between train and test.

---

### Step 3: Baseline Behavior Evaluation

Create `src/eval_baseline.py` and `notebooks/02_activation_exploration.ipynb` to visualize.

Tasks:

1. For each test example:

   * Tokenize the prompt.
   * Generate a short completion.
   * Parse the predicted answer (extract digits at the end).
2. Compute accuracy: number of correct answers divided by total.
3. Log additional information:

   * Which sums it tends to get wrong.
   * Whether it is biased toward particular answers.

**Metrics**:

* Overall accuracy on test set.
* Accuracy by sum size bucket (for example sums between 20 and 50, 50 and 100, etc).

**Success criteria**:

* You have a simple table or plot showing baseline performance.
* The notebook documents these observations.

---

### Step 4: Activation Collection

Create `src/activations.py`.

Using TransformerLens:

1. Load GPT-2 small as a `HookedTransformer`.
2. For a batch of prompts, run `run_with_cache` to get activations.
3. Store:

   * Residual stream activations for selected layers.
   * Attention head outputs for selected layers.

For example:

* Cache layer 4, 5, 6 and their attention head outputs.
* Store them in memory for a small batch, then add code to save to disk if needed.

**Analysis in notebook**:

* Compare average activations between correct and incorrect examples.
* Identify heads where the difference is large.

**Success criteria**:

* You can plot some simple statistics per head (for example mean activation magnitude for correct vs incorrect).
* You identify a shortlist of candidate heads and layers to investigate.

---

### Step 5: Circuit Discovery Strategy

Create `src/patching.py` and expand `notebooks/03_activation_patching.ipynb`.

You will use two main techniques:

1. **Ablation**: zeroing out or randomizing a head or layer and measuring the drop in accuracy.
2. **Activation patching**: replacing activations from a corrupted example with activations from a clean example.

Procedure:

1. Select 1 or 2 layers and a small subset of heads per layer.
2. For each head:

   * Run ablation across a batch of test examples and measure accuracy change.
   * For a selected pair of (clean, corrupted) examples, perform activation patching and see if the answer changes.
3. Rank heads by:

   * How much accuracy they destroy when ablated.
   * How much they fix corrupted runs when patched.

**Success criteria**:

* You identify a handful of heads that strongly affect addition performance.
* You have plots that show performance with and without those heads.

---

### Step 6: Causal Validation

Refine patching experiments:

1. Define standardized prompt pairs:

   * Clean: example where the model is correct.
   * Corrupted: example where the model is incorrect, but similar in structure.
2. For each candidate head:

   * Perform many patching experiments over different example pairs.
   * Measure how often patching flips the answer from wrong to right.
3. Aggregate statistics:

   * Patch success rate per head.
   * Confidence intervals if you can compute them.

**Goal**:

Provide evidence like:

* “Head 5.3, when patched from clean runs, corrects 40 percent of previously wrong answers on this subset, while other heads rarely have this effect.”

This is the core of your mechanistic claim.

---

### Step 7: Cross Model Comparison

Once GPT-2 path is working:

1. Select a second model that is small enough to run on Colab.
2. Repeat a light version of steps 3 to 6:

   * Baseline evaluation on the same addition dataset.
   * Simple ablations and a few patching experiments.
3. Compare:

   * Which layers seem important.
   * Whether the head count or layer depth of critical components is similar.

**Success criteria**:

* You can say something specific about whether the second model uses a similar or different structural pattern for addition.

---

### Step 8: Evaluation, Robustness, and Extensions

If time allows:

* Vary the prompt wording and check robustness of the circuit.
* Test extrapolation: sums slightly outside the training range.
* Check whether the same heads are important for closely related tasks (for example subtraction or increment by 1).

You do not need all of this for a first paper, but any extra robustness checks increase credibility.

---

## 7. Timeline and Milestones (10 to 12 Weeks)

This assumes roughly 6 to 8 hours per week.

* **Week 1**

  * Set up repo, environment, and accounts.
  * Run sanity checks on GPT-2 small.

* **Week 2**

  * Implement dataset creation and save train/test splits.
  * Finish baseline evaluation script.

* **Week 3**

  * Complete baseline evaluation and analysis notebook.
  * Choose final task formulation and locking of dataset.

* **Week 4**

  * Implement activation collection for a subset of layers.
  * Build first plots that compare activations across correct vs incorrect runs.

* **Week 5**

  * Implement ablation for a small set of heads.
  * Identify top candidate heads and layers.

* **Week 6**

  * Implement activation patching for pairs of examples.
  * Run initial causal experiments and refine code.

* **Week 7**

  * Systematically run patching experiments and log results.
  * Produce tables and graphs with patch success rates.

* **Week 8**

  * Start cross model comparison on a second model.
  * Run shortened baseline and ablation tests.

* **Week 9**

  * Draft results section: baseline, ablations, patching, cross model notes.
  * Polish figures and captions.

* **Week 10 to 11**

  * Write introduction, related work, methods.
  * Integrate code and experimental logs into a clean repo structure.

* **Week 12**

  * Final pass on paper or report.
  * Prepare GitHub README that explains the project clearly.

You can compress or extend this depending on class load and exams.

---

## 8. Publication and Communication Plan

### 8.1 Artifacts

By the end you should have:

* Public GitHub repo with:

  * Clear README, environment setup, and usage instructions.
  * Scripts in `src/` that can reproduce baseline and key experiments.
  * Notebooks with figures and analysis.
* Written report:

  * Introduction and motivation.
  * Task and dataset description.
  * Model and methods.
  * Results with plots and tables.
  * Discussion and limitations.

### 8.2 Use Cases

* Attach repo and report in research applications or transfer essays.
* Mention specific findings when talking to faculty or labs:

  * “I implemented activation patching and discovered a set of heads in GPT-2 small that are causally linked to two digit addition behavior.”

---

## 9. Daily and Weekly Execution Rules

* Always end the week with something committed that runs, even if it is small.
* Never keep experiments only in Colab. Sync important code back into `src/` and commit.
* Keep a running `reports/notes.md` log of what you tried, what worked, and what failed.
* Prioritize:

  1. Getting a full baseline pipeline working.
  2. Getting one solid causal result.
  3. Only then polishing and adding extra complexity.

---

## 10. Summary

This project is designed so that:

* It fits your current tools: VS Code, GitHub, Colab, Hugging Face.
* It produces concrete, measurable results.
* It gives you a real, defensible story as “the freshman who found and tested a reasoning circuit in GPT-2.”

Follow the steps in order, protect the weekly milestones, and treat the repo as your lab notebook.

```

The next move after you save this as `PROJECT_PLAN.md` is simple: create the repo with this file, add `requirements.txt`, and run the first sanity check notebook that loads GPT-2 small and prints a few completions on addition prompts. That unlocks everything else.
::contentReference[oaicite:0]{index=0}
```


| Model        | Domain      | Example Experiment                 | AGI Relevance         |
| ------------ | ----------- | ---------------------------------- | --------------------- |
| GPT-2        | Text        | Circuit tracing for arithmetic     | Reasoning foundations |
| Qwen-3B      | Text        | Feature analysis by sparse AE      | Scaling/robustness    |
| DeepSeek-VL2 | Images+Text | Multimodal circuit/attention study | AGI-level integration |