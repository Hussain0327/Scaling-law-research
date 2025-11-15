GPT-2 small does not perform multi digit arithmetic.
It copies the numbers.
It predicts formats, not answers.

Use a dataset from Hugging Face instead of only local CSVs.
Create a Colab notebook that:
clones your repo

installs deps

loads the HF dataset

runs your eval_baseline / activations code on GPU
Connect Colab to your GitHub repo so everything stays in sync.

Make it public + reusable with a Colab badge and clear instructions.

Use a dataset from Hugging Face

Right now you have data/raw/addition_test.csv. For the research project, you can:

Either push your own dataset to HF later

Or start by consuming an existing HF dataset (for example, math/arithmetic) and then eventually publish your own interpretability dataset.

For now, let’s make the code HF-ready.

A. Add datasets to requirements.txt

In requirements.txt:

torch
transformers
transformer-lens
datasets
numpy
pandas
matplotlib
einops
jupyter


Create a Colab notebook wired to your repo

In your repo, add:

notebooks/01_gpt2_mechinterp_colab.ipynb

Inside that notebook, the core cells should be:

Cell 1: Setup (clone repo, install)
# If running on Colab:
!git clone https://github.com/Hussain0327/Scaling-law-research.git
%cd Scaling-law-research

!pip install -r requirements.txt


You run this once per Colab session.

Cell 2: Quick sanity check
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
print("Loaded GPT-2 small on CUDA")

Cell 3: Baseline eval using HF dataset
from src.eval_baseline import load_model, evaluate_from_hf

model = load_model()
acc = evaluate_from_hf(model, max_rows=200)
print(f"HF baseline accuracy on 200 examples: {acc * 100:.2f}%")

Cell 4: Activations analysis
from src.activations import main as activations_main

activations_main()


Keep activations.main() small and deterministic so others can run it and see the same stats.

Commit this notebook to your repo.

3. Connect Colab and GitHub nicely

You now have a notebook in notebooks/. To make it easy for others:

Push to GitHub.

On GitHub, open 01_gpt2_mechinterp_colab.ipynb.

Copy the URL.

Then, in your README.md, add a Colab badge:

## Run on Colab

You can run the core experiments on a free GPU using Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hussain0327/Scaling-law-research/blob/main/notebooks/01_gpt2_mechinterp_colab.ipynb)


Now anyone can click that badge:

Colab opens

It clones your repo

Installs deps

Runs your evaluation and activations on GPU

That is exactly the “public, reusable research notebook” you want.

4. Publishing the research “on Colab”

Practically, the pipeline for you looks like this:

You develop the core logic in src/ in VS Code.

You mirror that in one or more Colab notebooks that:

demonstrate how to run the experiments

visualize key plots

explain the steps in Markdown

You keep notebooks in the repo so they are version-controlled.

People use:

GitHub repo for code

Colab for execution

Hugging Face datasets as the shared data source

Later, if you want the full “research artifact” look, you can:

Push your custom addition dataset or interpretability-processed data to Hugging Face Hub as a dataset.

Link that HF dataset in the notebook and README.

Then your story becomes:

“I built a mechanistic interpretability pipeline in Python using VS Code and GitHub, ran large-scale experiments on Colab GPUs, consumed and published datasets on Hugging Face Hub, and released notebooks that anyone can execute with one click.”

One concrete next step

Do this in order:

Add datasets to requirements.txt and commit.

Create src/datasets_hf.py with a stub load_hf_addition_dataset (we can refine the exact HF dataset choice later).

Add evaluate_from_hf to eval_baseline.py.

Create notebooks/01_gpt2_mechinterp_colab.ipynb with:

git clone + pip install

load model

run evaluate_from_hf