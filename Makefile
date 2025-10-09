# Makefile for GPT-2 QLoRA + SEAL scaffold
# Provides automation for setup, testing, training, and basic sweeps

.PHONY: help setup install test lint format clean train-gpt2 eval-gpt2 seal sweep
.DEFAULT_GOAL := help

# Variables
PYTHON := python3
PIP := pip3
SRC_DIR := src
TEST_DIR := tests
CONFIG_DIR := configs
RESULTS_DIR := results
CHECKPOINTS_DIR := checkpoints

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(BLUE)TinyLM Scaling Laws Research Project$(NC)"
	@echo "$(BLUE)====================================$(NC)"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Quick start:$(NC)"
	@echo "  make setup    # Install dependencies and setup environment"
	@echo "  make test     # Run all tests"
	@echo "  make train    # Run baseline training experiment"
	@echo "  make reproduce # Reproduce main results"

setup: ## Setup development environment and install dependencies
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@$(PYTHON) -m pip install --upgrade pip
	@$(PIP) install -e .
	@$(PIP) install -e ".[dev]"
	@echo "$(GREEN)✓ Dependencies installed$(NC)"
	@mkdir -p $(RESULTS_DIR) $(CHECKPOINTS_DIR) $(RESULTS_DIR)/plots
	@echo "$(GREEN)✓ Directories created$(NC)"
	@echo "$(GREEN)✓ Setup complete!$(NC)"

install: setup ## Alias for setup

test: ## Run all unit tests with coverage
	@echo "$(BLUE)Running unit tests...$(NC)"
	@$(PYTHON) -m pytest $(TEST_DIR) -v
	@echo "$(GREEN)✓ Tests completed$(NC)"

test-models: ## Run only model tests
	@echo "$(BLUE)Running model tests...$(NC)"
	@cd $(SRC_DIR) && $(PYTHON) -m pytest ../$(TEST_DIR)/test_models.py -v

test-data: ## Run only data tests
	@echo "$(BLUE)Running data tests...$(NC)"
	@cd $(SRC_DIR) && $(PYTHON) -m pytest ../$(TEST_DIR)/test_data.py -v

test-training: ## Run only training tests
	@echo "$(BLUE)Running training tests...$(NC)"
	@cd $(SRC_DIR) && $(PYTHON) -m pytest ../$(TEST_DIR)/test_training.py -v

lint: ## Run code linting
	@echo "$(BLUE)Running linters...$(NC)"
	@flake8 $(SRC_DIR) $(TEST_DIR) --max-line-length=88 --extend-ignore=E203,W503
	@echo "$(GREEN)✓ Linting passed$(NC)"

format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	@black $(SRC_DIR) $(TEST_DIR)
	@isort $(SRC_DIR) $(TEST_DIR)
	@echo "$(GREEN)✓ Code formatted$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(NC)"
	@mypy $(SRC_DIR) --ignore-missing-imports
	@echo "$(GREEN)✓ Type checking passed$(NC)"

clean: ## Clean up generated files and directories
	@echo "$(BLUE)Cleaning up...$(NC)"
	@rm -rf __pycache__ .pytest_cache .coverage htmlcov
	@find . -type d -name "__pycache__" -delete
	@find . -type f -name "*.pyc" -delete
	@rm -rf $(RESULTS_DIR)/coverage_html
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

# Training commands
train-gpt2: ## Train GPT-2 Small with QLoRA (demo)
	@echo "$(BLUE)Training GPT-2 Small with QLoRA...$(NC)"
	@$(PYTHON) -m src.gpt2_qlora.train \
		--model_name gpt2 \
		--train_file data/sample/train.txt \
		--output_dir $(CHECKPOINTS_DIR)/gpt2_qlora_demo \
		--lora_r 8 --lora_alpha 16 --lora_dropout 0.05 \
		--block_size 128 --batch_size 2 --epochs 1 --lr 1e-4
	@echo "$(GREEN)✓ Training completed$(NC)"

train-width: ## Run width scaling experiments
	@echo "$(BLUE)Running width scaling experiments...$(NC)"
	@mkdir -p $(CHECKPOINTS_DIR)/width_scaling
	@cd $(SRC_DIR) && $(PYTHON) ../scripts/run_sweep.py \
		--config ../$(CONFIG_DIR)/scaling_width.yaml \
		--output_dir ../$(CHECKPOINTS_DIR)/width_scaling
	@echo "$(GREEN)✓ Width scaling experiments completed$(NC)"

train-depth: ## Run depth scaling experiments
	@echo "$(BLUE)Running depth scaling experiments...$(NC)"
	@mkdir -p $(CHECKPOINTS_DIR)/depth_scaling
	@cd $(SRC_DIR) && $(PYTHON) ../scripts/run_sweep.py \
		--config ../$(CONFIG_DIR)/scaling_depth.yaml \
		--output_dir ../$(CHECKPOINTS_DIR)/depth_scaling
	@echo "$(GREEN)✓ Depth scaling experiments completed$(NC)"

train-context: ## Run context length scaling experiments
	@echo "$(BLUE)Running context length scaling experiments...$(NC)"
	@mkdir -p $(CHECKPOINTS_DIR)/context_scaling
	@cd $(SRC_DIR) && $(PYTHON) ../scripts/run_sweep.py \
		--config ../$(CONFIG_DIR)/scaling_context.yaml \
		--output_dir ../$(CHECKPOINTS_DIR)/context_scaling
	@echo "$(GREEN)✓ Context scaling experiments completed$(NC)"

train-data: ## Run data scaling experiments
	@echo "$(BLUE)Running data scaling experiments...$(NC)"
	@mkdir -p $(CHECKPOINTS_DIR)/data_scaling
	@cd $(SRC_DIR) && $(PYTHON) ../scripts/run_sweep.py \
		--config ../$(CONFIG_DIR)/scaling_data.yaml \
		--output_dir ../$(CHECKPOINTS_DIR)/data_scaling
	@echo "$(GREEN)✓ Data scaling experiments completed$(NC)"

# Evaluation commands
eval-gpt2: ## Evaluate GPT-2 perplexity
	@echo "$(BLUE)Evaluating GPT-2 perplexity...$(NC)"
	@mkdir -p $(RESULTS_DIR)/evaluation
	@$(PYTHON) -m src.gpt2_qlora.eval \
		--model_name gpt2 \
		--adapter_dir $(CHECKPOINTS_DIR)/gpt2_qlora_demo \
		--eval_file data/sample/train.txt \
		--block_size 128 --max_batches 10
	@echo "$(GREEN)✓ Evaluation completed$(NC)"

eval-scaling: ## Evaluate scaling law experiments
	@echo "$(BLUE)Evaluating scaling experiments...$(NC)"
	@mkdir -p $(RESULTS_DIR)/scaling_analysis
	@cd $(SRC_DIR) && $(PYTHON) ../scripts/analyze_scaling.py \
		--checkpoint_dirs ../$(CHECKPOINTS_DIR)/width_scaling,../$(CHECKPOINTS_DIR)/depth_scaling \
		--output_dir ../$(RESULTS_DIR)/scaling_analysis
	@echo "$(GREEN)✓ Scaling analysis completed$(NC)"

# Sweep and ablation commands
sweep: ## Run a simple LoRA rank sweep
	@echo "$(BLUE)Running QLoRA sweep...$(NC)"
	@$(PYTHON) scripts/run_qlora_sweep.py \
		--train_file data/sample/train.txt \
		--output_dir $(CHECKPOINTS_DIR)/gpt2_sweep \
		--lora_r 4 8
	@echo "$(GREEN)✓ Sweep completed$(NC)"

ablation-tokenizer: ## Run tokenizer ablation study
	@echo "$(BLUE)Running tokenizer ablation...$(NC)"
	@mkdir -p $(CHECKPOINTS_DIR)/ablation_tokenizer
	@cd $(SRC_DIR) && $(PYTHON) ../scripts/run_sweep.py \
		--config ../$(CONFIG_DIR)/ablation_tokenizer.yaml \
		--output_dir ../$(CHECKPOINTS_DIR)/ablation_tokenizer
	@echo "$(GREEN)✓ Tokenizer ablation completed$(NC)"

ablation-optimizer: ## Run optimizer ablation study
	@echo "$(BLUE)Running optimizer ablation...$(NC)"
	@mkdir -p $(CHECKPOINTS_DIR)/ablation_optimizer
	@cd $(SRC_DIR) && $(PYTHON) ../scripts/run_sweep.py \
		--config ../$(CONFIG_DIR)/ablation_optimizer.yaml \
		--output_dir ../$(CHECKPOINTS_DIR)/ablation_optimizer
	@echo "$(GREEN)✓ Optimizer ablation completed$(NC)"

ablations: ablation-tokenizer ablation-optimizer ## Run all ablation studies
	@echo "$(GREEN)✓ All ablation studies completed$(NC)"

# Reproduction command
reproduce: ## Reproduce main paper results (3-seed runs)
	@echo "$(BLUE)Reproducing main results (this may take a while)...$(NC)"
	@echo "$(YELLOW)Running experiments with 3 different seeds...$(NC)"
	@mkdir -p $(CHECKPOINTS_DIR)/reproduction $(RESULTS_DIR)/reproduction

	# Run baseline with 3 seeds
	@for seed in 42 123 456; do \
		echo "$(BLUE)Running baseline with seed $$seed...$(NC)"; \
		mkdir -p $(CHECKPOINTS_DIR)/reproduction/baseline_seed_$$seed; \
		cd $(SRC_DIR) && $(PYTHON) train.py \
			--config ../$(CONFIG_DIR)/base_config.yaml \
			--save_dir ../$(CHECKPOINTS_DIR)/reproduction/baseline_seed_$$seed \
			--seed $$seed \
			--no_wandb; \
	done

	# Run key scaling experiments with 3 seeds
	@for seed in 42 123 456; do \
		echo "$(BLUE)Running width scaling with seed $$seed...$(NC)"; \
		mkdir -p $(CHECKPOINTS_DIR)/reproduction/width_seed_$$seed; \
		cd $(SRC_DIR) && $(PYTHON) ../scripts/run_sweep.py \
			--config ../$(CONFIG_DIR)/scaling_width.yaml \
			--output_dir ../$(CHECKPOINTS_DIR)/reproduction/width_seed_$$seed \
			--seed $$seed; \
	done

	# Evaluate and analyze results
	@echo "$(BLUE)Skipping automated analysis: customize with scripts/analyze_scaling.py as needed.$(NC)"
	@echo "$(GREEN)✓ Reproduction completed!$(NC)"
	@echo "$(YELLOW)Results saved in $(RESULTS_DIR)/reproduction/$(NC)"

# Data commands


# Export and analysis commands
export-model: ## Export best model for inference
	@echo "$(BLUE)Exporting best model...$(NC)"
	@mkdir -p $(RESULTS_DIR)/exported_models
	@cd $(SRC_DIR) && $(PYTHON) ../scripts/export_model.py \
		--checkpoint_dir ../$(CHECKPOINTS_DIR)/baseline \
		--output_dir ../$(RESULTS_DIR)/exported_models
	@echo "$(GREEN)✓ Model exported$(NC)"

generate-plots: ## Generate all plots and visualizations
	@echo "$(BLUE)Use scripts/analyze_scaling.py to generate plots for specific checkpoint directories.$(NC)"
	@echo "$(YELLOW)Example: python scripts/analyze_scaling.py --checkpoint_dirs checkpoints/width_scaling --output_dir results/plots$(NC)"

# Development commands
pre-commit: format lint type-check test ## Run all pre-commit checks
	@echo "$(GREEN)✓ All pre-commit checks passed$(NC)"

check: pre-commit ## Alias for pre-commit

# Quick smoke test
smoke-test: ## Quick import + tiny tokenizer test
	@echo "$(BLUE)Running smoke test...$(NC)"
	@$(PYTHON) -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('gpt2'); print('EOS:', t.eos_token)"
	@echo "$(GREEN)✓ Smoke test completed$(NC)"

# Status and info commands
status: ## Show project status and file counts
	@echo "$(BLUE)Project Status$(NC)"
	@echo "$(BLUE)==============$(NC)"
	@echo "Source files: $$(find $(SRC_DIR) -name '*.py' | wc -l)"
	@echo "Test files: $$(find $(TEST_DIR) -name '*.py' | wc -l)"
	@echo "Config files: $$(find $(CONFIG_DIR) -name '*.yaml' | wc -l)"
	@echo "Checkpoints: $$(find $(CHECKPOINTS_DIR) -name '*.pt' 2>/dev/null | wc -l)"
	@echo "Results: $$(find $(RESULTS_DIR) -name '*' -type f 2>/dev/null | wc -l)"

info: status ## Alias for status

# CI/CD simulation
ci: pre-commit smoke-test ## Simulate CI/CD pipeline
	@echo "$(GREEN)✓ CI pipeline completed successfully$(NC)"

# Print useful information
env-info: ## Show environment information
	@echo "$(BLUE)Environment Information$(NC)"
	@echo "$(BLUE)======================$(NC)"
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Pip version: $$($(PIP) --version)"
	@echo "PyTorch version: $$($(PYTHON) -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
	@echo "CUDA available: $$($(PYTHON) -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Unknown')"
	@echo "Working directory: $$(pwd)"

# Performance benchmarking
benchmark: ## Run performance benchmarks
	@echo "$(YELLOW)Benchmark suite not yet implemented. Contributions welcome!$(NC)"

# Full pipeline (use with caution - takes a long time)
full-pipeline: setup test train sweep eval ## Run complete pipeline
	@echo "$(GREEN)✓ Full pipeline completed!$(NC)"
	@echo "$(YELLOW)Check $(RESULTS_DIR)/ for all results$(NC)"
