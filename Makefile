.PHONY: all dev test clean curate train generate evaluate aggregate visualize controls \
       curate-dev train-dev generate-dev evaluate-dev test-rewards clean-all

PYTHON = python
DEV_MODEL = Qwen/Qwen2.5-0.5B-Instruct
RESULTS_DIR = results
DATA_DIR = $(RESULTS_DIR)/data
METRICS_DIR = $(RESULTS_DIR)/metrics
COMPLETIONS_DIR = $(RESULTS_DIR)/completions
FIGURES_DIR = $(RESULTS_DIR)/figures
AGGREGATED_DIR = $(RESULTS_DIR)/aggregated

# ============================================================
# Full pipeline
# ============================================================
all: curate train generate evaluate aggregate visualize

# ============================================================
# Development (local, sample data, MPS/CPU)
# ============================================================
dev: curate-dev train-dev generate-dev evaluate-dev

curate-dev:
	$(PYTHON) scripts/01_curate_prompts.py \
		--output_dir $(DATA_DIR) \
		--prompts_per_domain 10 \
		--dev

train-dev:
	$(PYTHON) scripts/02_train_grpo.py \
		--config configs/experiment/length_bias.yaml \
		--model $(DEV_MODEL) \
		--lambda_value 0.3 \
		--seed 42 \
		--output_dir $(RESULTS_DIR)/checkpoints/dev \
		--prompt_file $(DATA_DIR)/prompt_suite.jsonl \
		--dev

generate-dev:
	$(PYTHON) scripts/03_generate_completions.py \
		--checkpoint_dir $(RESULTS_DIR)/checkpoints/dev \
		--prompt_file $(DATA_DIR)/prompt_suite.jsonl \
		--output_file $(COMPLETIONS_DIR)/dev_completions.jsonl \
		--num_samples 1 \
		--batch_size 4

evaluate-dev:
	$(PYTHON) scripts/05_evaluate_behavioral.py \
		--completions_file $(COMPLETIONS_DIR)/dev_completions.jsonl \
		--output_file $(METRICS_DIR)/dev_behavioral.json \
		--skip_nli

# ============================================================
# Production (cluster, submitted via SLURM)
# ============================================================
curate:
	$(PYTHON) scripts/01_curate_prompts.py \
		--output_dir $(DATA_DIR) \
		--prompts_per_domain 330

train:
	sbatch slurm/train_sweep.sh

generate:
	sbatch slurm/generate_completions.sh

evaluate:
	sbatch slurm/evaluate_all.sh

# ============================================================
# Analysis (can run locally or on cluster)
# ============================================================
aggregate:
	$(PYTHON) scripts/06_aggregate_results.py \
		--metrics_dir $(METRICS_DIR) \
		--output_dir $(AGGREGATED_DIR)

visualize:
	$(PYTHON) visualization/plot_reward_curves.py \
		--input_dir $(METRICS_DIR) --output_dir $(FIGURES_DIR)
	$(PYTHON) visualization/plot_length_distributions.py \
		--input_dir $(AGGREGATED_DIR) --output_dir $(FIGURES_DIR)
	$(PYTHON) visualization/plot_behavioral_drift.py \
		--input_dir $(AGGREGATED_DIR) --output_dir $(FIGURES_DIR)
	$(PYTHON) visualization/plot_accuracy_degradation.py \
		--input_dir $(AGGREGATED_DIR) --output_dir $(FIGURES_DIR)
	$(PYTHON) visualization/plot_heatmaps.py \
		--input_dir $(AGGREGATED_DIR) --output_dir $(FIGURES_DIR)

controls:
	$(PYTHON) scripts/07_run_controls.py \
		--completions_dir $(COMPLETIONS_DIR) \
		--metrics_dir $(METRICS_DIR) \
		--output_dir $(METRICS_DIR)

# ============================================================
# Testing
# ============================================================
test:
	$(PYTHON) -m pytest tests/ -v

test-rewards:
	$(PYTHON) -m pytest tests/test_reward_functions.py -v

# ============================================================
# Cleanup
# ============================================================
clean:
	rm -rf $(RESULTS_DIR)/checkpoints/dev
	rm -rf $(COMPLETIONS_DIR)/dev*
	rm -rf $(METRICS_DIR)/dev*

clean-all:
	rm -rf $(RESULTS_DIR)/checkpoints
	rm -rf $(COMPLETIONS_DIR)
	rm -rf $(METRICS_DIR)
	rm -rf $(AGGREGATED_DIR)
	rm -rf $(FIGURES_DIR)
