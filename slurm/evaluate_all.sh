#!/bin/bash
#SBATCH --job-name=rlhf-eval
#SBATCH --partition=h200x4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/lustre/nvwulf/scratch/nijjohnson/logs/rlhf-misspec/eval_%A_%a.out
#SBATCH --error=/lustre/nvwulf/scratch/nijjohnson/logs/rlhf-misspec/eval_%A_%a.err
#SBATCH --array=0-29

set -euo pipefail

module load cuda12.8/toolkit/12.8.0
conda activate rlhf-misspec

export HF_HOME=/lustre/nvwulf/scratch/nijjohnson/hf_cache

BIAS_TYPES=("length" "agreement")
LAMBDA_VALUES=(0.0 0.1 0.3 0.5 1.0)
SEEDS=(42 123 456)

BIAS_IDX=$(( SLURM_ARRAY_TASK_ID / 15 ))
REMAINING=$(( SLURM_ARRAY_TASK_ID % 15 ))
LAMBDA_IDX=$(( REMAINING / 3 ))
SEED_IDX=$(( REMAINING % 3 ))

BIAS_TYPE=${BIAS_TYPES[$BIAS_IDX]}
LAMBDA_VALUE=${LAMBDA_VALUES[$LAMBDA_IDX]}
SEED=${SEEDS[$SEED_IDX]}

COMPLETIONS=results/completions/${BIAS_TYPE}_bias/lambda_${LAMBDA_VALUE}_seed_${SEED}.jsonl
METRICS_DIR=results/metrics/${BIAS_TYPE}_bias

echo "Evaluating: bias=${BIAS_TYPE}, lambda=${LAMBDA_VALUE}, seed=${SEED}"

# Task accuracy
python scripts/04_evaluate_task.py \
    --completions_file ${COMPLETIONS} \
    --ground_truth_file results/data/ground_truth.jsonl \
    --output_file ${METRICS_DIR}/task_accuracy_lambda_${LAMBDA_VALUE}_seed_${SEED}.json

# Behavioral metrics (with NLI model)
python scripts/05_evaluate_behavioral.py \
    --completions_file ${COMPLETIONS} \
    --output_file ${METRICS_DIR}/behavioral_lambda_${LAMBDA_VALUE}_seed_${SEED}.json

echo "Done."
