#!/bin/bash
#SBATCH --job-name=rlhf-misspec-sweep
#SBATCH --partition=h200x4
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=/lustre/nvwulf/scratch/nijjohnson/logs/rlhf-misspec/train_%A_%a.out
#SBATCH --error=/lustre/nvwulf/scratch/nijjohnson/logs/rlhf-misspec/train_%A_%a.err
#SBATCH --array=0-29

# Phase 1: 2 bias types x 5 lambdas x 3 seeds = 30 jobs

set -euo pipefail

module load cuda12.8/toolkit/12.8.0
conda activate rlhf-misspec

export HF_HOME=/lustre/nvwulf/scratch/nijjohnson/hf_cache
export HF_TOKEN=$(cat ${HOME}/.cache/huggingface/token)

# Decode array task ID -> (bias_type, lambda, seed)
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

SCRATCH=/lustre/nvwulf/scratch/nijjohnson/rlhf-misspec
OUTPUT_DIR=${SCRATCH}/checkpoints/${BIAS_TYPE}_bias/lambda_${LAMBDA_VALUE}_seed_${SEED}

echo "================================================================"
echo "Job: bias=${BIAS_TYPE}, lambda=${LAMBDA_VALUE}, seed=${SEED}"
echo "Output: ${OUTPUT_DIR}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "================================================================"

mkdir -p ${SCRATCH}/logs

accelerate launch \
    --config_file configs/accelerate/deepspeed_zero2.yaml \
    --num_processes 4 \
    scripts/02_train_grpo.py \
    --config configs/experiment/${BIAS_TYPE}_bias.yaml \
    --lambda_value ${LAMBDA_VALUE} \
    --seed ${SEED} \
    --output_dir ${OUTPUT_DIR} \
    --prompt_file results/data/prompt_suite.jsonl

echo "Training complete for bias=${BIAS_TYPE}, lambda=${LAMBDA_VALUE}, seed=${SEED}"
