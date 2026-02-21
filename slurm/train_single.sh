#!/bin/bash
#SBATCH --job-name=rlhf-single
#SBATCH --partition=h200x4
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=/lustre/nvwulf/scratch/nijjohnson/logs/rlhf-misspec/train_single_%j.out
#SBATCH --error=/lustre/nvwulf/scratch/nijjohnson/logs/rlhf-misspec/train_single_%j.err

# Usage: BIAS_TYPE=length LAMBDA_VALUE=0.3 SEED=42 sbatch slurm/train_single.sh

set -euo pipefail

module load cuda12.8/toolkit/12.8.0
conda activate rlhf-misspec

export HF_HOME=/lustre/nvwulf/scratch/nijjohnson/hf_cache
export HF_TOKEN=$(cat ${HOME}/.cache/huggingface/token)

BIAS_TYPE=${BIAS_TYPE:-length}
LAMBDA_VALUE=${LAMBDA_VALUE:-0.3}
SEED=${SEED:-42}

SCRATCH=/lustre/nvwulf/scratch/nijjohnson/rlhf-misspec
OUTPUT_DIR=${SCRATCH}/checkpoints/${BIAS_TYPE}_bias/lambda_${LAMBDA_VALUE}_seed_${SEED}

echo "Training: bias=${BIAS_TYPE}, lambda=${LAMBDA_VALUE}, seed=${SEED}"

accelerate launch \
    --config_file configs/accelerate/deepspeed_zero2.yaml \
    --num_processes 4 \
    scripts/02_train_grpo.py \
    --config configs/experiment/${BIAS_TYPE}_bias.yaml \
    --lambda_value ${LAMBDA_VALUE} \
    --seed ${SEED} \
    --output_dir ${OUTPUT_DIR} \
    --prompt_file results/data/prompt_suite.jsonl

echo "Done."
