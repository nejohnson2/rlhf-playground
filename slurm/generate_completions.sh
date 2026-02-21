#!/bin/bash
#SBATCH --job-name=rlhf-generate
#SBATCH --partition=h200x4
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/lustre/nvwulf/scratch/nijjohnson/logs/rlhf-misspec/gen_%A_%a.out
#SBATCH --error=/lustre/nvwulf/scratch/nijjohnson/logs/rlhf-misspec/gen_%A_%a.err
#SBATCH --array=0-29

set -euo pipefail

module load cuda12.8/toolkit/12.8.0
conda activate rlhf-misspec

export HF_HOME=/lustre/nvwulf/scratch/nijjohnson/hf_cache
export HF_TOKEN=$(cat ${HOME}/.cache/huggingface/token)

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
CHECKPOINT_DIR=${SCRATCH}/checkpoints/${BIAS_TYPE}_bias/lambda_${LAMBDA_VALUE}_seed_${SEED}
OUTPUT_FILE=results/completions/${BIAS_TYPE}_bias/lambda_${LAMBDA_VALUE}_seed_${SEED}.jsonl

echo "Generating: bias=${BIAS_TYPE}, lambda=${LAMBDA_VALUE}, seed=${SEED}"

python scripts/03_generate_completions.py \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --prompt_file results/data/prompt_suite.jsonl \
    --output_file ${OUTPUT_FILE} \
    --num_samples 5 \
    --max_new_tokens 512 \
    --temperature 0.0 \
    --batch_size 16

echo "Done."
