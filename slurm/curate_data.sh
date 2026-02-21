#!/bin/bash
#SBATCH --job-name=rlhf-curate
#SBATCH --partition=debug-h200x4
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/lustre/nvwulf/scratch/nijjohnson/logs/rlhf-misspec/curate_%j.out
#SBATCH --error=/lustre/nvwulf/scratch/nijjohnson/logs/rlhf-misspec/curate_%j.err

set -euo pipefail

module load cuda12.8/toolkit/12.8.0
conda activate rlhf-misspec

export HF_HOME=/lustre/nvwulf/scratch/nijjohnson/hf_cache
export HF_TOKEN=$(cat ${HOME}/.cache/huggingface/token)

echo "Curating prompt suite..."

python scripts/01_curate_prompts.py \
    --output_dir results/data/ \
    --prompts_per_domain 330

echo "Done."
