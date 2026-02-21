# RLHF Reward Misspecification Experiment

## Abstract

Reinforcement learning from human feedback (RLHF) has become the dominant paradigm for aligning large language models with human intent, yet the reward models that drive this optimization are imperfect proxies for what we actually want. This project investigates a simple but consequential question: when a reward signal contains a small, systematic bias — such as a mild preference for longer responses or for agreeing with the user — how does policy optimization amplify that bias into observable changes in model behavior?

We introduce controlled reward misspecifications of the form `reward = task_score + λ * bias_feature`, where `λ` governs the strength of a known bias injected alongside a legitimate task reward. By training a 7B-parameter language model under varying bias intensities across six prompt domains (coding, math, factual QA, advice, opinion, and creative writing), we construct dose-response curves that quantify how behavioral metrics — response length, hedging frequency, agreement rate, and others — shift as a function of misspecification strength. Crucially, we measure these shifts separately for objective domains (where correctness is verifiable) and subjective domains (where the model has more latitude to drift), revealing that the same reward bias produces markedly different behavioral signatures depending on task structure.

Our experimental design holds all other variables constant — same base model, same prompts, same sampling — so that any measured drift is attributable to the reward signal alone. This controlled setup allows us to characterize reward misspecification not as a binary failure mode, but as a graded phenomenon with predictable, domain-dependent dynamics.

## Research Question

If a reward model has a plausible bias (e.g., preference for longer responses, agreement with user assertions), how does GRPO-style policy optimization amplify it? The central claim: mild bias in reward signals produces **predictable, domain-dependent** shifts in learned behavior, even when base model capability is unchanged.

## Experimental Design

**Formula**: `reward = task_score + λ × bias_feature`

| Variable | Values |
|----------|--------|
| Base model | Qwen2.5-7B-Instruct + LoRA |
| Algorithm | GRPO (TRL GRPOTrainer) |
| Bias types | Length, Agreement (Phase 1); Politeness + Mitigation (Phase 2) |
| Lambda (λ) | 0.0, 0.1, 0.3, 0.5, 1.0 |
| Seeds | 42, 123, 456 |
| Total runs | 30 (Phase 1) + ~20 (Phase 2) |

**Prompt suite**: ~2000 prompts balanced across 6 domains:
- **Objective**: Coding (HumanEval/MBPP), Math (GSM8K), Factual QA (TriviaQA/NQ)
- **Subjective**: Advice, Opinion, Creative Writing (UltraFeedback, HH-RLHF)

**Behavioral metrics**: Length distribution, hedging frequency, agreement rate, politeness markers, refusal rate, self-contradiction, task accuracy, LLM-judge scores.

**Controls**: Length-matched (isolate length effects) and reward-matched (normalize reward magnitude).

## Setup

### Local Development (macOS)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### NVWulf Cluster

```bash
module load cuda12.8/toolkit/12.8.0
conda env create -f environment.yml
conda activate rlhf-misspec
```

## Usage

### Local Development Pipeline

```bash
# Full dev pipeline (60 prompts, 20 training steps)
make dev

# Individual steps
make curate-dev     # Build 60-prompt mini-suite
make train-dev      # Train 20 GRPO steps on MPS
make generate-dev   # Generate completions
make evaluate-dev   # Run behavioral metrics
```

### Production Pipeline (Cluster)

```bash
# Step 1: Build prompt suite
make curate
# or: sbatch slurm/curate_data.sh

# Step 2: Train all 30 conditions
make train
# Submits: sbatch slurm/train_sweep.sh (array job 0-29)

# Step 3: Generate completions from checkpoints
make generate
# Submits: sbatch slurm/generate_completions.sh

# Step 4: Evaluate
make evaluate
# Submits: sbatch slurm/evaluate_all.sh

# Step 5: Aggregate and visualize (can run locally)
make aggregate
make visualize
make controls
```

### Running a Single Condition

```bash
# Local
python scripts/02_train_grpo.py \
    --config configs/experiment/length_bias.yaml \
    --lambda_value 0.3 --seed 42 \
    --output_dir results/checkpoints/test --dev

# Cluster
sbatch slurm/train_single.sh  # Set BIAS_TYPE, LAMBDA_VALUE, SEED env vars
```

## Project Structure

```
rlhf-playground/
├── configs/              # YAML configs (experiment, model, accelerate)
├── src/
│   ├── data/             # Prompt curation and dataset building
│   ├── rewards/          # Task rewards + bias reward functions
│   ├── training/         # GRPOTrainer setup and callbacks
│   ├── evaluation/       # Task accuracy + behavioral metrics
│   └── utils/            # Device detection, logging, seeds, config
├── scripts/              # Numbered entry points (01-07)
├── slurm/                # SLURM job scripts for NVWulf
├── notebooks/            # EDA only (disposable)
├── visualization/        # Reads saved results → publication figures
├── results/              # All outputs (gitignored)
└── tests/                # Unit and integration tests
```

## Pipeline Data Flow

```
Source datasets → [01_curate] → prompt_suite.jsonl
    → [02_train] x 30 → LoRA checkpoints
    → [03_generate] x 30 → completion files
    → [04_evaluate_task] → accuracy JSONs
    → [05_evaluate_behavioral] → behavioral JSONs
    → [06_aggregate] → summary tables
    → [visualization/] → publication figures
```

## Key Design Decisions

- **GRPO over PPO**: Native custom reward function support, no value model overhead, vLLM integration. PPO is being deprecated in TRL.
- **DeepSpeed ZeRO-2 (not ZeRO-3)**: ZeRO-3 has compatibility issues with LoRA adapter toggling in GRPO's reference model handling.
- **Keyword-based agreement during training**: Fast, no extra GPU memory. Full NLI model reserved for evaluation accuracy.
- **Proxy task rewards during training**: Format/pattern checking (not code execution or symbolic math) keeps training fast. True accuracy measured post-training.
- **Visualization separated from analysis**: Figures regenerated without re-running experiments.

## Reproducibility

- All random seeds fixed (Python, NumPy, PyTorch)
- Exact prompt suite published as `results/data/prompt_suite.jsonl`
- All configs version-controlled
- Pinned dependencies in `requirements.txt`
- SLURM scripts fully specify compute environment

## Tests

```bash
make test               # Run all tests
make test-rewards       # Reward function contract tests only
```
