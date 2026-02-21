# Results Directory

This directory holds all experiment outputs. Subdirectories are gitignored.

## Structure

```
results/
├── data/              # Curated prompt suite and ground truth
│   ├── prompt_suite.jsonl
│   ├── ground_truth.jsonl
│   └── domain_counts.json
├── checkpoints/       # LoRA adapter weights per condition
│   └── {bias_type}/lambda_{λ}_seed_{s}/
├── completions/       # Generated text from trained models
│   └── {bias_type}/lambda_{λ}_seed_{s}.jsonl
├── metrics/           # Per-condition evaluation results
│   └── {bias_type}/
│       ├── task_accuracy_lambda_{λ}_seed_{s}.json
│       ├── behavioral_lambda_{λ}_seed_{s}.json
│       └── training_log_lambda_{λ}_seed_{s}.jsonl
├── aggregated/        # Cross-condition summary tables
│   ├── summary_table.csv
│   ├── pivot_{metric}.csv
│   ├── drift_scores.csv
│   └── correlation_matrix.csv
└── figures/           # Publication-ready plots
    ├── dose_response_*.pdf
    ├── length_dist_*.pdf
    ├── heatmap_*.pdf
    └── accuracy_*.pdf
```
