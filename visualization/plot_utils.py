"""Shared plotting utilities and styling."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# Publication-quality defaults
STYLE_CONFIG = {
    "figure.figsize": (8, 5),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 2,
    "lines.markersize": 8,
}

# Color palette for bias types
BIAS_COLORS = {
    "length": "#1f77b4",
    "agreement": "#ff7f0e",
    "politeness": "#2ca02c",
    "mitigated": "#9467bd",
}

# Color palette for domains
DOMAIN_COLORS = {
    "coding": "#1f77b4",
    "math": "#ff7f0e",
    "qa": "#2ca02c",
    "advice": "#d62728",
    "opinion": "#9467bd",
    "creative": "#8c564b",
}

LAMBDA_VALUES = [0.0, 0.1, 0.3, 0.5, 1.0]


def setup_style():
    """Apply publication-quality matplotlib style."""
    plt.rcParams.update(STYLE_CONFIG)
    sns.set_palette("colorblind")
    sns.set_style("whitegrid")


def save_figure(fig, output_dir: str | Path, name: str, formats=("pdf", "png")):
    """Save figure in multiple formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        filepath = output_dir / f"{name}.{fmt}"
        fig.savefig(filepath, format=fmt, bbox_inches="tight")
    plt.close(fig)
