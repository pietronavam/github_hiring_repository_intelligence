"""
Visualization

Each chart is chosen to answer a specific analytical question:
  1. Label distribution — checks for class imbalance before modeling
  2. Signal distributions per category — reveals which features discriminate classes
  3. Confusion matrix — shows where the model confuses similar categories
  4. Model comparison bar chart — contextualizes BERT's improvement over baselines
  5. Feature correlation heatmap — identifies multicollinearity and redundant signals
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.utils import CATEGORIES

LABELED_PATH = Path("data/labeled/repositories_labeled.csv")
PROCESSED_PATH = Path("data/processed/repositories_clean.csv")
METRICS_DIR = Path("output/metrics")
TABLES_DIR = Path("output/tables")
FIGURES_DIR = Path("output/figures")

PALETTE = {
    "intern": "#6c757d",
    "junior": "#17a2b8",
    "senior": "#28a745",
    "lead": "#007bff",
    "template": "#ffc107",
    "low_value": "#dc3545",
}


def plot_label_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    counts = df["label"].value_counts().reindex(CATEGORIES).fillna(0)
    colors = [PALETTE.get(c, "#999") for c in counts.index]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(int(val)), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title("Repository Category Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "label_distribution.png", dpi=150)
    plt.close()


def plot_signal_boxplots(df: pd.DataFrame) -> None:
    signals = ["stars", "contributors", "commits_30d", "engineering_score", "readme_length", "releases"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax, signal in zip(axes, signals):
        data = [df[df["label"] == cat][signal].dropna().values for cat in CATEGORIES]
        bp = ax.boxplot(data, labels=CATEGORIES, patch_artist=True, showfliers=False)
        for patch, cat in zip(bp["boxes"], CATEGORIES):
            patch.set_facecolor(PALETTE.get(cat, "#999"))
            patch.set_alpha(0.7)
        ax.set_title(signal.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Signal Distributions by Engineering Maturity Category", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "signal_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix() -> None:
    cm_path = METRICS_DIR / "confusion_matrix.npy"
    labels_path = METRICS_DIR / "label_names.json"
    if not cm_path.exists():
        return

    cm = np.load(cm_path)
    with open(labels_path) as f:
        label_names = json.load(f)

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", xticklabels=label_names,
                yticklabels=label_names, linewidths=0.5, ax=ax, cbar_kws={"label": "Recall"})
    ax.set_title("Confusion Matrix — DistilBERT (test set)", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def plot_model_comparison() -> None:
    comp_path = METRICS_DIR / "model_comparison.csv"
    if not comp_path.exists():
        return

    df = pd.read_csv(comp_path)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(df))
    width = 0.35
    bars1 = ax.bar([i - width / 2 for i in x], df["accuracy"], width, label="Accuracy", color="#007bff", alpha=0.8)
    bars2 = ax.bar([i + width / 2 for i in x], df["f1_weighted"], width, label="F1 (weighted)", color="#28a745", alpha=0.8)

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["model"], fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Comparison: Baselines vs DistilBERT", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
    plt.close()


def plot_feature_correlation(df: pd.DataFrame) -> None:
    numeric_cols = ["stars", "forks", "contributors", "commits_30d", "has_ci",
                    "has_tests", "readme_length", "releases", "age_days",
                    "days_since_push", "engineering_score", "topics_count"]
    cols = [c for c in numeric_cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "feature_correlation.png", dpi=150)
    plt.close()


def generate_all(labeled_path: Path = LABELED_PATH, processed_path: Path = PROCESSED_PATH) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df_labeled = pd.read_csv(labeled_path).dropna(subset=["label"])
    df_processed = pd.read_csv(processed_path)

    # Merge labels into processed for signal analysis
    df_merged = df_processed.merge(df_labeled[["full_name", "label", "confidence"]], on="full_name", how="inner")
    df_merged = df_merged[df_merged["confidence"] >= 0.6]

    plot_label_distribution(df_merged)
    plot_signal_boxplots(df_merged)
    plot_confusion_matrix()
    plot_model_comparison()
    plot_feature_correlation(df_merged)

    print(f"All figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    generate_all()
