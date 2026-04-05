"""
utils.py
--------
Shared utility helpers: data loading, visualisation, metrics pretty-printing,
and model persistence (save / load).
"""

import os
import pickle
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from wordcloud import WordCloud
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Colour palette ───────────────────────────────────────────────────────────
SENTIMENT_COLORS = {
    "positive": "#2ecc71",   # green
    "negative": "#e74c3c",   # red
    "neutral":  "#3498db",   # blue
}


# ── Data helpers ─────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load a CSV dataset and perform basic validation.

    Expected columns: 'text', 'sentiment'

    Parameters
    ----------
    filepath : str – Path to the CSV file

    Returns
    -------
    pd.DataFrame
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath)
    logger.info(f"Loaded {len(df):,} rows from '{filepath}'")

    # Basic validation
    required_cols = {"text", "sentiment"}
    missing = required_cols - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    # Normalise column names and sentiment labels
    df.columns = df.columns.str.lower()
    df["sentiment"] = df["sentiment"].str.lower().str.strip()

    # Drop empty text rows
    before = len(df)
    df = df.dropna(subset=["text", "sentiment"])
    df = df[df["text"].str.strip() != ""]
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} rows with missing/empty values.")

    logger.info(f"Class distribution:\n{df['sentiment'].value_counts().to_string()}")
    return df.reset_index(drop=True)


def encode_labels(series: pd.Series) -> tuple:
    """
    Encode string sentiment labels to integers.

    Returns
    -------
    (encoded_array, label_map_dict)
    e.g. {"positive": 2, "neutral": 1, "negative": 0}
    """
    label_map = {label: idx for idx, label in enumerate(sorted(series.unique()))}
    encoded   = series.map(label_map).values
    return encoded, label_map


# ── Model persistence ────────────────────────────────────────────────────────

def save_artifact(obj, filepath: str):
    """Pickle an object (model, vectoriser, label_map, etc.) to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Saved artifact → {filepath}")


def load_artifact(filepath: str):
    """Load a pickled object from disk."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Artifact not found: {filepath}")
    with open(filepath, "rb") as f:
        obj = pickle.load(f)
    logger.info(f"Loaded artifact ← {filepath}")
    return obj


# ── Metric helpers ───────────────────────────────────────────────────────────

def print_metrics(y_true, y_pred, model_name: str = "Model", label_names: list = None):
    """Pretty-print accuracy + full sklearn classification report."""
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'─'*50}")
    print(f"  {model_name}")
    print(f"{'─'*50}")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print()
    print(classification_report(y_true, y_pred, target_names=label_names, zero_division=0))
    return acc


# ── Visualisation helpers ────────────────────────────────────────────────────

def plot_sentiment_distribution(df: pd.DataFrame, save_path: str = None):
    """Bar chart of class distribution with percentage labels."""
    counts = df["sentiment"].value_counts()
    colors = [SENTIMENT_COLORS.get(s, "#95a5a6") for s in counts.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white",
                  linewidth=1.5, width=0.55)

    # Percentage annotations
    total = counts.sum()
    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.005,
            f"{val:,}\n({val/total*100:.1f}%)",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.set_title("Sentiment Distribution", fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Sentiment Class", fontsize=12)
    ax.set_ylabel("Number of Tweets",  fontsize=12)
    ax.set_ylim(0, counts.max() * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved plot → {save_path}")
    plt.show()


def plot_wordcloud(df: pd.DataFrame, sentiment: str, save_path: str = None):
    """Generate and display a word-cloud for the given sentiment class."""
    subset = df[df["sentiment"] == sentiment]["cleaned_text"].dropna()
    if subset.empty:
        logger.warning(f"No cleaned text found for sentiment='{sentiment}'.")
        return

    text   = " ".join(subset)
    color  = SENTIMENT_COLORS.get(sentiment, "#555")

    wc = WordCloud(
        width=800, height=400,
        background_color="white",
        colormap="RdYlGn" if sentiment == "positive" else
                 "Reds"   if sentiment == "negative" else "Blues",
        max_words=150,
        collocations=False,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Word Cloud — {sentiment.capitalize()} Tweets",
                 fontsize=15, fontweight="bold", color=color, pad=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved word cloud → {save_path}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, label_names: list,
                          model_name: str = "Model", save_path: str = None):
    """Heatmap confusion matrix using sklearn's ConfusionMatrixDisplay."""
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved confusion matrix → {save_path}")
    plt.show()


def plot_model_comparison(results: dict, save_path: str = None):
    """
    Horizontal bar chart comparing accuracy across multiple models.

    Parameters
    ----------
    results : dict  e.g. {"Logistic Regression": 0.87, "Naive Bayes": 0.83}
    """
    models     = list(results.keys())
    accuracies = [results[m] for m in models]
    colors     = ["#2ecc71" if acc == max(accuracies) else "#3498db"
                  for acc in accuracies]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(models, [a * 100 for a in accuracies],
                   color=colors, edgecolor="white", height=0.45)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_width() - 1, bar.get_y() + bar.get_height() / 2,
                f"{acc*100:.2f}%", va="center", ha="right",
                color="white", fontsize=11, fontweight="bold")

    ax.set_xlim(0, 105)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend: best model highlight
    best_patch = mpatches.Patch(color="#2ecc71", label="Best Model")
    ax.legend(handles=[best_patch], loc="lower right", fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
        logger.info(f"Saved model comparison → {save_path}")
    plt.show()


# ── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df   = load_data(os.path.join(base, "data", "tweets.csv"))
    print(df.head())
