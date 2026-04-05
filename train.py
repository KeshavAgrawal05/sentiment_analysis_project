"""
train.py
--------
Model training pipeline for Twitter sentiment analysis.

Workflow:
  1. Load and preprocess data
  2. TF-IDF feature extraction
  3. Train Logistic Regression & Multinomial Naive Bayes
  4. Evaluate both and select the best
  5. Save model bundle (vectoriser + best model + label map) to disk
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

# ── Allow imports from project root ─────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import TextPreprocessor
from src.utils import (
    load_data,
    encode_labels,
    save_artifact,
    print_metrics,
    plot_sentiment_distribution,
    plot_wordcloud,
    plot_confusion_matrix,
    plot_model_comparison,
    logger,
)

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(PROJECT_ROOT, "data",  "tweets.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "model.pkl")


# ── Feature engineering ──────────────────────────────────────────────────────

def build_tfidf_vectorizer(
    max_features: int = 10_000,
    ngram_range: tuple = (1, 2),
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """
    Instantiate a TF-IDF vectoriser with sensible defaults.

    Parameters
    ----------
    max_features : int   – Vocabulary cap (default 10 000)
    ngram_range  : tuple – Unigrams + bigrams (1,2) by default
    sublinear_tf : bool  – Apply log-normalisation to term frequencies

    Returns
    -------
    TfidfVectorizer (unfitted)
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=2,          # ignore tokens appearing in < 2 documents
        strip_accents="unicode",
        analyzer="word",
    )


# ── Model catalogue ──────────────────────────────────────────────────────────

def get_models() -> dict:
    """
    Return a dict of {model_name: unfitted_estimator}.

    Both models work well with sparse TF-IDF matrices:
    - LogisticRegression: strong linear baseline with L2 regularisation
    - MultinomialNB: fast probabilistic model suited for word-count features
    """
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1_000,
            C=1.0,
            solver="lbfgs",
            multi_class="auto",
            random_state=42,
        ),
        "Naive Bayes": MultinomialNB(
            alpha=0.5,   # Laplace smoothing factor
        ),
    }


# ── Main training function ───────────────────────────────────────────────────

def train(
    data_path: str = DATA_PATH,
    model_path: str = MODEL_PATH,
    test_size: float = 0.20,
    random_state: int = 42,
    generate_plots: bool = False,
) -> dict:
    """
    Full end-to-end training pipeline.

    Parameters
    ----------
    data_path     : str   – Path to the CSV dataset
    model_path    : str   – Where to save the best model bundle
    test_size     : float – Fraction of data used for evaluation
    random_state  : int   – Reproducibility seed
    generate_plots: bool  – Whether to render visualisation plots

    Returns
    -------
    dict with keys: best_model_name, best_accuracy, model_bundle
    """

    # ── 1. Load data ─────────────────────────────────────────────────────────
    logger.info("Step 1/6 — Loading dataset …")
    df = load_data(data_path)

    # ── 2. Preprocess text ───────────────────────────────────────────────────
    logger.info("Step 2/6 — Preprocessing text …")
    preprocessor = TextPreprocessor(remove_stops=True, lemmatize=True)
    df["cleaned_text"] = preprocessor.clean_series(df["text"])

    # Drop rows where cleaning produced empty strings
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)
    logger.info(f"Usable samples after cleaning: {len(df):,}")

    # ── 3. Optional visualisations ───────────────────────────────────────────
    if generate_plots:
        logger.info("Generating data visualisations …")
        plots_dir = os.path.join(PROJECT_ROOT, "plots")
        plot_sentiment_distribution(df, save_path=os.path.join(plots_dir, "sentiment_distribution.png"))
        for sentiment in df["sentiment"].unique():
            plot_wordcloud(df, sentiment,
                           save_path=os.path.join(plots_dir, f"wordcloud_{sentiment}.png"))

    # ── 4. Encode labels & split ─────────────────────────────────────────────
    logger.info("Step 3/6 — Encoding labels and splitting data …")
    y, label_map = encode_labels(df["sentiment"])
    label_names  = [k for k, _ in sorted(label_map.items(), key=lambda x: x[1])]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["cleaned_text"], y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    logger.info(f"Train: {len(X_train_raw):,}  |  Test: {len(X_test_raw):,}")

    # ── 5. TF-IDF vectorisation ──────────────────────────────────────────────
    logger.info("Step 4/6 — Fitting TF-IDF vectoriser …")
    vectorizer = build_tfidf_vectorizer()
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test  = vectorizer.transform(X_test_raw)
    logger.info(f"Vocabulary size: {len(vectorizer.vocabulary_):,} features")

    # ── 6. Train & evaluate models ───────────────────────────────────────────
    logger.info("Step 5/6 — Training and evaluating models …")
    models      = get_models()
    results     = {}
    predictions = {}

    for model_name, model in models.items():
        logger.info(f"  → Training {model_name} …")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = print_metrics(y_test, y_pred,
                            model_name=model_name,
                            label_names=label_names)
        results[model_name]     = acc
        predictions[model_name] = y_pred

        # 5-fold cross-validation on the training set
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
        logger.info(
            f"  {model_name} — 5-fold CV: "
            f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}"
        )

        if generate_plots:
            plots_dir = os.path.join(PROJECT_ROOT, "plots")
            plot_confusion_matrix(
                y_test, y_pred, label_names,
                model_name=model_name,
                save_path=os.path.join(plots_dir, f"cm_{model_name.lower().replace(' ', '_')}.png"),
            )

    # Compare models
    if generate_plots:
        plots_dir = os.path.join(PROJECT_ROOT, "plots")
        plot_model_comparison(results, save_path=os.path.join(plots_dir, "model_comparison.png"))

    best_name = max(results, key=results.get)
    best_acc  = results[best_name]
    logger.info(f"\n🏆  Best model: {best_name}  (accuracy={best_acc:.4f})")

    # ── 7. Save best model bundle ────────────────────────────────────────────
    logger.info("Step 6/6 — Saving model bundle …")
    model_bundle = {
        "vectorizer":   vectorizer,
        "model":        models[best_name],
        "label_map":    label_map,        # {"negative": 0, "neutral": 1, "positive": 2}
        "label_names":  label_names,
        "preprocessor": preprocessor,
        "best_model_name": best_name,
        "test_accuracy":   best_acc,
    }
    save_artifact(model_bundle, model_path)
    logger.info(f"Model bundle saved to '{model_path}'")

    return {
        "best_model_name": best_name,
        "best_accuracy":   best_acc,
        "all_results":     results,
        "model_bundle":    model_bundle,
    }


# ── Entry-point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Twitter Sentiment Analysis models")
    parser.add_argument("--data",   default=DATA_PATH,  help="Path to dataset CSV")
    parser.add_argument("--model",  default=MODEL_PATH, help="Output model path (.pkl)")
    parser.add_argument("--plots",  action="store_true", help="Generate visualisation plots")
    args = parser.parse_args()

    result = train(
        data_path=args.data,
        model_path=args.model,
        generate_plots=args.plots,
    )
    print(f"\n✅  Training complete!  Best model: {result['best_model_name']}"
          f"  |  Accuracy: {result['best_accuracy']*100:.2f}%")
