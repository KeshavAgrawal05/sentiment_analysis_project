"""
predict.py
----------
Prediction module for Twitter Sentiment Analysis.

Provides:
  - SentimentPredictor class  (load model, predict single & batch)
  - predict_sentiment()       (convenience function for quick use)
  - CLI entry-point
"""

import os
import sys
import logging

# ── Allow imports from project root ─────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils import load_artifact, logger

# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "model.pkl")


# ── Confidence labels ─────────────────────────────────────────────────────────
CONFIDENCE_LEVELS = {
    (0.80, 1.01): "High",
    (0.60, 0.80): "Medium",
    (0.00, 0.60): "Low",
}

def _confidence_label(prob: float) -> str:
    for (lo, hi), label in CONFIDENCE_LEVELS.items():
        if lo <= prob < hi:
            return label
    return "Low"


# ── Predictor class ───────────────────────────────────────────────────────────

class SentimentPredictor:
    """
    Load a saved model bundle and predict sentiment for new tweet text.

    The bundle is expected to contain:
        vectorizer   : fitted TfidfVectorizer
        model        : fitted classifier
        label_map    : dict {str_label: int_index}
        label_names  : list of str labels (sorted by index)
        preprocessor : TextPreprocessor instance

    Usage
    -----
    >>> predictor = SentimentPredictor()
    >>> predictor.predict("I love this product!")
    {'sentiment': 'positive', 'confidence': 0.94, 'confidence_level': 'High', 'probabilities': {...}}
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'.\n"
                "Run  python src/train.py  first to generate the model."
            )
        bundle = load_artifact(model_path)

        self._vectorizer   = bundle["vectorizer"]
        self._model        = bundle["model"]
        self._label_map    = bundle["label_map"]
        self._label_names  = bundle["label_names"]   # sorted by index
        self._preprocessor = bundle["preprocessor"]
        self._reverse_map  = {v: k for k, v in self._label_map.items()}

        logger.info(
            f"SentimentPredictor loaded  |  model={bundle.get('best_model_name','?')}"
            f"  |  classes={self._label_names}"
        )

    # ── Core predict ──────────────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """
        Predict sentiment for a single tweet.

        Parameters
        ----------
        text : str – Raw tweet text (any length, can contain URLs/mentions)

        Returns
        -------
        dict with keys:
            sentiment        : str  – "positive" | "negative" | "neutral"
            confidence       : float – max class probability [0, 1]
            confidence_level : str  – "High" | "Medium" | "Low"
            probabilities    : dict – {class_label: probability}
            cleaned_text     : str  – preprocessed text used for inference
        """
        if not isinstance(text, str) or not text.strip():
            return {
                "sentiment":        "neutral",
                "confidence":       0.0,
                "confidence_level": "Low",
                "probabilities":    {n: 0.0 for n in self._label_names},
                "cleaned_text":     "",
                "warning":          "Empty or invalid input — defaulting to neutral.",
            }

        # 1. Clean
        cleaned = self._preprocessor.clean(text)

        # 2. Handle edge case: cleaning removed everything meaningful
        if not cleaned.strip():
            cleaned = text.lower()  # fallback: just lowercase original

        # 3. Vectorise
        features = self._vectorizer.transform([cleaned])

        # 4. Get class probabilities (if available) or hard predictions
        if hasattr(self._model, "predict_proba"):
            proba        = self._model.predict_proba(features)[0]
            predicted_idx = int(proba.argmax())
            confidence    = float(proba.max())
            prob_dict     = {
                self._reverse_map[i]: round(float(p), 4)
                for i, p in enumerate(proba)
            }
        else:
            predicted_idx = int(self._model.predict(features)[0])
            confidence    = 1.0
            prob_dict     = {
                n: (1.0 if i == predicted_idx else 0.0)
                for i, n in enumerate(self._label_names)
            }

        sentiment = self._reverse_map[predicted_idx]

        return {
            "sentiment":        sentiment,
            "confidence":       round(confidence, 4),
            "confidence_level": _confidence_label(confidence),
            "probabilities":    prob_dict,
            "cleaned_text":     cleaned,
        }

    def predict_batch(self, texts: list) -> list:
        """
        Predict sentiment for a list of tweets.

        Parameters
        ----------
        texts : list of str

        Returns
        -------
        list of result dicts (same structure as predict())
        """
        return [self.predict(t) for t in texts]

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self):
        return (
            f"SentimentPredictor("
            f"classes={self._label_names}, "
            f"model={self._model.__class__.__name__})"
        )


# ── Convenience function ─────────────────────────────────────────────────────

def predict_sentiment(text: str, model_path: str = DEFAULT_MODEL_PATH) -> dict:
    """
    One-shot convenience wrapper — loads model, predicts, returns result.

    Parameters
    ----------
    text       : str – Tweet text to classify
    model_path : str – Path to the saved model bundle

    Returns
    -------
    dict (see SentimentPredictor.predict)
    """
    predictor = SentimentPredictor(model_path=model_path)
    return predictor.predict(text)


# ── CLI entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Predict sentiment of tweet text.")
    parser.add_argument("text",  nargs="?", help="Tweet text to classify (or use --batch)")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to model.pkl")
    parser.add_argument("--batch", nargs="+", help="Predict multiple tweets at once")
    args = parser.parse_args()

    predictor = SentimentPredictor(model_path=args.model)

    if args.batch:
        results = predictor.predict_batch(args.batch)
        for tweet, res in zip(args.batch, results):
            print(f"\nTweet      : {tweet}")
            print(f"Sentiment  : {res['sentiment'].upper()}  ({res['confidence_level']} confidence: {res['confidence']*100:.1f}%)")
            print(f"Probs      : {json.dumps(res['probabilities'])}")
    elif args.text:
        res = predictor.predict(args.text)
        print(f"\nTweet      : {args.text}")
        print(f"Sentiment  : {res['sentiment'].upper()}  ({res['confidence_level']} confidence: {res['confidence']*100:.1f}%)")
        print(f"Probs      : {json.dumps(res['probabilities'])}")
        print(f"Cleaned    : {res['cleaned_text']}")
    else:
        # Interactive mode
        print("=== Twitter Sentiment Analyser (interactive) ===")
        print("Type a tweet and press Enter. Type 'quit' to exit.\n")
        while True:
            text = input("Tweet > ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            res = predictor.predict(text)
            print(f"  ➜  {res['sentiment'].upper()}  "
                  f"({res['confidence_level']} confidence: {res['confidence']*100:.1f}%)\n")
