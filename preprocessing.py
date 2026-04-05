"""
preprocessing.py
----------------
Handles all text cleaning and preprocessing steps for Twitter sentiment analysis.
Includes a reusable TextPreprocessor class and standalone utility functions.
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ── Download required NLTK resources (safe to call multiple times) ──────────
def download_nltk_resources():
    """Download all required NLTK datasets if not already present."""
    resources = [
        ("tokenizers/punkt",           "punkt"),
        ("tokenizers/punkt_tab",       "punkt_tab"),
        ("corpora/stopwords",          "stopwords"),
        ("corpora/wordnet",            "wordnet"),
        ("corpora/omw-1.4",            "omw-1.4"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

download_nltk_resources()


# ── Core cleaning helpers ────────────────────────────────────────────────────

def remove_urls(text: str) -> str:
    """Remove http/https URLs and bare www links from text."""
    return re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)


def remove_mentions_hashtags(text: str) -> str:
    """Remove Twitter @mentions and #hashtags."""
    return re.sub(r"@\w+|#\w+", "", text)


def remove_punctuation(text: str) -> str:
    """Strip all punctuation characters."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text: str) -> str:
    """Remove standalone numbers and digits."""
    return re.sub(r"\b\d+\b", "", text)


def remove_extra_whitespace(text: str) -> str:
    """Collapse multiple spaces / newlines into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def remove_stopwords(tokens: list, stop_words: set) -> list:
    """Filter out stopwords from a token list."""
    return [t for t in tokens if t not in stop_words]


def lemmatize_tokens(tokens: list, lemmatizer: WordNetLemmatizer) -> list:
    """Apply WordNet lemmatization to every token."""
    return [lemmatizer.lemmatize(t) for t in tokens]


# ── Pipeline class ───────────────────────────────────────────────────────────

class TextPreprocessor:
    """
    End-to-end text preprocessing pipeline for Twitter data.

    Steps (all enabled by default):
        1. Lowercase
        2. Remove URLs
        3. Remove @mentions and #hashtags
        4. Remove punctuation
        5. Remove numbers
        6. Tokenize
        7. Remove stopwords
        8. Lemmatize
        9. Re-join tokens into a clean string

    Parameters
    ----------
    remove_stops   : bool  – Remove English stopwords (default True)
    lemmatize      : bool  – Apply lemmatization (default True)
    min_token_len  : int   – Discard tokens shorter than this (default 2)
    extra_stopwords: list  – Additional domain-specific words to remove
    """

    def __init__(
        self,
        remove_stops: bool = True,
        lemmatize: bool = True,
        min_token_len: int = 2,
        extra_stopwords: list = None,
    ):
        self.remove_stops   = remove_stops
        self.lemmatize      = lemmatize
        self.min_token_len  = min_token_len

        # Build stopword set
        self._stop_words = set(stopwords.words("english"))
        if extra_stopwords:
            self._stop_words.update(extra_stopwords)

        self._lemmatizer = WordNetLemmatizer()

    # ── Public interface ─────────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        """
        Full pipeline: clean raw tweet text → return processed string.

        Parameters
        ----------
        text : str – Raw tweet text

        Returns
        -------
        str – Cleaned, joined text ready for vectorisation
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # 1. Lowercase
        text = text.lower()

        # 2. Remove URLs
        text = remove_urls(text)

        # 3. Remove @mentions and #hashtags
        text = remove_mentions_hashtags(text)

        # 4. Remove punctuation
        text = remove_punctuation(text)

        # 5. Remove numbers
        text = remove_numbers(text)

        # 6. Collapse whitespace
        text = remove_extra_whitespace(text)

        # 7. Tokenize
        tokens = word_tokenize(text)

        # 8. Remove short tokens
        tokens = [t for t in tokens if len(t) >= self.min_token_len]

        # 9. Remove stopwords
        if self.remove_stops:
            tokens = remove_stopwords(tokens, self._stop_words)

        # 10. Lemmatize
        if self.lemmatize:
            tokens = lemmatize_tokens(tokens, self._lemmatizer)

        return " ".join(tokens)

    def clean_series(self, series) -> list:
        """
        Apply clean() to a pandas Series or list of strings.

        Parameters
        ----------
        series : iterable of str

        Returns
        -------
        list of str
        """
        return [self.clean(text) for text in series]

    def __repr__(self):
        return (
            f"TextPreprocessor("
            f"remove_stops={self.remove_stops}, "
            f"lemmatize={self.lemmatize}, "
            f"min_token_len={self.min_token_len})"
        )


# ── Quick smoke-test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_tweets = [
        "I LOVE this new phone! 😍 https://example.com #tech @Apple",
        "Worst service EVER!! 😡 Never coming back #angry @CustomerSupport",
        "Just got my package today. It's fine I guess.",
    ]

    preprocessor = TextPreprocessor()
    print("=== TextPreprocessor smoke-test ===")
    for tweet in sample_tweets:
        print(f"\n  Raw  : {tweet}")
        print(f"  Clean: {preprocessor.clean(tweet)}")
