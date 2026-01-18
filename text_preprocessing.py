"""
text_preprocessing.py

Text cleaning utilities used before feature extraction.

Pipeline:
    1) Lowercase
    2) Remove non-alphanumeric characters (keep spaces)
    3) Tokenize (split on whitespace)
    4) Remove stopwords
    5) Stemming (can be done)
"""

from __future__ import annotations

import re
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


class TextPreprocessor:
    """Cleans raw input text into a normalized string for TF-IDF."""

    def __init__(self, use_stemming: bool = False):
        self.use_stemming = use_stemming
        self.stemmer = PorterStemmer() if use_stemming else None

        # Ensure NLTK resources exist (keeps runtime errors away).
        try:
            _ = stopwords.words("english")
        except LookupError:
            nltk.download("stopwords", quiet=True)

        self.stopwords = set(stopwords.words("english"))

    def clean_text(self, text: str) -> str:
        """
        Converts raw text into a cleaned string.

        Args:
            text: Raw input string.

        Returns:
            Cleaned string suitable for TF-IDF.
        """
        if text is None:
            return ""

        s = str(text).lower()
        s = re.sub(r"[^a-z0-9\s]", " ", s)   # replace punctuation with space
        s = re.sub(r"\s+", " ", s).strip()  # collapse multiple spaces

        tokens = self._tokenize(s)
        tokens = [t for t in tokens if t not in self.stopwords]

        if self.use_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]

        return " ".join(tokens)

    def _tokenize(self, text: str) -> List[str]:
        """Splits text into tokens using whitespace."""
        return text.split()
