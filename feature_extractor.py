"""
feature_extractor.py

Turns cleaned text into numeric features using TF-IDF.

Notes:
    - Call `fit_transform()` on training data first.
    - Then call `transform()` on validation/test/user input using the same vocabulary.
"""

from __future__ import annotations

from typing import List, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer


class FeatureExtractor:
    """Wrapper around scikit-learn TF-IDF vectorizer with a stable vocabulary."""

    def __init__(self, max_features: int = 10_000):
        # `max_features` limits vocabulary size to reduce memory usage.
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        self.is_fit = False

    def fit_transform(self, cleaned_documents: Sequence[str]):
        """
        Learns vocabulary from training data and returns TF-IDF features.

        Args:
            cleaned_documents: List/sequence of cleaned documents (strings).

        Returns:
            TF-IDF feature matrix for the training set.
        """
        X = self.vectorizer.fit_transform(cleaned_documents)
        self.is_fit = True
        return X

    def transform(self, cleaned_documents: Sequence[str]):
        """
        Converts new documents into TF-IDF features using the learned vocabulary.

        Args:
            cleaned_documents: List/sequence of cleaned documents (strings).

        Returns:
            TF-IDF feature matrix for the provided documents.
        """
        if not self.is_fit:
            raise ValueError("Vectorizer not fitted. Call fit_transform() first.")
        return self.vectorizer.transform(cleaned_documents)

    def get_feature_names(self) -> List[str]:
        """
        Returns the learned vocabulary feature names.

        Raises:
            ValueError: If the vectorizer has not been fitted yet.
        """
        if not self.is_fit:
            raise ValueError("Vectorizer not fitted yet.")
        return list(self.vectorizer.get_feature_names_out())
