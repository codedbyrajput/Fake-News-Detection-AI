"""
fake_news_classifier.py

Train and run a binary classifier for fake-news detection.

Label convention used throughout the project:
    0 -> FAKE
    1 -> REAL
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from myExceptions import ModelTrainingException


class FakeNewsClassifier:
    """Thin wrapper around scikit-learn LogisticRegression for this project."""

    def __init__(self):
        # `max_iter` prevents early stopping warnings on larger vocabularies.
        self.model = LogisticRegression(max_iter=1000, random_state=10, class_weight="balanced")
        self.isTrained = False

    def train(self, X, y) -> None:
        """
        Fits the model on TF-IDF features.

        Args:
            X: Feature matrix (e.g., TF-IDF sparse matrix).
            y: Labels (0 for FAKE, 1 for REAL).
        """
        if X is None or y is None or len(y) == 0:
            raise ModelTrainingException("Training data is incomplete.")

        try:
            self.model.fit(X, y)
            self.isTrained = True
        except Exception as e:
            raise ModelTrainingException(f"Training failed: {e}")

    def predict(self, X):
        """
        Predicts labels for feature matrix X.

        Returns:
            Array-like of predicted labels (0/1).
        """
        if not self.isTrained:
            raise ModelTrainingException("Model not trained yet.")
        return self.model.predict(X)

    def predict_proba_fake(self, X):
        """
        Predicts P(FAKE) for each row of X.

        Returns:
            A 1D array of probabilities in [0, 1], one per input row.
        """
        if not self.isTrained:
            raise ModelTrainingException("Model not trained yet.")

        proba = self.model.predict_proba(X)
        classes = list(self.model.classes_)

        # Ensure the model was trained with the expected label scheme.
        if 0 not in classes:
            raise ModelTrainingException(f"Expected class 0 for FAKE, got classes {classes}")

        idx_fake = classes.index(0)
        return proba[:, idx_fake]

    def save_model(self, path: str) -> None:
        """Saves the underlying scikit-learn model to disk."""
        from joblib import dump
        dump(self.model, path)

    @classmethod
    def load_model(cls, path: str) -> "FakeNewsClassifier":
        """Loads a saved scikit-learn model from disk."""
        from joblib import load
        clf = cls()
        clf.model = load(path)
        clf.isTrained = True
        return clf
