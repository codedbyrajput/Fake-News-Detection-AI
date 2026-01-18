"""
interface.py

Small CLI + wrapper to run predictions on raw user input.

Label convention:
    0 -> FAKE
    1 -> REAL
"""

from __future__ import annotations

from typing import Tuple


class PredictionInterface:
    """Connects preprocessor, feature extractor, and classifier for end-to-end predictions."""

    def __init__(self, preprocessor, feature_extractor, classifier):
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def predict_from_text(self, raw_text: str) -> Tuple[int, float]:
        """
        Predicts a label and confidence for a single input string.

        Returns:
            (label, confidence)
            label: 0 (FAKE) or 1 (REAL)
            confidence: probability of the predicted class in [0, 1]
        """
        clean = self.preprocessor.clean_text(raw_text)
        X = self.feature_extractor.transform([clean])

        print("TOKENS:", len(clean.split()))
        print("NONZERO FEATURES:", X.nnz)
        label = int(self.classifier.predict(X)[0])

        # Probability that the input is FAKE
        prob_fake = float(self.classifier.predict_proba_fake(X)[0])

        # Make it harder to call something FAKE (tune this number)
        threshold = 0.70
        label = 0 if prob_fake >= threshold else 1

        confidence = prob_fake if label == 0 else (1.0 - prob_fake)
        return label, confidence

    def run_cli(self) -> None:
        """Runs a simple command-line loop for interactive testing."""
        print("Fake News Detector - CLI")
        print("Enter a news article or headline (or 'quit' to exit):")

        while True:
            user_input = input("> ").strip()
            if user_input.lower() in ("quit", "exit", ""):
                print("Exiting.")
                break

            try:
                label, confidence = self.predict_from_text(user_input)
                result = "FAKE" if label == 0 else "REAL"

                print(f"\nResult: This article is {result} news.")
                print(f"Confidence: {confidence * 100:.1f}%\n")
            except Exception as e:
                print(f"ERROR: {e}")
