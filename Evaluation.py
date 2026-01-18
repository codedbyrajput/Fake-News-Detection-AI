"""
evaluator.py

Evaluation utilities for binary fake-news classification.

Label convention:
    0 -> FAKE (positive class for precision/recall)
    1 -> REAL
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Union

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


Label = Union[int, str]


class Evaluator:
    """Static helpers for computing metrics and visualizing results."""

    @staticmethod
    def evaluate(y_true: Sequence[Label], y_pred: Sequence[Label]) -> Dict[str, float]:
        """
        Computes accuracy, precision, recall, and F1 score.

        Notes:
            - Precision/Recall/F1 treat FAKE (0) as the positive class.
            - Accepts labels as int (0/1) or str ("FAKE"/"REAL").
        """
        y_true_int = Evaluator._to_int_labels(y_true)
        y_pred_int = Evaluator._to_int_labels(y_pred)

        if len(y_true_int) != len(y_pred_int):
            raise ValueError("Mismatch in number of true and predicted labels")

        return {
            "accuracy": accuracy_score(y_true_int, y_pred_int),
            "precision": precision_score(y_true_int, y_pred_int, pos_label=0, zero_division=0),
            "recall": recall_score(y_true_int, y_pred_int, pos_label=0, zero_division=0),
            "f1_score": f1_score(y_true_int, y_pred_int, pos_label=0, zero_division=0),
        }

    @staticmethod
    def confusion_matrix_values(y_true: Sequence[Label], y_pred: Sequence[Label]):
        """
        Returns confusion matrix in the order:
            [[TN, FP],
             [FN, TP]]
        where the positive class is FAKE (0).
        """
        y_true_int = Evaluator._to_int_labels(y_true)
        y_pred_int = Evaluator._to_int_labels(y_pred)

        if len(y_true_int) != len(y_pred_int):
            raise ValueError("Mismatch in number of true and predicted labels")

        return confusion_matrix(y_true_int, y_pred_int, labels=[0, 1])

    @staticmethod
    def plot_confusion_matrix(y_true: Sequence[Label], y_pred: Sequence[Label]) -> None:
        """Displays a confusion matrix heatmap."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        cm = Evaluator.confusion_matrix_values(y_true, y_pred)

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Pred: FAKE", "Pred: REAL"],
            yticklabels=["Actual: FAKE", "Actual: REAL"],
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Class")
        plt.ylabel("Actual Class")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _to_int_labels(labels: Sequence[Label]) -> List[int]:
        """
        Converts labels to int format.

        Accepts:
            - 0/1
            - "FAKE"/"REAL" (any case, with whitespace)
        """
        if len(labels) == 0:
            return []

        out: List[int] = []
        for x in labels:
            if isinstance(x, int):
                out.append(x)
                continue

            s = str(x).strip().upper()
            if s in ("FAKE", "0"):
                out.append(0)
            elif s in ("REAL", "1"):
                out.append(1)
            else:
                raise ValueError(f"Unknown label value: {x}")
        return out
