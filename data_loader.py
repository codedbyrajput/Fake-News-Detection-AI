"""
data_loader.py

Loads news articles from a CSV file into `NewsArticle` objects.

Expected CSV schema (case-insensitive):
    - title: str
    - text: str
    - label: "FAKE"/"REAL" or 0/1

Label convention used by this project:
    - 0 => FAKE
    - 1 => REAL

Invalid rows (missing fields, unknown labels, or bad formats) are skipped.
"""

from __future__ import annotations

import os
from typing import List

import pandas as pd

from news_article import NewsArticle
from myExceptions import DataFormatException


class DataLoader:
    """Utility class for loading training/evaluation data from disk."""

    @staticmethod
    def load_data(file_path: str = "myDataSet.csv") -> List[NewsArticle]:
        """
        Reads the dataset and returns a list of NewsArticle objects.

        Args:
            file_path: Path to the CSV file.

        Returns:
            A list of NewsArticle objects. If the file is missing or unreadable,
            returns an empty list.
        """
        articles: List[NewsArticle] = []

        if not os.path.exists(file_path):
            print(f"ERROR: File not found -> {file_path}")
            return articles

        try:
            data = pd.read_csv(file_path)
        except Exception as e:
            print(f"ERROR: Could not read file -> {e}")
            return articles

        # Validate required columns (case-insensitive).
        cols = [c.lower().strip() for c in data.columns]
        required = {"title", "text", "label"}

        if not required.issubset(set(cols)):
            print("ERROR: CSV must have columns: title, text, label")
            print("Found columns:", list(data.columns))
            return articles

        title_col = data.columns[cols.index("title")]
        text_col = data.columns[cols.index("text")]
        label_col = data.columns[cols.index("label")]

        skipped = 0

        for _, row in data.iterrows():
            title = row[title_col]
            text = row[text_col]
            label = row[label_col]

            if pd.isna(title) or pd.isna(text) or pd.isna(label):
                skipped += 1
                continue

            label_int = DataLoader._to_label_int(label)
            if label_int is None:
                skipped += 1
                continue

            try:
                articles.append(NewsArticle(str(title), str(text), label_int))
            except DataFormatException:
                skipped += 1

        print(f"Loaded {len(articles)} articles from {file_path} (skipped {skipped})")
        return articles

    @staticmethod
    def _to_label_int(label) -> int | None:
        """
        Converts a raw label value to the project's int convention.

        Returns:
            0 for FAKE, 1 for REAL, or None if the label is unrecognized.
        """
        label_str = str(label).strip().upper()

        if label_str in ("FAKE", "0"):
            return 0
        if label_str in ("REAL", "1"):
            return 1
        return None
