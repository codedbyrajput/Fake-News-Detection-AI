"""
news_article.py

Domain model for a single news article used throughout the pipeline.

Label convention:
    0 -> FAKE
    1 -> REAL
"""

from __future__ import annotations

from myExceptions import DataFormatException


class NewsArticle:
    """Represents a single labeled news example."""

    def __init__(self, title: str, text: str, label: int):
        """
        Args:
            title: Article title.
            text: Article body text.
            label: 0 (FAKE) or 1 (REAL).
        """
        if title is None or text is None or label is None:
            raise DataFormatException("Missing title/text/label.")

        title_str = str(title).strip()
        text_str = str(text).strip()

        if title_str == "" or text_str == "":
            raise DataFormatException("Title/text cannot be empty.")

        self.title = title_str
        self.text = text_str
        self.label = int(label)

    def __repr__(self) -> str:
        """Debug-friendly representation (short title preview)."""
        snippet = self.title[:30]
        return f"NewsArticle(label={self.label}, title='{snippet}...')"
