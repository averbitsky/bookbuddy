"""
Book recommendation engine with content-based similarity for finding similar books.
"""

import json
import os
from typing import Any, Dict, List

import numpy as np
from loguru import logger


class BookRecommendationEngine:
    """
    Book recommendation engine providing:
    1. Content-based similarity (book metadata)
    2. Title search
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.catalog = {}
        self.students = {}  # Virtual students created at runtime (grade-based)
        self.book_ids = []

        # Content-based similarity matrix
        self.content_similarity = None

        # Load and initialize
        self._load_data()
        self._build_matrices()

    def _load_data(self):
        """Load book catalog."""
        catalog_path = os.path.join(self.data_dir, "book_catalog.json")

        with open(catalog_path) as f:
            catalog_list = json.load(f)
            self.catalog = {book["book_id"]: book for book in catalog_list}

        logger.info(f"Loaded {len(self.catalog)} books")

    def _build_matrices(self):
        """Build content similarity matrix."""
        logger.info("Building content similarity matrix...")
        self.book_ids = list(self.catalog.keys())
        self._build_content_similarity()
        logger.info("Content similarity matrix built")

    def _build_content_similarity(self):
        """Build a content-based similarity matrix using book metadata."""
        n_books = len(self.book_ids)

        # Create feature vectors for each book
        # Features: genre (one-hot), themes (multi-hot), reading_level (one-hot)
        all_genres = list(set(b["genre"] for b in self.catalog.values()))
        all_themes = list(set(t for b in self.catalog.values() for t in b["themes"]))
        all_levels = ["K-2", "3-5", "6-8", "9-12"]

        n_features = len(all_genres) + len(all_themes) + len(all_levels)
        book_features = np.zeros((n_books, n_features))

        for i, bid in enumerate(self.book_ids):
            book = self.catalog[bid]

            # Genre one-hot
            if book["genre"] in all_genres:
                book_features[i, all_genres.index(book["genre"])] = 1.0

            # Themes multi-hot
            for theme in book["themes"]:
                if theme in all_themes:
                    book_features[i, len(all_genres) + all_themes.index(theme)] = 1.0

            # Reading level one-hot
            if book["reading_level"] in all_levels:
                book_features[
                    i,
                    len(all_genres)
                    + len(all_themes)
                    + all_levels.index(book["reading_level"]),
                ] = 1.0

        # Compute cosine similarity
        norms = np.linalg.norm(book_features, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = book_features / norms
        self.content_similarity = np.dot(normalized, normalized.T)

    def get_similar_books(self, book_id: str, n: int = 5) -> List[Dict[str, Any]]:
        """Get books similar to a given book (content-based only)."""
        book_idx = {bid: i for i, bid in enumerate(self.book_ids)}

        if book_id not in book_idx:
            return []

        b_i = book_idx[book_id]
        similarities = list(enumerate(self.content_similarity[b_i]))
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Skip the book itself (similarity = 1.0)
        similar_books = []
        for other_i, sim in similarities[1 : n + 1]:
            bid = self.book_ids[other_i]
            book = self.catalog.get(bid, {})
            similar_books.append(
                {
                    "book_id": bid,
                    "title": book.get("title", "Unknown"),
                    "author": book.get("author", "Unknown"),
                    "genre": book.get("genre", "Unknown"),
                    "themes": book.get("themes", []),
                    "reading_level": book.get("reading_level", ""),
                    "description": book.get("description", ""),
                    "cover_url": book.get("cover_url", ""),
                    "similarity_score": round(sim, 3),
                    "shelf_location": book.get("shelf_location", {}),
                }
            )

        return similar_books

    def find_book_by_title(self, title_query: str) -> List[Dict[str, Any]]:
        """Search for books by title (fuzzy matching)."""
        query_lower = title_query.lower()
        matches = []

        for bid, book in self.catalog.items():
            title_lower = book["title"].lower()
            if query_lower in title_lower:
                score = len(query_lower) / len(title_lower)
                matches.append({**book, "match_score": score})

        matches.sort(key=lambda x: x["match_score"], reverse=True)
        return matches[:10]


# Singleton instance
_engine_instance = None


def get_engine(data_dir: str = "data") -> BookRecommendationEngine:
    """Get or create the recommendation engine instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = BookRecommendationEngine(data_dir)
    return _engine_instance


if __name__ == "__main__":
    # Test the engine
    engine = get_engine()

    # Test similar books
    sample_book = list(engine.catalog.keys())[0]
    similar = engine.get_similar_books(sample_book, n=3)
    logger.info(f"Books similar to {engine.catalog[sample_book]['title']}:")
    for book in similar:
        logger.info(f"  - {book['title']} (Similarity: {book['similarity_score']})")

    # Test title search
    matches = engine.find_book_by_title("Harry")
    logger.info(f"Title search for 'Harry':")
    for book in matches[:3]:
        logger.info(f"  - {book['title']}")
