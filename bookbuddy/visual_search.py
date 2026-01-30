"""
Visual book search using OpenAI Vision and Sentence Transformers. Enables students to take photos of book covers and
find similar books. Pre-computes and caches book embeddings for fast similarity search.
"""

import base64
import json
import os
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from loguru import logger


class VisualSearch:
    """
    Visual book search using OpenAI's vision capabilities and semantic embeddings. Embeddings are pre-computed and
    cached in data/book_embeddings.pkl.
    """

    def __init__(self, data_dir: str = "data"):
        """Initialize with an OpenAI client and embedding model."""
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.data_dir = Path(data_dir)
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")
        self._book_embeddings = None
        self._book_ids = None
        self._embeddings_path = self.data_dir / "book_embeddings.pkl"

    def _encode_image(self, image_source) -> str:
        """Encode image to base64."""
        if isinstance(image_source, str):
            with open(image_source, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(image_source, bytes):
            return base64.b64encode(image_source).decode("utf-8")
        else:
            raise ValueError("Unsupported image source type")

    def _get_book_text(self, book: Dict) -> str:
        """Create a searchable text representation of a book."""
        pages = book.get("pages", 0)
        # Add length descriptors for semantic matching
        if pages <= 100:
            length_desc = "short quick read"
        elif pages <= 200:
            length_desc = "short book"
        elif pages <= 350:
            length_desc = "medium length"
        else:
            length_desc = "long book"

        parts = [
            book.get("title", ""),
            book.get("genre", ""),
            " ".join(book.get("themes", [])),
            book.get("description", "")[:300] if book.get("description") else "",
            f"{pages} pages {length_desc}",
        ]
        return " ".join(filter(None, parts))

    def _load_or_build_embeddings(self, catalog: Dict) -> tuple:
        """Load pre-computed embeddings from the cache or build them."""
        # Try to load from the cache
        if self._embeddings_path.exists():
            try:
                with open(self._embeddings_path, "rb") as f:
                    cached = pickle.load(f)
                    # Verify cache matches the current catalog
                    if set(cached["book_ids"]) == set(catalog.keys()):
                        self._book_ids = cached["book_ids"]
                        self._book_embeddings = cached["embeddings"]
                        logger.info(
                            f"Loaded {len(self._book_ids)} book embeddings from cache"
                        )
                        return self._book_ids, self._book_embeddings
                    else:
                        logger.info("Cache outdated, rebuilding embeddings...")
            except Exception as e:
                logger.warning(f"Cache load failed: {e}, rebuilding...")

        # Build embeddings
        logger.info("Building book embeddings (one-time operation)...")
        self._book_ids = list(catalog.keys())
        book_texts = [self._get_book_text(catalog[bid]) for bid in self._book_ids]
        self._book_embeddings = self.embedding_model.encode(
            book_texts, show_progress_bar=True, convert_to_numpy=True
        )

        # Save to cache
        try:
            with open(self._embeddings_path, "wb") as f:
                pickle.dump(
                    {"book_ids": self._book_ids, "embeddings": self._book_embeddings}, f
                )
            logger.info(
                f"Saved {len(self._book_ids)} book embeddings to {self._embeddings_path}"
            )
        except Exception as e:
            logger.error(f"Failed to save embeddings cache: {e}")

        return self._book_ids, self._book_embeddings

    def analyze_book_cover(self, image_source) -> Dict[str, Any]:
        """
        Analyze a book cover image to extract metadata.
        """
        if not image_source:
            return self._fallback_result()

        try:
            image_base64 = self._encode_image(image_source)

            # Detect the image type from first bytes
            image_type = "jpeg"
            if isinstance(image_source, bytes):
                if image_source[:8] == b"\x89PNG\r\n\x1a\n":
                    image_type = "png"
                elif image_source[:2] == b"\xff\xd8":
                    image_type = "jpeg"

            response = self.client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": 'Read the book cover. Return JSON only: {"title": "...", "author": "...", "genres": ["..."], "themes": ["..."]}',
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_type};base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ],
                max_completion_tokens=1000,
            )

            content = response.choices[0].message.content
            if not content:
                return self._fallback_result()

            content = content.strip()

            # Remove Markdown code blocks if present
            if "```" in content:
                parts = content.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        content = part
                        break

            data = json.loads(content)

            return {
                "title": data.get("title", "Unknown"),
                "author": data.get("author", "Unknown"),
                "genre_hints": data.get("genres", []),
                "themes": data.get("themes", []),
                "visual_style": data.get("visual_description", ""),
                "confidence": 0.95,
            }

        except Exception as e:
            logger.exception(f"Vision API error: {type(e).__name__}: {e}")
            return self._fallback_result()

    def _fallback_result(self) -> Dict[str, Any]:
        """Return the fallback result when analysis fails."""
        return {
            "title": "Unknown",
            "author": "Unknown",
            "genre_hints": [],
            "themes": [],
            "visual_style": "",
            "confidence": 0.0,
        }

    def _get_acceptable_levels(self, reading_level: str) -> set:
        """Get acceptable reading levels (current + adjacent)."""
        levels = ["K-2", "3-5", "6-8", "9-12"]
        if reading_level not in levels:
            return set(levels)
        idx = levels.index(reading_level)
        # Include the current level and one level up/down
        acceptable = {reading_level}
        if idx > 0:
            acceptable.add(levels[idx - 1])
        if idx < len(levels) - 1:
            acceptable.add(levels[idx + 1])
        return acceptable

    def find_similar_books(
        self,
        book_info: Dict[str, Any],
        recommendation_engine,
        student_id: Optional[str] = None,
        n: int = 5,
    ) -> Dict[str, Any]:
        """Find similar books using semantic embeddings, filtered by reading level."""

        # Get the student's reading level for age-appropriate filtering
        reading_level = None
        acceptable_levels = None
        if student_id and student_id in recommendation_engine.students:
            reading_level = recommendation_engine.students[student_id].get(
                "reading_level"
            )
            acceptable_levels = self._get_acceptable_levels(reading_level)

        # First, try the exact title match
        title = book_info.get("title", "")
        if title and title != "Unknown":
            exact_matches = recommendation_engine.find_book_by_title(title)
            if exact_matches:
                matched_book = exact_matches[0]
                similar = recommendation_engine.get_similar_books(
                    matched_book["book_id"],
                    n=n * 2,  # Get more to filter by reading level
                )
                # Filter by reading level for age-appropriate recommendations
                if acceptable_levels:
                    similar = [
                        b
                        for b in similar
                        if b.get("reading_level") in acceptable_levels
                    ]
                return {
                    "matched_book": matched_book,
                    "similar_books": similar[:n],
                    "match_type": "exact_title_match",
                }

        # No exact match - use semantic similarity
        query_parts = [
            " ".join(book_info.get("genres", book_info.get("genre_hints", []))),
            " ".join(book_info.get("themes", [])),
        ]
        query_text = " ".join(filter(None, query_parts))

        # Load or build embeddings for all books
        book_ids, book_embeddings = self._load_or_build_embeddings(
            recommendation_engine.catalog
        )

        # FILTER BY READING LEVEL FIRST - only consider books in acceptable grade range
        if acceptable_levels:
            filtered_indices = []
            filtered_book_ids = []
            for idx, bid in enumerate(book_ids):
                book = recommendation_engine.catalog.get(bid, {})
                if book.get("reading_level") in acceptable_levels:
                    filtered_indices.append(idx)
                    filtered_book_ids.append(bid)

            if not filtered_indices:
                return {
                    "matched_book": None,
                    "similar_books": [],
                    "match_type": "no_books_in_level",
                }

            # Use only filtered embeddings
            filtered_embeddings = book_embeddings[filtered_indices]
        else:
            # No filtering - use all books
            filtered_book_ids = book_ids
            filtered_embeddings = book_embeddings

        if not query_text.strip():
            # No query text - return books from the filtered set with default scores
            similar_books = []
            for bid in filtered_book_ids[:n]:
                book = recommendation_engine.catalog.get(bid, {})
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
                        "similarity_score": 0.75,
                        "shelf_location": book.get("shelf_location", {}),
                    }
                )
            return {
                "matched_book": None,
                "similar_books": similar_books,
                "match_type": "fallback_by_level",
            }

        # Embed the query
        query_embedding = self.embedding_model.encode(
            [query_text], convert_to_numpy=True
        )[0]

        # Compute cosine similarities only for filtered books
        similarities = np.dot(filtered_embeddings, query_embedding) / (
            np.linalg.norm(filtered_embeddings, axis=1)
            * np.linalg.norm(query_embedding)
            + 1e-8
        )

        # Get top matches sorted by similarity (from the pre-filtered set)
        top_indices = np.argsort(similarities)[::-1][:n]

        similar_books = []
        for idx in top_indices:
            bid = filtered_book_ids[idx]
            book = recommendation_engine.catalog.get(bid, {})

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
                    "similarity_score": float(similarities[idx]),
                    "shelf_location": book.get("shelf_location", {}),
                }
            )

        return {
            "matched_book": None,
            "similar_books": similar_books,
            "match_type": "semantic_similarity",
            "query_text": query_text,
        }
