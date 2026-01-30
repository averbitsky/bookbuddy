"""
Download book covers from Open Library API.
Uses the Open Library Search API to find books and download their covers.
"""

import json
import os
import re
import time
import urllib.request
import urllib.parse
from pathlib import Path
from loguru import logger

from bookbuddy import sanitize_filename


def search_open_library(title: str, author: str) -> dict | None:
    """Search Open Library for a book and return cover info."""
    # Clean up title for search
    search_title = re.sub(r"\s*\(.*?\)\s*", "", title)  # Remove parenthetical
    search_title = re.sub(r":\s*Book\s*\d+", "", search_title)  # Remove "Book N"

    query = urllib.parse.quote(f"{search_title} {author}")
    url = f"https://openlibrary.org/search.json?q={query}&limit=1"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BookBuddy/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode())

        if data.get("docs"):
            doc = data["docs"][0]
            cover_i = doc.get("cover_i")
            isbn = doc.get("isbn", [None])[0] if doc.get("isbn") else None
            return {
                "cover_id": cover_i,
                "isbn": isbn,
                "found_title": doc.get("title", ""),
                "olid": doc.get("key", "").replace("/works/", ""),
            }
    except Exception as e:
        logger.warning(f"Search error: {e}")

    return None


def download_cover(cover_id: int, filepath: str) -> bool:
    """Download a cover image from Open Library."""
    url = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BookBuddy/1.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            image_data = response.read()

            # Check if we got a real image (not the placeholder)
            if len(image_data) < 1000:  # Placeholder is tiny
                return False

            with open(filepath, "wb") as f:
                f.write(image_data)
            return True
    except Exception as e:
        logger.warning(f"Download error: {e}")

    return False


def create_placeholder_cover(title: str, author: str, filepath: str):
    """Create a simple SVG placeholder cover."""
    # Get initials from title
    words = title.split()[:2]
    initials = "".join(w[0].upper() for w in words if w)

    # Simple color based on title hash
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]
    color = colors[hash(title) % len(colors)]

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="200" height="300" viewBox="0 0 200 300">
  <rect width="200" height="300" fill="{color}"/>
  <text x="100" y="120" font-family="Georgia, serif" font-size="48" fill="white" text-anchor="middle">{initials}</text>
  <text x="100" y="180" font-family="Arial, sans-serif" font-size="12" fill="white" text-anchor="middle" opacity="0.9">
    <tspan x="100" dy="0">{title[:25]}</tspan>
    <tspan x="100" dy="18">{'...' if len(title) > 25 else ''}</tspan>
  </text>
  <text x="100" y="260" font-family="Arial, sans-serif" font-size="10" fill="white" text-anchor="middle" opacity="0.7">{author[:30]}</text>
</svg>"""

    with open(filepath, "w") as f:
        f.write(svg)


def main():
    # Load catalog
    catalog_path = Path("data/book_catalog.json")
    with open(catalog_path) as f:
        books = json.load(f)

    covers_dir = Path("static/covers")
    covers_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    placeholder_count = 0

    logger.info(f"Downloading covers for {len(books)} books...")

    for i, book in enumerate(books):
        title = book["title"]
        author = book["author"]
        filename = sanitize_filename(title)

        logger.info(f"[{i+1}/{len(books)}] {title}")

        # Search for the book
        result = search_open_library(title, author)

        cover_downloaded = False
        if result and result.get("cover_id"):
            jpg_path = covers_dir / f"{filename}.jpg"
            if download_cover(result["cover_id"], str(jpg_path)):
                book["cover_url"] = f"/static/covers/{filename}.jpg"
                logger.success(f"  Downloaded cover")
                success_count += 1
                cover_downloaded = True

        if not cover_downloaded:
            # Create placeholder
            svg_path = covers_dir / f"{filename}.svg"
            create_placeholder_cover(title, author, str(svg_path))
            book["cover_url"] = f"/static/covers/{filename}.svg"
            logger.info(f"  Created placeholder")
            placeholder_count += 1

        # Rate limiting - be nice to the API
        time.sleep(0.5)

    # Save updated catalog
    with open(catalog_path, "w") as f:
        json.dump(books, f, indent=2)

    logger.info(
        f"Done! Downloaded {success_count} covers, created {placeholder_count} placeholders. Updated {catalog_path}"
    )


if __name__ == "__main__":
    main()
