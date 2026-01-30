"""
BookBuddy - Immersive Virtual Library Server
Serves the 3D virtual library with dynamic recommendations.
"""

import base64
import json as json_module
import os
from flask import Flask, jsonify, send_file, request, Response
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from bookbuddy.data_generator import generate_all_data
from bookbuddy.recommendation_engine import BookRecommendationEngine
from bookbuddy.visual_search import VisualSearch
from bookbuddy.llm_agent import BookBuddyAgent

app = Flask(__name__, static_folder="static", static_url_path="/static")

# Initialize recommendation engine
data_dir = Path(__file__).parent / "data"
if not (data_dir / "book_catalog.json").exists():
    logger.info("Generating synthetic library data...")
    generate_all_data(str(data_dir))

engine = BookRecommendationEngine(str(data_dir))

# Vision provider: "openai" or "moondream" (set via VISION_PROVIDER env var)
vision_provider = os.environ.get("VISION_PROVIDER", "openai")
visual_search = VisualSearch(vision_provider=vision_provider)

agent = BookBuddyAgent(engine, visual_search, use_openai=True)


@app.route("/")
def index():
    """Serve the virtual library HTML."""
    return send_file("templates/virtual_library.html")


def get_reading_level_from_grade(grade_str):
    """Convert grade string to reading level."""
    grade = 0 if grade_str == "K" else int(grade_str)
    if grade <= 2:
        return "K-2"
    elif grade <= 5:
        return "3-5"
    elif grade <= 8:
        return "6-8"
    return "9-12"


@app.route("/api/chat", methods=["POST"])
def chat_endpoint():
    """Chat with the AI librarian using Server-Sent Events for real-time updates."""
    data = request.json
    student_id = data.get("student_id")
    message = data.get("message", "")

    # Handle grade-based virtual users
    if student_id and student_id.startswith("grade-"):
        grade = student_id.replace("grade-", "")
        reading_level = get_reading_level_from_grade(grade)
        # Create a virtual student profile for the chat
        virtual_student = {
            "student_id": student_id,
            "name": "Reader",
            "grade": int(grade) if grade != "K" else 0,
            "reading_level": reading_level,
            "favorite_genres": [],
            "favorite_themes": [],
        }
        engine.students[student_id] = virtual_student
    elif not student_id or student_id not in engine.students:
        return jsonify({"error": "Invalid student"}), 400

    def generate():
        # Get the result (this will execute tool calls)
        result = agent.chat(message, student_id)

        # Stream tool calls that were made
        tool_display_names = {
            "semantic_search": "Searching books",
            "get_similar_books": "Finding similar books",
            "search_by_title": "Looking up book",
        }
        tool_calls = result.get("tool_calls", [])
        for tc in tool_calls:
            tool_name = tc.get("name", "unknown")
            display_name = tool_display_names.get(tool_name, f"Processing {tool_name}")
            yield f"data: {json_module.dumps({'type': 'tool', 'name': display_name})}\n\n"

        # Stream the final response
        yield f"data: {json_module.dumps({'type': 'response', 'text': result.get('response', '')})}\n\n"

        # Stream any recommendations found
        recs = result.get("recommendations", [])
        if recs:
            # Add 3D positions for new recommendations
            positions = [
                {"x": -6, "y": 1.4, "z": -7},
                {"x": -2, "y": 1.2, "z": -7},
                {"x": 2, "y": 1.6, "z": -7},
                {"x": 6, "y": 1.0, "z": -7},
                {"x": -8, "y": 1.3, "z": 0},
                {"x": -8, "y": 1.5, "z": -3},
                {"x": 8, "y": 1.2, "z": -3},
                {"x": -4, "y": 1.4, "z": -7},
                {"x": 4, "y": 1.3, "z": -7},
                {"x": 0, "y": 1.5, "z": -7},
            ]
            formatted_recs = []
            for i, rec in enumerate(recs[:10]):
                shelf_loc = rec.get("shelf_location", {})
                formatted_recs.append(
                    {
                        "id": i + 1,
                        "title": rec.get("title", "Unknown"),
                        "author": rec.get("author", "Unknown"),
                        "genre": rec.get("genre", "Book"),
                        "score": rec.get("similarity_score", rec.get("score", 0.5)),
                        "cover_url": rec.get("cover_url", ""),
                        "description": rec.get("description", ""),
                        "shelf_location": {
                            "section": shelf_loc.get("section", chr(65 + i)),
                            "row": shelf_loc.get("row", i + 1),
                            "shelf": shelf_loc.get("shelf", i + 1),
                            "position_3d": (
                                positions[i] if i < len(positions) else positions[0]
                            ),
                        },
                        "pitch": f"A great {rec.get('genre', 'book').lower()} pick for you!",
                    }
                )
            yield f"data: {json_module.dumps({'type': 'recommendations', 'books': formatted_recs})}\n\n"

        yield 'data: {"type": "done"}\n\n'

    # Check if client wants streaming
    if request.headers.get("Accept") == "text/event-stream":
        return Response(generate(), mimetype="text/event-stream")

    # Fallback to regular JSON response
    result = agent.chat(message, student_id)
    return jsonify(
        {
            "response": result.get("response", "I'm not sure how to help with that."),
            "books": result.get("recommendations", []),
        }
    )


@app.route("/api/visual-search", methods=["POST"])
def visual_search_endpoint():
    """Analyze a book cover and find similar books."""
    data = request.json
    student_id = data.get("student_id")
    image_base64 = data.get("image")

    # Handle grade-based virtual users
    if student_id and student_id.startswith("grade-"):
        grade = student_id.replace("grade-", "")
        reading_level = get_reading_level_from_grade(grade)
        if student_id not in engine.students:
            engine.students[student_id] = {
                "student_id": student_id,
                "name": "Reader",
                "grade": int(grade) if grade != "K" else 0,
                "reading_level": reading_level,
                "favorite_genres": [],
                "favorite_themes": [],
            }
    elif not student_id or student_id not in engine.students:
        return jsonify({"error": "Invalid student"}), 400

    if not image_base64:
        return jsonify({"error": "No image provided"}), 400

    # Decode base64 image
    try:
        image_bytes = base64.b64decode(
            image_base64.split(",")[1] if "," in image_base64 else image_base64
        )
    except Exception:
        return jsonify({"error": "Invalid image data"}), 400

    # Analyze the cover
    analysis = visual_search.analyze_book_cover(image_bytes)

    # Find similar books
    similar_result = visual_search.find_similar_books(analysis, engine, student_id)
    similar_books = similar_result.get("similar_books", [])

    return jsonify(
        {
            "analysis": {
                "title": analysis.get("title", "Unknown"),
                "author": analysis.get("author", "Unknown"),
                "genres": analysis.get("genre_hints", []),
                "themes": analysis.get("themes", []),
            },
            "similar_books": similar_books[:10],
        }
    )


if __name__ == "__main__":
    logger.info(
        "BookBuddy Virtual Library - Open http://127.0.0.1:5001 in your browser"
    )
    app.run(host="0.0.0.0", debug=False, port=5001)
