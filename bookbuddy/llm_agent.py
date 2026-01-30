"""
LLM chat with tool calling. Uses OpenAI GPT-5-mini with function calling for personalized book recommendations.
"""

import json
import os
import re
from typing import Dict, Any, List, Optional

import numpy as np
from openai import OpenAI
from loguru import logger

from bookbuddy.recommendation_engine import get_engine


class BookBuddyAgent:
    """
    Conversational AI agent for book recommendations.

    Capabilities:
    - Natural language understanding of book preferences
    - Tool calling for recommendations, search, and visual analysis
    - Personalized "book talk" generation
    - Context-aware conversation
    """

    TOOLS = [
        {
            "name": "semantic_search",
            "description": "Find books using natural language. Use this for any book request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language description of what kind of books to find",
                    },
                    "expanded_terms": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Synonyms and related words to boost search (e.g., for 'mischief' add ['pranks', 'troublemaking', 'tricks'])",
                    },
                    "exclude_titles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Book titles to exclude",
                    },
                    "must_contain": {
                        "type": "string",
                        "description": "Keyword that MUST appear in book title or description (for strict filtering like 'only books with captains')",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "get_similar_books",
            "description": "Find books similar to a specific book the user mentions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "book_title": {"type": "string", "description": "Title of the book"}
                },
                "required": ["book_title"],
            },
        },
        {
            "name": "search_by_title",
            "description": "Look up information about a specific book to answer questions about it. Use this when user asks questions like 'Does X have pigs?' or 'Tell me about X'. This does NOT update recommendations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Book title to look up"}
                },
                "required": ["title"],
            },
        },
    ]

    SYSTEM_PROMPT = """You are BookBuddy, a friendly school librarian assistant.

RULES:
- Reply in 1 short sentence only
- NEVER list books (they show in the panel)
- NEVER output JSON, code, or technical terms
- NEVER mention tool names or narrate actions
- NEVER use em dashes
- Vary your responses naturally - don't repeat the same phrase twice

When user asks to find books: use semantic_search with expanded_terms
When user asks to filter/keep only certain books: use must_contain parameter
When user asks to remove books: use exclude_titles parameter
When user asks about a book: use search_by_title, answer directly

QUERY EXPANSION: When using semantic_search, always include expanded_terms with synonyms and related words to improve matches.
Examples:
- "mischief" -> expanded_terms: ["pranks", "troublemaking", "tricks", "naughty", "misbehavior"]
- "scary" -> expanded_terms: ["spooky", "frightening", "creepy", "horror", "terrifying"]
- "funny" -> expanded_terms: ["humor", "comedy", "hilarious", "silly", "laugh"]

You can only: recommend books, answer book questions, give shelf locations."""

    def __init__(
        self, recommendation_engine, visual_search=None, use_openai: bool = True
    ):
        """
        Initialize the BookBuddy agent.
        """
        self.engine = recommendation_engine
        self.visual_search = visual_search
        self.use_openai = use_openai

        if use_openai:
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        else:
            self.client = None

        self.conversations: Dict[str, List[Dict]] = {}

    def _get_conversation(self, session_id: str) -> List[Dict]:
        """Get or create a conversation history for a session."""
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        return self.conversations[session_id]

    def _execute_tool(
        self, tool_name: str, arguments: Dict, student_id: str = None
    ) -> Dict[str, Any]:
        """Execute a tool and return results."""
        if tool_name == "semantic_search":
            query = arguments["query"]
            exclude_titles = arguments.get("exclude_titles", [])
            must_contain = arguments.get("must_contain", "")
            expanded_terms = arguments.get("expanded_terms", [])

            if self.visual_search:
                # Use the embedding model from visual_search for semantic search
                results = self._semantic_book_search(
                    query,
                    student_id,
                    n=10,
                    exclude_titles=exclude_titles,
                    must_contain=must_contain,
                    expanded_terms=expanded_terms,
                )
                return {
                    "results": results,
                    "count": len(results),
                    "query": query,
                    "excluded": exclude_titles,
                }
            else:
                # Fallback to keyword search if no embedding model
                results = self.engine.find_book_by_title(query)
                return {"results": results[:10], "count": len(results)}

        elif tool_name == "get_similar_books":
            title = arguments["book_title"]
            matches = self.engine.find_book_by_title(title)
            if matches:
                book_id = matches[0]["book_id"]
                similar = self.engine.get_similar_books(book_id)
                return {"original_book": matches[0]["title"], "similar_books": similar}
            return {"error": f"Book '{title}' not found"}

        elif tool_name == "search_by_title":
            title = arguments["title"]
            results = self.engine.find_book_by_title(title)
            # Return as "book_info" not "results" so it doesn't replace recommendations
            # This is for answering questions about specific books
            return {"book_info": results, "count": len(results)}

        return {"error": f"Unknown tool: {tool_name}"}

    def _semantic_book_search(
        self,
        query: str,
        student_id: str,
        n: int = 10,
        exclude_titles: List[str] = None,
        must_contain: str = "",
        expanded_terms: List[str] = None,
    ) -> List[Dict]:
        """Search books using semantic embeddings, filtered by reading level FIRST."""
        exclude_titles = exclude_titles or []
        expanded_terms = expanded_terms or []
        # Normalize excluded titles for comparison
        exclude_titles_lower = [t.lower().strip() for t in exclude_titles]
        # Normalize must_contain keyword
        must_contain_lower = must_contain.lower().strip() if must_contain else ""

        # Get student's reading level for filtering
        reading_level = None
        acceptable_levels = None
        if student_id and student_id in self.engine.students:
            reading_level = self.engine.students[student_id].get("reading_level")
            acceptable_levels = self.visual_search._get_acceptable_levels(reading_level)
        elif student_id and student_id.startswith("grade-"):
            # Handle grade-based virtual users
            grade = student_id.replace("grade-", "")
            grade_num = 0 if grade == "K" else int(grade)
            if grade_num <= 2:
                reading_level = "K-2"
            elif grade_num <= 5:
                reading_level = "3-5"
            elif grade_num <= 8:
                reading_level = "6-8"
            else:
                reading_level = "9-12"
            acceptable_levels = self.visual_search._get_acceptable_levels(reading_level)

        # Load or build embeddings for all books
        book_ids, book_embeddings = self.visual_search._load_or_build_embeddings(
            self.engine.catalog
        )

        # FILTER BY READING LEVEL FIRST - only consider books in acceptable grade range
        # Also exclude any books the user doesn't want
        # And filter by must_contain keyword if specified
        filtered_indices = []
        filtered_book_ids = []
        for idx, bid in enumerate(book_ids):
            book = self.engine.catalog.get(bid, {})
            # Check reading level
            if acceptable_levels and book.get("reading_level") not in acceptable_levels:
                continue
            # Check if excluded
            if book.get("title", "").lower().strip() in exclude_titles_lower:
                continue
            # Check must_contain keyword (in title, description, or themes)
            if must_contain_lower:
                title = book.get("title", "").lower()
                desc = book.get("description", "").lower()
                themes = " ".join(book.get("themes", [])).lower()
                if (
                    must_contain_lower not in title
                    and must_contain_lower not in desc
                    and must_contain_lower not in themes
                ):
                    continue
            filtered_indices.append(idx)
            filtered_book_ids.append(bid)

        if not filtered_indices:
            return []

        # Use only filtered embeddings
        filtered_embeddings = book_embeddings[filtered_indices]

        # Combine query with expanded terms for richer embedding
        search_text = query
        if expanded_terms:
            search_text = f"{query} {' '.join(expanded_terms)}"

        # Embed the query (with expanded terms if provided)
        query_embedding = self.visual_search.embedding_model.encode(
            [search_text], convert_to_numpy=True
        )[0]

        # Compute cosine similarities only for filtered books
        similarities = np.dot(filtered_embeddings, query_embedding) / (
            np.linalg.norm(filtered_embeddings, axis=1)
            * np.linalg.norm(query_embedding)
            + 1e-8
        )

        # Get top matches sorted by similarity (from the pre-filtered set)
        top_indices = np.argsort(similarities)[::-1][:n]

        results = []
        for idx in top_indices:
            bid = filtered_book_ids[idx]
            book = self.engine.catalog.get(bid, {})

            results.append(
                {
                    "book_id": bid,
                    "title": book.get("title", "Unknown"),
                    "author": book.get("author", "Unknown"),
                    "genre": book.get("genre", "Unknown"),
                    "themes": book.get("themes", []),
                    "reading_level": book.get("reading_level", ""),
                    "pages": book.get("pages", 0),
                    "description": book.get("description", ""),
                    "cover_url": book.get("cover_url", ""),
                    "similarity_score": float(similarities[idx]),
                    "shelf_location": book.get("shelf_location", {}),
                    "available_copies": book.get("available_copies", 0),
                    "total_copies": book.get("total_copies", 0),
                }
            )

        return results

    def chat(
        self, message: str, student_id: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process a chat message and return a response."""
        session_id = session_id or student_id
        conversation = self._get_conversation(session_id)
        conversation.append({"role": "user", "content": message})

        if self.use_openai and self.client:
            return self._openai_chat(message, student_id, conversation)
        return self._simulated_chat(message, student_id, conversation)

    def _openai_chat(
        self, message: str, student_id: str, conversation: List[Dict]
    ) -> Dict[str, Any]:
        """Chat using OpenAI API with function calling."""
        openai_functions = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                },
            }
            for tool in self.TOOLS
        ]

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]
        messages.extend(
            {"role": msg["role"], "content": msg["content"]} for msg in conversation
        )

        response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=messages,
            tools=openai_functions,
            tool_choice="auto",
        )

        tool_calls = []
        choice = response.choices[0]

        # If the model wants to call tools, execute them and get final response
        if choice.message.tool_calls:
            # Add assistant message with tool calls
            messages.append(choice.message)

            for tc in choice.message.tool_calls:
                args = json.loads(tc.function.arguments)
                result = self._execute_tool(
                    tc.function.name, args, student_id=student_id
                )
                tool_calls.append(
                    {"name": tc.function.name, "input": args, "result": result}
                )

                # Add tool result to messages
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result),
                    }
                )

            # Get final response with tool results
            final_response_obj = self.client.chat.completions.create(
                model="gpt-5-mini", messages=messages
            )
            final_response = final_response_obj.choices[0].message.content or ""
        else:
            final_response = choice.message.content or ""

        # Sanitize response - remove any JSON that slipped through
        final_response = self._sanitize_response(final_response)

        conversation.append({"role": "assistant", "content": final_response})

        return {
            "response": final_response,
            "tool_calls": tool_calls,
            "recommendations": self._extract_recommendations(tool_calls),
        }

    def _sanitize_response(self, response: str) -> str:
        """Remove any JSON or code that slipped into the response."""
        # Remove JSON objects
        response = re.sub(r'\{[^{}]*"query"[^{}]*\}', "", response)
        response = re.sub(r'\{[^{}]*"results"[^{}]*\}', "", response)
        response = re.sub(r'\{"[^"]+":.*?\}', "", response)
        # Remove any remaining JSON-like structures (nested)
        while '{"' in response or "[{" in response:
            response = re.sub(r"\{[^{}]+\}", "", response)
            response = re.sub(r"\[[^\[\]]+\]", "", response)
        # Clean up extra whitespace
        response = re.sub(r"\s+", " ", response).strip()
        return response

    def _simulated_chat(
        self, message: str, student_id: str, conversation: List[Dict]
    ) -> Dict[str, Any]:
        """Simulated chat for demo without API - uses semantic search."""
        message_lower = message.lower()

        # Check for similar book requests
        if any(
            word in message_lower for word in ["like", "similar to", "loved", "enjoyed"]
        ):
            for book in self.engine.catalog.values():
                if book["title"].lower() in message_lower:
                    tool_result = self._execute_tool(
                        "get_similar_books", {"book_title": book["title"]}, student_id
                    )
                    similar = tool_result.get("similar_books", [])
                    response = (
                        self._generate_book_talk(similar)
                        if similar
                        else f"Let me find books like {book['title']}..."
                    )
                    conversation.append({"role": "assistant", "content": response})
                    return {
                        "response": response,
                        "tool_calls": [
                            {"name": "get_similar_books", "result": tool_result}
                        ],
                        "recommendations": similar,
                    }

        # Default: use semantic search for any book-related query
        if self.visual_search:
            tool_result = self._execute_tool(
                "semantic_search", {"query": message, "num_results": 10}, student_id
            )
            results = tool_result.get("results", [])
            response = (
                self._generate_book_talk(results)
                if results
                else "I'd love to help! What kind of books interest you?"
            )
            conversation.append({"role": "assistant", "content": response})
            return {
                "response": response,
                "tool_calls": [{"name": "semantic_search", "result": tool_result}],
                "recommendations": results,
            }

        response = "Hi! I'm BookBuddy. What kind of books are you looking for?"
        conversation.append({"role": "assistant", "content": response})
        return {"response": response, "tool_calls": []}

    def _generate_book_talk(self, recommendations: List[Dict]) -> str:
        """Generate a brief book talk for recommendations."""
        books = []
        for rec in recommendations[:3]:
            theme = rec.get("themes", ["great story"])[0]
            books.append(f"**{rec['title']}** by {rec['author']} - {theme}")
        return (
            "Here are some picks for you:\n"
            + "\n".join(books)
            + "\n\nWant details on any of these?"
        )

    def _extract_recommendations(self, tool_calls: List[Dict]) -> List[Dict]:
        """Extract recommendations from tool call results."""
        for tc in tool_calls:
            result = tc.get("result", {})
            if "recommendations" in result:
                return result["recommendations"]
            if "similar_books" in result:
                return result["similar_books"]
            if "results" in result:
                return result["results"]
        return []


if __name__ == "__main__":
    engine = get_engine()
    agent = BookBuddyAgent(engine, use_openai=False)

    # Use a grade-based virtual student
    test_student = "grade-5"
    engine.students[test_student] = {
        "student_id": test_student,
        "reading_level": "3-5",
    }

    logger.info("BookBuddy Agent Test (Simulation Mode)")

    queries = [
        "Hi! Can you recommend some books for me?",
        "I really loved Percy Jackson! Can you find something similar?",
        "I want to read books about friendship and adventure",
    ]

    for query in queries:
        logger.info(f"Student: {query}")
        result = agent.chat(query, test_student)
        logger.info(f"BookBuddy: {result['response']}")
        if result.get("tool_calls"):
            logger.debug(f"Tools used: {[tc['name'] for tc in result['tool_calls']]}")
