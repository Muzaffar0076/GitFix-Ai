"""
app/rag/retriever.py
─────────────────────
Searches ChromaDB for code chunks semantically similar to a given query.

This is the "R" in RAG (Retrieval-Augmented Generation).

CONCEPT — How Semantic Search Works:
  1. The query (issue description) is embedded into a vector using the same
     model that embedded the code chunks (all-MiniLM-L6-v2).
  2. ChromaDB computes cosine similarity between the query vector and all
     stored chunk vectors.
  3. The top-N most similar chunks are returned.

  Example:
    Query: "login fails when password contains special characters"
    Returns: chunks from auth/login.py, security/validator.py, etc.
    — even if those files never mention "login fails"!

CONCEPT — Cosine Similarity:
  Two vectors are similar if they point in the same direction.
  Score ranges from 0.0 (completely different) to 1.0 (identical).
  We filter out chunks with score < MIN_SIMILARITY_SCORE (0.3)
  to avoid returning totally unrelated code.

FLOW:
  issue_text (string)
      ↓
  ChromaDB.query()            → embed query + find nearest neighbors
      ↓
  filter by similarity score  → remove irrelevant results
      ↓
  format as RetrievedChunk    → returns list of dicts with content + metadata
"""

from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings
from app.core.constants import TOP_K_RESULTS, MIN_SIMILARITY_SCORE
from app.core.logger import logger


# ── Data Structure for Retrieved Results ──────────────────────────────────────

@dataclass
class RetrievedChunk:
    """
    A single search result returned by the retriever.

    CONCEPT — @dataclass:
    A dataclass is a Python class that auto-generates __init__, __repr__,
    and __eq__ based on the fields you declare. It's cleaner than a plain
    dict because you get type hints and dot-access (result.file_path).

    Fields:
        chunk_id:   Unique identifier (e.g. "repo::src/auth.py::chunk_2")
        file_path:  Relative file path (e.g. "src/auth/login.py")
        content:    The actual code text
        start_line: First line of this chunk in the original file
        end_line:   Last line of this chunk in the original file
        score:      Similarity score 0.0–1.0 (higher = more relevant)
    """
    chunk_id:   str
    file_path:  str
    content:    str
    start_line: int
    end_line:   int
    score:      float


# ── Core Retrieval Function ────────────────────────────────────────────────────

def retrieve_relevant_chunks(
    query: str,
    repo_name: str,
    top_k: int = TOP_K_RESULTS,
    min_score: float = MIN_SIMILARITY_SCORE,
) -> list[RetrievedChunk]:
    """
    Searches ChromaDB for the most relevant code chunks for a given query.

    CONCEPT — ChromaDB .query():
    collection.query(
        query_texts=["your query here"],  → auto-embeds this text
        n_results=10,                     → return top 10 matches
        include=["documents", "metadatas", "distances"]
    )

    ChromaDB returns "distances" (not similarities). With cosine distance:
        distance = 0.0 → identical
        distance = 1.0 → completely different
    We convert to similarity: similarity = 1 - distance

    Args:
        query:     The search query (usually the GitHub issue title + body).
        repo_name: The ChromaDB collection to search in.
        top_k:     How many results to return (default from constants.py).
        min_score: Minimum similarity threshold (chunks below this are filtered).

    Returns:
        List of RetrievedChunk objects, sorted by relevance (highest first).
        Returns empty list if the collection doesn't exist yet.
    """
    settings = get_settings()

    # Connect to ChromaDB
    client = chromadb.PersistentClient(
        path=settings.CHROMA_DB_PATH,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    # Check if the collection exists (repo might not be embedded yet)
    existing_collections = [c.name for c in client.list_collections()]
    if repo_name not in existing_collections:
        logger.warning(
            "Collection '%s' not found in ChromaDB. "
            "Has the repository been embedded yet?",
            repo_name,
        )
        return []

    collection = client.get_collection(name=repo_name)

    # Check if collection has any documents
    if collection.count() == 0:
        logger.warning("Collection '%s' is empty. Nothing to search.", repo_name)
        return []

    # Limit top_k to what's actually available
    actual_k = min(top_k, collection.count())

    logger.info(
        "Searching collection '%s' (%d chunks) for: '%s...'",
        repo_name,
        collection.count(),
        query[:60],  # Only log first 60 chars of query
    )

    # ── Perform the semantic search ────────────────────────────────────────────
    results = collection.query(
        query_texts=[query],       # ChromaDB embeds this automatically
        n_results=actual_k,
        include=["documents", "metadatas", "distances"],
        # documents → the raw code text
        # metadatas → file_path, start_line, end_line
        # distances → cosine distance (0.0 = identical, 1.0 = opposite)
    )

    # ── Parse and Filter Results ───────────────────────────────────────────────
    # ChromaDB returns nested lists because you can query multiple texts at once.
    # We only queried one text, so we access index [0] for each.
    ids        = results["ids"][0]
    documents  = results["documents"][0]
    metadatas  = results["metadatas"][0]
    distances  = results["distances"][0]

    retrieved = []

    for chunk_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
        # Convert cosine distance → similarity score
        # distance=0.0 → similarity=1.0 (perfect match)
        # distance=1.0 → similarity=0.0 (no match)
        similarity = 1.0 - distance

        # Filter out chunks below the minimum similarity threshold
        if similarity < min_score:
            logger.debug(
                "Skipping chunk '%s' (score=%.3f < min=%.3f)",
                chunk_id, similarity, min_score,
            )
            continue

        retrieved.append(RetrievedChunk(
            chunk_id=chunk_id,
            file_path=metadata.get("file_path", "unknown"),
            content=content,
            start_line=int(metadata.get("start_line", 0)),
            end_line=int(metadata.get("end_line", 0)),
            score=round(similarity, 4),
        ))

    # Sort by score descending (most relevant first)
    retrieved.sort(key=lambda c: c.score, reverse=True)

    logger.info(
        "Retrieved %d relevant chunks (from %d candidates, min_score=%.2f)",
        len(retrieved),
        len(ids),
        min_score,
    )

    return retrieved


# ── Format Chunks for LLM Prompt ──────────────────────────────────────────────

def format_chunks_for_prompt(chunks: list[RetrievedChunk]) -> str:
    """
    Formats retrieved chunks into a clean string to inject into the LLM prompt.

    The LLM needs to see the code in a structured way so it understands:
      - Which file the code comes from
      - Which lines it corresponds to (for the patch)
      - The actual code content

    Args:
        chunks: List of RetrievedChunk objects from retrieve_relevant_chunks().

    Returns:
        A formatted string like:

        ### File: src/auth/login.py (lines 45-105, relevance: 0.87)
        ```python
        def login(user, password):
            ...
        ```

        ### File: utils/validator.py (lines 10-70, relevance: 0.72)
        ...
    """
    if not chunks:
        return "No relevant code context found."

    sections = []
    for chunk in chunks:
        # Detect language from file extension for syntax highlighting
        ext = chunk.file_path.rsplit(".", 1)[-1] if "." in chunk.file_path else ""
        lang_map = {
            "py": "python", "js": "javascript", "ts": "typescript",
            "java": "java", "go": "go", "rs": "rust", "cpp": "cpp",
            "c": "c", "rb": "ruby", "php": "php",
        }
        lang = lang_map.get(ext, "")

        section = (
            f"### File: `{chunk.file_path}` "
            f"(lines {chunk.start_line}–{chunk.end_line}, "
            f"relevance: {chunk.score:.2f})\n"
            f"```{lang}\n"
            f"{chunk.content}\n"
            f"```"
        )
        sections.append(section)

    return "\n\n".join(sections)
