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

from sqlalchemy import select
from chromadb.utils import embedding_functions

from app.core.database import SessionLocal
from app.models.db_models import CodeChunk

# We still use ChromaDB's default embedding function locally (all-MiniLM-L6-v2)
_embedding_function = embedding_functions.DefaultEmbeddingFunction()

def retrieve_relevant_chunks(
    query: str,
    repo_name: str,
    top_k: int = TOP_K_RESULTS,
    min_score: float = MIN_SIMILARITY_SCORE,
) -> list[RetrievedChunk]:
    """
    Searches Postgres for the most relevant code chunks for a given query using pgvector.
    """
    # 1. Embed the query
    query_embeddings = _embedding_function([query])
    if not query_embeddings:
        logger.warning("Failed to embed query.")
        return []
    
    query_embedding = query_embeddings[0]

    retrieved = []

    with SessionLocal() as db:
        # Check if we have any chunks for this repo
        count = db.query(CodeChunk).filter_by(repo_name=repo_name).count()
        if count == 0:
            logger.warning("No chunks found for repo '%s'.", repo_name)
            return []

        actual_k = min(top_k, count)
        
        logger.info(
            "Searching Postgres CodeChunks '%s' (%d chunks) for: '%s...'",
            repo_name,
            count,
            query[:60],
        )

        # 2. Perform the semantic search using pgvector cosine_distance
        # cosine_distance in pgvector works identically to ChromaDB (0=identical, 1=opposite)
        # We query the distance and the chunk
        distance_col = CodeChunk.embedding.cosine_distance(query_embedding).label("distance")
        
        results = db.execute(
            select(CodeChunk, distance_col)
            .filter_by(repo_name=repo_name)
            .order_by(distance_col)
            .limit(actual_k)
        ).all()

        for chunk, distance in results:
            similarity = 1.0 - distance

            if similarity < min_score:
                logger.debug(
                    "Skipping chunk '%s' (score=%.3f < min=%.3f)",
                    chunk.chunk_id, similarity, min_score,
                )
                continue

            retrieved.append(RetrievedChunk(
                chunk_id=chunk.chunk_id,
                file_path=chunk.file_path,
                content=chunk.content,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                score=round(similarity, 4),
            ))

    logger.info(
        "Retrieved %d relevant chunks (from %d candidates, min_score=%.2f)",
        len(retrieved),
        actual_k,
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
