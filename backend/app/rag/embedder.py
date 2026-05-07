"""
app/rag/embedder.py
────────────────────
Takes the chunks from chunker.py, generates embeddings using Chroma's default model,
and stores them in PostgreSQL using pgvector.
"""

from chromadb.utils import embedding_functions
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.core.database import SessionLocal
from app.models.db_models import CodeChunk
from app.core.logger import logger

# We still use ChromaDB's default embedding function locally (all-MiniLM-L6-v2)
# to avoid heavy PyTorch dependencies while getting great quality.
_embedding_function = embedding_functions.DefaultEmbeddingFunction()

def embed_chunks(chunks: list[dict], repo_name: str) -> None:
    """
    Takes all chunks from chunker.py, generates embeddings, and inserts them into PostgreSQL.
    """
    if not chunks:
        logger.warning("No chunks to embed — skipping.")
        return

    with SessionLocal() as db:
        # Get existing chunk IDs for this repo
        existing = db.scalars(
            select(CodeChunk.chunk_id).filter_by(repo_name=repo_name)
        ).all()
        existing_ids = set(existing)

        # Filter out chunks that are already embedded
        new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

        if not new_chunks:
            logger.info("All %d chunks already embedded in Postgres. Nothing to do.", len(chunks))
            return

        logger.info(
            "Embedding %d new chunks (skipping %d already stored)...",
            len(new_chunks),
            len(existing_ids),
        )

        # Process in batches
        BATCH_SIZE = 200
        total_batches = (len(new_chunks) + BATCH_SIZE - 1) // BATCH_SIZE

        for batch_num in range(total_batches):
            start = batch_num * BATCH_SIZE
            end = min(start + BATCH_SIZE, len(new_chunks))
            batch = new_chunks[start:end]

            documents = [c["content"] for c in batch]
            
            # Generate embeddings for the batch
            embeddings = _embedding_function(documents)

            # Create CodeChunk objects
            db_chunks = []
            for i, chunk in enumerate(batch):
                db_chunk = CodeChunk(
                    repo_name=repo_name,
                    chunk_id=chunk["chunk_id"],
                    content=chunk["content"],
                    file_path=chunk["file_path"],
                    start_line=chunk["start_line"],
                    end_line=chunk["end_line"],
                    embedding=embeddings[i]
                )
                db_chunks.append(db_chunk)

            db.add_all(db_chunks)
            db.commit()

            logger.info(
                "Embedded and saved batch %d/%d (%d chunks)",
                batch_num + 1,
                total_batches,
                len(batch),
            )

        total_count = db.query(CodeChunk).filter_by(repo_name=repo_name).count()
        logger.info(
            "✅ Embedding complete. Database now has %d total chunks for '%s'.",
            total_count,
            repo_name,
        )

def embed_repository(repo_path: str, repo_name: str) -> int:
    """
    Full pipeline: chunks the repo then embeds everything into Postgres.
    """
    from app.rag.chunker import chunk_repository

    logger.info("Starting embedding pipeline for repo: %s", repo_name)

    chunks = chunk_repository(repo_path, repo_name)

    if not chunks:
        logger.warning("No chunks generated — repo may have no supported source files.")
        return 0

    embed_chunks(chunks, repo_name)

    with SessionLocal() as db:
        return db.query(CodeChunk).filter_by(repo_name=repo_name).count()
