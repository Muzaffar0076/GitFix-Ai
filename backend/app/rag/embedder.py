"""
app/rag/embedder.py
────────────────────
Takes the chunks from chunker.py and stores them in ChromaDB
as vector embeddings — making them semantically searchable.

CONCEPT — What is a Vector Embedding?
  An embedding model converts text → a list of numbers (a vector).
  Similar text → similar vectors (close together in space).

  Example:
    "def login(user, password):"  → [0.12, -0.45, 0.87, ...]  (384 numbers)
    "def authenticate(usr, pwd):" → [0.11, -0.43, 0.85, ...]  ← very close!
    "print('hello world')"        → [0.92,  0.33, -0.21, ...]  ← very different

  ChromaDB uses this to find code chunks SEMANTICALLY related to an issue,
  even if the exact words don't match.

CONCEPT — What is ChromaDB?
  ChromaDB is a vector database — it stores embeddings and lets you search
  by similarity. Think of it like a "nearest neighbour" search engine.

  Instead of:  WHERE content LIKE '%login%'    (keyword search)
  We do:       "find chunks semantically similar to this issue text"

CONCEPT — Embedding Model Used:
  We use ChromaDB's built-in default: "all-MiniLM-L6-v2"
  - Runs 100% locally (no API calls, no cost)
  - Produces 384-dimensional vectors
  - Fast and good at code/technical text
  - Downloads automatically on first run (~90 MB)

FLOW:
  chunks (list of dicts from chunker.py)
      ↓
  get_or_create_collection()  → get/create a "collection" (table) in ChromaDB
      ↓
  embed_chunks()              → store all chunks with auto-generated embeddings
      ↓
  ChromaDB persists to disk   → saved at CHROMA_DB_PATH
"""

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import get_settings
from app.core.logger import logger


# ── 1. Get or Create a ChromaDB Collection ────────────────────────────────────

def get_or_create_collection(repo_name: str) -> chromadb.Collection:
    """
    Connects to ChromaDB and returns a collection for this repository.

    CONCEPT — ChromaDB Collection:
    A "collection" in ChromaDB is like a table in SQL — it stores all
    the embeddings for ONE repository. Each repo gets its own collection
    so they don't mix together.

    Collection name example: "octocat_Hello-World"

    CONCEPT — Persistent Client vs In-Memory:
    chromadb.PersistentClient(path=...) → saves data to disk
    chromadb.Client()                   → only keeps data in RAM (lost on restart)

    We use PersistentClient so embeddings survive server restarts.
    The data is saved at CHROMA_DB_PATH (default: ./chroma_store).

    Args:
        repo_name: The unique name for this repo (e.g. "octocat_Hello-World").

    Returns:
        A ChromaDB Collection object ready for adding/querying documents.
    """
    settings = get_settings()

    # Connect to ChromaDB with persistent storage
    client = chromadb.PersistentClient(
        path=settings.CHROMA_DB_PATH,
        settings=ChromaSettings(anonymized_telemetry=False),
        # anonymized_telemetry=False → don't send usage data to ChromaDB servers
    )

    # get_or_create_collection → gets the collection if it exists, creates if not
    # The default embedding_function uses "all-MiniLM-L6-v2" automatically
    collection = client.get_or_create_collection(
        name=repo_name,
        metadata={"hnsw:space": "cosine"},
        # "cosine" distance metric → measures angle between vectors
        # (better for text similarity than euclidean distance)
    )

    logger.info(
        "ChromaDB collection '%s' ready (%d docs already stored).",
        repo_name,
        collection.count(),
    )

    return collection


# ── 2. Embed and Store All Chunks ─────────────────────────────────────────────

def embed_chunks(chunks: list[dict], repo_name: str) -> None:
    """
    Takes all chunks from chunker.py and upserts them into ChromaDB.

    CONCEPT — Upsert vs Insert:
    "Upsert" = Update if exists, Insert if not.
    We use upsert instead of add so re-running on the same repo
    doesn't create duplicate embeddings.

    CONCEPT — Batching:
    ChromaDB has a limit on how many documents you can add at once.
    We process in batches of BATCH_SIZE (200) to avoid hitting that limit
    and to show progress for large repos.

    CONCEPT — What ChromaDB stores per chunk:
      ids        → unique string ID (our chunk_id)
      documents  → the raw text (ChromaDB embeds this automatically)
      metadatas  → extra info we can filter/return (file_path, line numbers)

    Args:
        chunks: List of chunk dicts from chunk_repository().
        repo_name: Used to get/create the correct collection.
    """
    if not chunks:
        logger.warning("No chunks to embed — skipping.")
        return

    collection = get_or_create_collection(repo_name)

    # Check which chunk IDs are already in ChromaDB (to skip re-embedding)
    existing_ids = set()
    try:
        existing = collection.get(include=[])  # Only fetch IDs, not content
        existing_ids = set(existing["ids"])
    except Exception:
        pass  # Collection is empty — that's fine

    # Filter out chunks that are already embedded
    new_chunks = [c for c in chunks if c["chunk_id"] not in existing_ids]

    if not new_chunks:
        logger.info("All %d chunks already embedded. Nothing to do.", len(chunks))
        return

    logger.info(
        "Embedding %d new chunks (skipping %d already stored)...",
        len(new_chunks),
        len(existing_ids),
    )

    # Process in batches of 200
    BATCH_SIZE = 200
    total_batches = (len(new_chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(total_batches):
        start = batch_num * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(new_chunks))
        batch = new_chunks[start:end]

        # Separate the chunk dict into the 3 things ChromaDB needs
        ids = [c["chunk_id"] for c in batch]

        documents = [c["content"] for c in batch]
        # ChromaDB will auto-embed these using all-MiniLM-L6-v2

        metadatas = [
            {
                "file_path":  c["file_path"],
                "start_line": c["start_line"],
                "end_line":   c["end_line"],
            }
            for c in batch
        ]
        # Metadata lets us return file path + line numbers with search results

        # Upsert this batch into ChromaDB
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        logger.info(
            "Embedded batch %d/%d (%d chunks)",
            batch_num + 1,
            total_batches,
            len(batch),
        )

    logger.info(
        "✅ Embedding complete. Collection '%s' now has %d total chunks.",
        repo_name,
        collection.count(),
    )


# ── 3. Master Function: Chunk + Embed in One Call ─────────────────────────────

def embed_repository(repo_path: str, repo_name: str) -> int:
    """
    Full pipeline: chunks the repo then embeds everything into ChromaDB.

    This is the single function the orchestrator calls.
    It combines chunker.py + embedder.py into one step.

    Args:
        repo_path: Absolute path to the cloned repository.
        repo_name: Unique identifier for this repo.

    Returns:
        Total number of chunks now stored in ChromaDB.
    """
    # Import here to avoid circular imports
    from app.rag.chunker import chunk_repository

    logger.info("Starting embedding pipeline for repo: %s", repo_name)

    # Step 1: Chunk all source files
    chunks = chunk_repository(repo_path, repo_name)

    if not chunks:
        logger.warning("No chunks generated — repo may have no supported source files.")
        return 0

    # Step 2: Embed and store in ChromaDB
    embed_chunks(chunks, repo_name)

    # Return total count in the collection
    collection = get_or_create_collection(repo_name)
    return collection.count()
