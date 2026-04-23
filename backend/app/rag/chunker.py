"""
app/rag/chunker.py
───────────────────
Walks a cloned repository, reads all supported source files,
and splits them into overlapping text chunks for embedding.

WHY DO WE CHUNK?
  A vector database stores fixed-size pieces of text ("chunks"), not entire files.
  Chunking lets us:
    1. Store large codebases in a searchable vector DB
    2. Retrieve only the RELEVANT pieces for a given issue (not 500 whole files)
    3. Stay within LLM token limits when building prompts

OVERLAP CONCEPT:
  If a function starts at line 58 and our chunk ends at line 60, without overlap
  we'd only see 2 lines of that function. With a 10-line overlap, the next chunk
  starts at line 50, capturing the full function in context.

  Chunk 1: lines  1–60
  Chunk 2: lines 50–110  ← starts 10 lines before Chunk 1 ends
  Chunk 3: lines 100–160
  ...

OUTPUT FORMAT:
  Each chunk is a dict:
  {
    "chunk_id":   "owner_repo::src/auth/login.py::chunk_0",
    "file_path":  "src/auth/login.py",
    "content":    "def login(user, password):\n    ...",
    "start_line": 1,
    "end_line":   60,
  }
"""

import os
from typing import Generator

from app.core.constants import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SUPPORTED_EXTENSIONS,
    SKIP_DIRS,
)
from app.core.logger import logger


# ── 1. Walk the Repo and Yield File Paths ─────────────────────────────────────

def iter_source_files(repo_path: str) -> Generator[str, None, None]:
    """
    Walks the repository directory tree and yields relative paths to
    all source code files we care about.

    CONCEPT — os.walk():
    os.walk() is a Python generator that recursively visits every folder.
    For each folder it gives you:
      - dirpath  : current folder path
      - dirnames : list of sub-folder names (we can MODIFY this to skip folders)
      - filenames: list of files in this folder

    By removing items from `dirnames` in-place, we tell os.walk to SKIP
    those directories entirely (e.g. node_modules, .git, venv).

    Args:
        repo_path: Absolute path to the cloned repository root.

    Yields:
        Relative file paths (e.g. "src/auth/login.py")
    """
    for dirpath, dirnames, filenames in os.walk(repo_path):

        # ── Skip unwanted directories ──────────────────────────────────────────
        # Modifying dirnames IN PLACE tells os.walk not to recurse into them.
        # This is more efficient than checking after the fact.
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".")
        ]

        for filename in filenames:
            # Get the file extension (e.g. ".py", ".js")
            _, ext = os.path.splitext(filename)

            if ext in SUPPORTED_EXTENSIONS:
                # Build the full path, then convert to relative path
                full_path = os.path.join(dirpath, filename)
                relative_path = os.path.relpath(full_path, repo_path)
                # os.path.relpath() converts:
                #   "/abs/path/to/repo/src/auth/login.py"
                #   → "src/auth/login.py"

                yield relative_path


# ── 2. Split a Single File into Chunks ────────────────────────────────────────

def chunk_file(
    file_path: str,
    content: str,
    repo_name: str,
) -> list[dict]:
    """
    Splits a source file's content into overlapping line-based chunks.

    Args:
        file_path: Relative path of the file (e.g. "src/auth/login.py")
        content:   Full text content of the file.
        repo_name: Used to build unique chunk IDs.

    Returns:
        A list of chunk dicts, each with chunk_id, file_path, content,
        start_line, end_line.
    """
    lines = content.splitlines()
    # splitlines() → ["line1", "line2", ...] — splits on \n, \r\n, etc.

    total_lines = len(lines)
    chunks = []
    chunk_index = 0

    # Slide a window of CHUNK_SIZE lines, stepping by (CHUNK_SIZE - CHUNK_OVERLAP)
    # CHUNK_SIZE    = 60  (from constants.py)
    # CHUNK_OVERLAP = 10
    # Step size     = 60 - 10 = 50 lines per step

    step = CHUNK_SIZE - CHUNK_OVERLAP  # How many lines to advance each iteration

    for start in range(0, total_lines, step):
        end = min(start + CHUNK_SIZE, total_lines)
        # min() prevents going past the end of the file

        chunk_lines = lines[start:end]
        chunk_content = "\n".join(chunk_lines)

        # Skip empty chunks (e.g. blank files or trailing whitespace)
        if not chunk_content.strip():
            continue

        # Build a unique ID for this chunk
        # Format: "repo_name::file_path::chunk_index"
        # e.g.:   "octocat_Hello-World::src/auth/login.py::chunk_0"
        chunk_id = f"{repo_name}::{file_path}::chunk_{chunk_index}"

        chunks.append({
            "chunk_id":   chunk_id,
            "file_path":  file_path,
            "content":    chunk_content,
            "start_line": start + 1,       # 1-indexed (humans count from 1)
            "end_line":   end,
        })

        chunk_index += 1

        # If we've reached the end of the file, stop
        if end >= total_lines:
            break

    return chunks


# ── 3. Chunk the Entire Repository ────────────────────────────────────────────

def chunk_repository(repo_path: str, repo_name: str) -> list[dict]:
    """
    Walks the entire repository and returns all chunks from all source files.

    This is the main function called by the RAG pipeline.

    Args:
        repo_path: Absolute path to the cloned repository.
        repo_name: A unique name for this repo (used in chunk IDs).
                   e.g. "octocat_Hello-World"

    Returns:
        A flat list of all chunks from all supported source files.
        e.g. 200 files × ~8 chunks each = ~1600 chunks total.
    """
    all_chunks = []
    files_processed = 0
    files_skipped = 0

    for relative_path in iter_source_files(repo_path):
        full_path = os.path.join(repo_path, relative_path)

        try:
            # Read file with UTF-8 encoding, skip files with encoding errors
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Skip empty files
            if not content.strip():
                files_skipped += 1
                continue

            # Split into chunks and add to our list
            file_chunks = chunk_file(relative_path, content, repo_name)
            all_chunks.extend(file_chunks)
            files_processed += 1

        except OSError as e:
            # File might be a binary file disguised with a .py extension, etc.
            logger.warning("Could not read file %s: %s", relative_path, str(e))
            files_skipped += 1

    logger.info(
        "Chunking complete: %d files → %d chunks (%d files skipped)",
        files_processed,
        len(all_chunks),
        files_skipped,
    )

    return all_chunks
