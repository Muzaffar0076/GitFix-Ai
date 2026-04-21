"""
app/core/constants.py
──────────────────────
Application-wide constants.

WHY THIS FILE EXISTS:
  Instead of scattering magic numbers like `3` or `500` across all files,
  we define them once here with clear names. This makes the code readable
  and easy to tune without hunting through multiple files.
"""

# ── RAG Pipeline ──────────────────────────────────────────────────────────────
CHUNK_SIZE = 60              # Number of lines per code chunk sent to ChromaDB
CHUNK_OVERLAP = 10           # Lines shared between consecutive chunks
                             # (overlap helps preserve context at boundaries)
TOP_K_RESULTS = 5            # How many relevant code chunks to retrieve per query

# ── LLM ───────────────────────────────────────────────────────────────────────
MAX_LLM_TOKENS = 8192        # Max tokens Claude can return in one response
LLM_TEMPERATURE = 0.2        # 0 = deterministic/focused, 1 = creative/random
                             # Bug fixing needs low temperature — we want precision

# ── Agent ─────────────────────────────────────────────────────────────────────
MAX_RETRIES = 3              # Max self-healing attempts before giving up

# ── File Filtering (RAG chunker) ──────────────────────────────────────────────
# Only these file types will be read and embedded into ChromaDB
SUPPORTED_EXTENSIONS = {
    ".py",    # Python
    ".js",    # JavaScript
    ".ts",    # TypeScript
    ".jsx",   # React JSX
    ".tsx",   # React TSX
    ".java",  # Java
    ".go",    # Go
    ".rb",    # Ruby
    ".cpp",   # C++
    ".c",     # C
    ".cs",    # C#
    ".rs",    # Rust
    ".php",   # PHP
}

# Directories to always skip during repo traversal
SKIP_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    "dist",
    "build",
    ".next",
    "vendor",
}

# ── Git / PR ──────────────────────────────────────────────────────────────────
BRANCH_PREFIX = "gitfix/issue-"  # All our branches look like: gitfix/issue-42
