# GitFix AI Architecture

This document describes the high-level architecture of GitFix AI, an autonomous bug-fixing agent.

## Core Components

### 1. Orchestrator (`backend/app/agent/orchestrator.py`)
The "brain" of the system. It coordinates the execution flow between different modules. It handles the lifecycle of a "fix run," from URL parsing to PR creation.

### 2. RAG Engine (`backend/app/rag/`)
Responsible for codebase awareness. 
- **Chunker:** Splits files into logical segments based on syntax (classes and methods).
- **Embedder:** Generates vector embeddings for each chunk.
- **Vector Store:** Uses ChromaDB for persistent storage and semantic search.

### 3. LLM Layer (`backend/app/llm/`)
Provides a unified interface to interact with multiple LLMs (Groq, Gemini, Claude). It handles prompt construction, response parsing, and error handling.

### 4. Sandbox Environment (`backend/app/sandbox/`)
Ensures safe execution of generated code.
- **Docker Runner:** Spins up isolated containers.
- **Test Executor:** Runs specific test suites (`pytest`) and captures output.

### 5. GitHub Integration (`backend/app/github/`)
Handles all external interactions with GitHub.
- **Repo Manager:** Clones and updates local copies of repositories.
- **PR Creator:** Branches the code, commits patches, and opens Pull Requests.

## Data Flow

1. **Request:** Frontend sends an Issue URL.
2. **Context:** Orchestrator retrieves relevant code via RAG.
3. **Plan:** LLM analyzes the context and generates a patch.
4. **Verify:** Sandbox applies the patch and runs tests.
5. **Retry:** If tests fail, the LLM is prompted to fix the code using the error logs.
6. **Deploy:** Once tests pass, a PR is opened.

## Tech Stack
- **Backend:** FastAPI, Pydantic, SQLAlchemy
- **Frontend:** React, Vite, Tailwind CSS
- **Database:** ChromaDB (Vector), SQLite (Relational)
- **Containerization:** Docker
- **LLMs:** Groq (Llama 3), Google Gemini, Anthropic Claude
