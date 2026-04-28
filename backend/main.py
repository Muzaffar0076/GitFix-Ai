"""
backend/main.py
────────────────
The entry point for the GitFix AI FastAPI application.

HOW TO RUN (from the backend/ directory):
  uvicorn main:app --reload --port 8000

  --reload  → Auto-restarts when you save a file (great for development)
  main:app  → "In main.py, find the variable named `app`"

WHAT HAPPENS AT STARTUP:
  1. Python imports this file
  2. FastAPI() creates the app
  3. Middleware (CORS) is attached
  4. Routers are registered
  5. uvicorn starts the event loop and begins accepting HTTP connections
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.dashboard import router as dashboard_router
from app.core.config import get_settings
from app.core.logger import logger


# ── Lifespan: runs setup ONCE at startup, cleanup at shutdown ─────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    CONCEPT — Lifespan:
    This is an async context manager. Everything BEFORE `yield` runs at startup.
    Everything AFTER `yield` runs at shutdown.

    Think of it like:
      try:
          # startup code
          yield
      finally:
          # shutdown code
    """
    settings = get_settings()

    # Create required directories if they don't exist yet
    os.makedirs(settings.REPOS_CLONE_PATH, exist_ok=True)
    os.makedirs(settings.CHROMA_DB_PATH, exist_ok=True)

    logger.info("GitFix AI backend started.")
    logger.info("Repos path : %s", settings.REPOS_CLONE_PATH)
    logger.info("ChromaDB   : %s", settings.CHROMA_DB_PATH)

    yield  # ← App runs here (handling requests)

    logger.info("GitFix AI backend shutting down.")


# ── Create the FastAPI Application ────────────────────────────────────────────
app = FastAPI(
    title="GitFix AI",
    description="Autonomous bug-fixing agent powered by Claude + RAG + Docker",
    version="0.1.0",
    lifespan=lifespan,
)


# ── CORS Middleware ───────────────────────────────────────────────────────────
# CONCEPT — CORS (Cross-Origin Resource Sharing):
# Your React app (port 5173) talks to this API (port 8000).
# Browsers block cross-origin requests by default for security.
# This middleware whitelists our React dev server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # Create React App (fallback)
    ],
    allow_credentials=True,
    allow_methods=["*"],           # GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],
)


# ── Health Check ──────────────────────────────────────────────────────────────
# A simple endpoint to confirm the server is alive.
# Used by Docker health checks and monitoring tools.
@app.get("/api/health", tags=["health"])
def health_check():
    return {"status": "ok", "service": "GitFix AI"}


# ── Placeholder: Routers will be added here in the next phases ───────────────
app.include_router(dashboard_router)
