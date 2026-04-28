"""
app/agent/orchestrator.py
──────────────────────────
The central brain of GitFix AI. Connects all modules into one pipeline.

THIS IS THE FILE THAT MAKES EVERYTHING WORK TOGETHER.

FLOW (end to end):
  User sends:  POST /api/fix { "issue_url": "https://github.com/..." }
                                      ↓
  orchestrator.run_fix_pipeline()
      │
      ├── Step 1: Parse URL + Fetch Issue      (repo_manager.py)
      ├── Step 2: Clone Repository             (repo_manager.py)
      ├── Step 3: Embed Repository into ChromaDB (embedder.py)
      ├── Step 4: Retrieve Relevant Code       (retriever.py)
      ├── Step 5: Generate Fix with LLM        (llm/client.py)
      ├── Step 6: Apply Patch to File          (patch_applier.py)
      ├── Step 7: Open Pull Request            (pr_creator.py)
      │
      └── Returns: RunLog with PR URL + all event logs

CONCEPT — Why a Separate Orchestrator?
  Each module (chunker, embedder, retriever, etc.) does ONE job.
  The orchestrator's only job is to call them in the right order
  and pass data between them. This is called "separation of concerns."
  If something breaks, you know exactly which step failed.

CONCEPT — RunLog + EventLog:
  Every step logs an EventLog (stage name, message, timestamp).
  All logs are collected into a RunLog — a complete audit trail.
  The API returns this so the user can see exactly what happened.
"""

import traceback
from datetime import datetime, timezone
from typing import Callable
from uuid import uuid4

from app.agent.patch_applier import apply_patch
from app.core.logger import logger
from app.github.pr_creator import create_fix_pr
from app.github.repo_manager import (
    clone_or_pull_repo,
    fetch_issue_details,
    parse_issue_url,
)
from app.llm.client import generate_patch
from app.models.event_log import AgentStage, EventLog, LogLevel
from app.models.run_log import RunLog, RunStatus
from app.rag.embedder import embed_repository
from app.rag.retriever import format_chunks_for_prompt, retrieve_relevant_chunks


# ── Helper: Log an Event ──────────────────────────────────────────────────────

def _log_event(
    logs: list[EventLog],
    stage: AgentStage,
    message: str,
    level: LogLevel = LogLevel.INFO,
    data: dict | None = None,
    event_callback: Callable[[EventLog], None] | None = None,
) -> None:
    """
    Creates an EventLog entry and appends it to the log list.
    Also writes to the application logger for terminal output.

    Args:
        logs:    The running list of events for this pipeline run.
        stage:   Which pipeline stage this event belongs to (enum value).
        message: Human-readable description of what happened.
        level:   INFO, WARNING, or ERROR.
        data:    Optional extra data dict (e.g. {"pr_url": "..."}).
    """
    event = EventLog(
        stage=stage,
        level=level,
        message=message,
        data=data or {},
        timestamp=datetime.now(timezone.utc),
    )
    logs.append(event)
    if event_callback is not None:
        event_callback(event)

    # Mirror to terminal logger
    if level == LogLevel.ERROR:
        logger.error("[%s] %s", stage.value, message)
    elif level == LogLevel.WARNING:
        logger.warning("[%s] %s", stage.value, message)
    else:
        logger.info("[%s] %s", stage.value, message)


# ── Master Pipeline Function ──────────────────────────────────────────────────

def run_fix_pipeline(
    issue_url: str,
    run_id: str | None = None,
    event_callback: Callable[[EventLog], None] | None = None,
) -> RunLog:
    """
    Runs the complete GitFix AI pipeline from URL to Pull Request.

    This function is called directly by the API route handler.
    It handles ALL errors internally and always returns a RunLog,
    even if the pipeline fails partway through.

    Args:
        issue_url: The GitHub issue URL pasted by the user.
                   e.g. "https://github.com/owner/repo/issues/42"

    Returns:
        A RunLog containing:
          - issue details (if fetched successfully)
          - patch details (if generated successfully)
          - pr_url (if PR was opened successfully)
          - all event logs (one per step)
          - final status: "success" or "failed"
    """
    logs: list[EventLog] = []

    # Initialize the RunLog — we'll fill in fields as we go
    run = RunLog(
        run_id=run_id or str(uuid4()),
        issue_url=issue_url,
        status=RunStatus.RUNNING,
        events=logs,
        created_at=datetime.now(timezone.utc),
    )

    logger.info("=" * 60)
    logger.info("GitFix AI Pipeline Started")
    logger.info("Issue URL: %s", issue_url)
    logger.info("=" * 60)

    try:
        # ── STEP 1: Parse the GitHub URL ──────────────────────────────────────
        _log_event(
            logs, AgentStage.PARSING, f"Parsing issue URL: {issue_url}", event_callback=event_callback
        )

        owner, repo_name, issue_number = parse_issue_url(issue_url)

        _log_event(
            logs, AgentStage.PARSING,
            f"Parsed → owner={owner}, repo={repo_name}, issue=#{issue_number}",
            data={"owner": owner, "repo": repo_name, "issue_number": issue_number},
            event_callback=event_callback,
        )

        # ── STEP 2: Fetch Issue from GitHub API ───────────────────────────────
        _log_event(
            logs,
            AgentStage.FETCHING_ISSUE,
            f"Fetching issue #{issue_number} from GitHub...",
            event_callback=event_callback,
        )

        issue = fetch_issue_details(owner, repo_name, issue_number)
        run.issue = issue  # Store on RunLog

        _log_event(
            logs, AgentStage.FETCHING_ISSUE,
            f"Fetched issue: '{issue.title}'",
            data={"title": issue.title, "labels": issue.labels},
            event_callback=event_callback,
        )

        # ── STEP 3: Clone or Update the Repository ────────────────────────────
        repo_name_safe = f"{owner}_{repo_name}"
        _log_event(
            logs, AgentStage.CLONING, f"Cloning/updating {owner}/{repo_name}...", event_callback=event_callback
        )

        repo_path = clone_or_pull_repo(owner, repo_name)

        _log_event(
            logs, AgentStage.CLONING,
            f"Repository ready at: {repo_path}",
            data={"repo_path": repo_path},
            event_callback=event_callback,
        )

        # ── STEP 4: Embed Repository into ChromaDB ────────────────────────────
        _log_event(
            logs, AgentStage.EMBEDDING, "Chunking and embedding repository...", event_callback=event_callback
        )

        total_chunks = embed_repository(repo_path, repo_name_safe)

        _log_event(
            logs, AgentStage.EMBEDDING,
            f"Embedded {total_chunks} chunks into ChromaDB.",
            data={"total_chunks": total_chunks},
            event_callback=event_callback,
        )

        # ── STEP 5: Retrieve Relevant Code ────────────────────────────────────
        _log_event(
            logs, AgentStage.RETRIEVING, "Searching for relevant code chunks...", event_callback=event_callback
        )

        # Build the search query from issue title + body
        search_query = f"{issue.title}\n\n{issue.body}"
        relevant_chunks = retrieve_relevant_chunks(search_query, repo_name_safe)

        if not relevant_chunks:
            _log_event(
                logs, AgentStage.RETRIEVING,
                "No relevant chunks found. The repo may have no supported source files.",
                level=LogLevel.WARNING,
                event_callback=event_callback,
            )
            context = "No relevant code found in the repository."
        else:
            context = format_chunks_for_prompt(relevant_chunks)
            _log_event(
                logs, AgentStage.RETRIEVING,
                f"Retrieved {len(relevant_chunks)} relevant chunks.",
                data={
                    "chunk_count": len(relevant_chunks),
                    "top_file": relevant_chunks[0].file_path,
                    "top_score": relevant_chunks[0].score,
                },
                event_callback=event_callback,
            )

        # ── STEP 6: Generate Fix with LLM ─────────────────────────────────────
        _log_event(logs, AgentStage.GENERATING, "Sending issue and code to LLM...", event_callback=event_callback)

        patch = generate_patch(issue, context)
        run.patch = patch  # Store on RunLog

        _log_event(
            logs, AgentStage.GENERATING,
            f"LLM generated fix for file: {patch.file_path}",
            data={"file_path": patch.file_path},
            event_callback=event_callback,
        )

        # ── STEP 7: Apply the Patch ────────────────────────────────────────────
        _log_event(
            logs, AgentStage.APPLYING_PATCH, f"Applying patch to {patch.file_path}...", event_callback=event_callback
        )

        applied_patch = apply_patch(patch, repo_path)
        run.patch = applied_patch  # Update with diff included

        _log_event(
            logs, AgentStage.APPLYING_PATCH,
            "Patch applied successfully.",
            data={"diff_preview": applied_patch.diff[:200] if applied_patch.diff else ""},
            event_callback=event_callback,
        )

        # ── STEP 8: Open Pull Request ──────────────────────────────────────────
        _log_event(
            logs,
            AgentStage.CREATING_PR,
            "Creating branch and opening Pull Request...",
            event_callback=event_callback,
        )

        pr_url = create_fix_pr(repo_path, issue, applied_patch)
        run.pr_url = pr_url  # Store on RunLog

        _log_event(
            logs, AgentStage.CREATING_PR,
            f"Pull Request opened: {pr_url}",
            data={"pr_url": pr_url},
            event_callback=event_callback,
        )

        # ── SUCCESS ────────────────────────────────────────────────────────────
        run.status = RunStatus.SUCCESS
        run.finished_at = datetime.now(timezone.utc)

        logger.info("=" * 60)
        logger.info("✅ Pipeline SUCCEEDED — PR: %s", pr_url)
        logger.info("=" * 60)

    except Exception as e:
        # ── FAILURE — catch ALL errors, log them, return partial RunLog ────────
        error_msg = str(e)
        tb = traceback.format_exc()

        _log_event(
            logs, AgentStage.FAILED,
            f"Pipeline failed: {error_msg}",
            level=LogLevel.ERROR,
            data={"traceback": tb},
            event_callback=event_callback,
        )

        run.status = RunStatus.FAILED
        run.error = error_msg
        run.finished_at = datetime.now(timezone.utc)

        logger.error("=" * 60)
        logger.error("❌ Pipeline FAILED: %s", error_msg)
        logger.error("=" * 60)

    return run
