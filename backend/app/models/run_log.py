"""
app/models/run_log.py
──────────────────────
Represents the full record of one agent run.

When a user submits a GitHub Issue URL, a RunLog is created.
It tracks the entire lifecycle: from PENDING → RUNNING → SUCCESS/FAILED.
This is what the API returns when the frontend polls for status.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from app.models.event_log import EventLog
from app.models.issue import IssueModel
from app.models.patch import PatchModel


class RunStatus(str, Enum):
    """
    The lifecycle states of a single fix attempt.
    """
    PENDING = "PENDING"     # Created, agent not started yet
    RUNNING = "RUNNING"     # Agent is actively working
    SUCCESS = "SUCCESS"     # PR was created successfully
    FAILED = "FAILED"       # All retries exhausted, no fix found


class RunLog(BaseModel):
    """
    The top-level container for everything about one fix attempt.

    CONCEPT — UUID:
    We use a UUID (Universally Unique ID) as the run_id instead of
    sequential numbers (1, 2, 3...) because:
    - UUIDs are safe to expose in URLs (no enumeration attacks)
    - They're unique globally, so two runs never clash
    - Example: "550e8400-e29b-41d4-a716-446655440000"
    """

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    # uuid4() generates a random UUID every time a new RunLog is created

    issue_url: str          # The original URL the user submitted

    status: RunStatus = RunStatus.PENDING

    events: list[EventLog] = Field(default_factory=list)
    # All log events emitted during this run, in order

    issue: Optional[IssueModel] = None
    # Populated once issue metadata is fetched from GitHub.

    patch: Optional[PatchModel] = None
    # Populated after the LLM generates/applies a fix.

    pr_url: Optional[str] = None
    # Set when the agent successfully creates a Pull Request
    # e.g. "https://github.com/owner/repo/pull/5"

    error: Optional[str] = None
    # Human-readable error message if the run fails.

    attempts: int = 0
    # How many LLM+Docker retry cycles were attempted (max = MAX_RETRIES)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None
