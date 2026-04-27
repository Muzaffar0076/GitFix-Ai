"""
app/models/event_log.py
────────────────────────
Represents a single log event emitted during an agent run.

Every time the agent transitions between stages (Cloning → Analyzing →
Fixing → Testing), it emits an EventLog. These events are:
  1. Saved to memory/DB per run
  2. Broadcast over WebSocket to the React dashboard in real time
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class LogLevel(str, Enum):
    """
    WHY AN ENUM?
    Using an enum instead of raw strings ("INFO", "ERROR") means:
    - Typos are caught at write time, not runtime
    - Your IDE gives you autocomplete
    - The JSON sent over WebSocket is always valid
    """
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"


class AgentStage(str, Enum):
    """
    All stages of the GitFix AI pipeline.
    The frontend uses these to drive the progress stepper UI.
    """
    PARSING        = "PARSING"        # Parsing the GitHub issue URL
    FETCHING_ISSUE = "FETCHING_ISSUE" # Fetching issue from GitHub API
    CLONING        = "CLONING"        # Cloning the repository
    EMBEDDING      = "EMBEDDING"      # Chunking + embedding into ChromaDB
    RETRIEVING     = "RETRIEVING"     # Semantic search for relevant code
    GENERATING     = "GENERATING"     # LLM generating the patch
    APPLYING_PATCH = "APPLYING_PATCH" # Writing fix to disk
    CREATING_PR    = "CREATING_PR"    # Opening Pull Request on GitHub
    FAILED         = "FAILED"         # Pipeline failed at some stage


class EventLog(BaseModel):
    """
    One log line sent to the frontend via WebSocket.
    """

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    # auto-set to current UTC time when object is created

    level: LogLevel = LogLevel.INFO

    stage: AgentStage               # Which pipeline stage produced this event

    message: str                    # Human-readable log message

    data: dict = Field(default_factory=dict)
    # Optional extra data (pr_url, chunk_count, etc.)
