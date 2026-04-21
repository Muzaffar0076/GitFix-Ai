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
    The 6 stages of the GitFix AI pipeline.
    The frontend uses these to drive the progress stepper UI.
    """
    CLONING = "CLONING"         # Fetching issue + cloning repo
    ANALYZING = "ANALYZING"     # Running RAG pipeline
    FIXING = "FIXING"           # LLM generating the patch
    TESTING = "TESTING"         # Docker sandbox running tests
    RETRYING = "RETRYING"       # Self-healing: sending errors back to Claude
    RESOLVING = "RESOLVING"     # Creating branch, committing, opening PR


class EventLog(BaseModel):
    """
    One log line sent to the frontend via WebSocket.
    """

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    # auto-set to current UTC time when object is created

    level: LogLevel = LogLevel.INFO

    stage: AgentStage           # Which pipeline stage produced this event

    message: str                # Human-readable log message
                                # e.g. "Cloned repo owner/repo in 3.2s"
