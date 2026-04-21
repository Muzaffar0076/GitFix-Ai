"""
app/models/issue.py
────────────────────
Represents a GitHub Issue fetched from the API.

This is the starting point of our pipeline — everything the agent
needs to know about the bug comes from this model.
"""

from pydantic import BaseModel, HttpUrl


class IssueModel(BaseModel):
    """
    Holds all metadata about a GitHub Issue.

    WHY PYDANTIC?
    If any field is missing or the wrong type, Pydantic raises a
    ValidationError immediately — so bugs are caught early, not deep
    inside the agent pipeline.
    """

    url: str                    # The original URL the user pasted, e.g.:
                                # "https://github.com/owner/repo/issues/42"

    repo_full_name: str         # Extracted from the URL: "owner/repo"

    issue_number: int           # The issue number: 42

    title: str                  # Issue title: "Fix NullPointerException in auth"

    body: str                   # Full issue description (markdown text)
                                # This is what we send to Claude as the "bug report"

    labels: list[str] = []     # e.g. ["bug", "good first issue"]
                                # Helps the LLM understand severity/category
