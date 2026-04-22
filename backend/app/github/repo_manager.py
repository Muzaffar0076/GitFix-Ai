"""
app/github/repo_manager.py
───────────────────────────
Handles all GitHub repository operations:
  1. Parse a GitHub Issue URL → extract owner, repo, issue number
  2. Fetch issue details from GitHub API → returns IssueModel
  3. Clone or update the repository locally → returns local path

LIBRARIES USED:
  - re         : Python's built-in regex module (no install needed)
  - github     : PyGithub — wraps GitHub REST API in Python objects
  - git        : GitPython — lets you run git commands from Python

FLOW:
  URL → parse_issue_url() → fetch_issue_details() → clone_or_pull_repo()
"""

import os
import re

import git                         # GitPython
from github import Github          # PyGithub
from github import GithubException # For catching API errors

from app.core.config import get_settings
from app.core.logger import logger
from app.models.issue import IssueModel


# ── 1. Parse the GitHub Issue URL ─────────────────────────────────────────────

def parse_issue_url(url: str) -> tuple[str, str, int]:
    """
    Extracts owner, repo name, and issue number from a GitHub URL.

    CONCEPT — Regex (Regular Expressions):
    A regex is a pattern that matches text. Here we describe the shape of a
    GitHub URL and use Python to extract the parts we care about.

    Example URL:
      "https://github.com/owner/my-repo/issues/42"

    Pattern breakdown:
      https://github\\.com/   → literal "https://github.com/"
      ([\\w.-]+)              → capture group 1: owner (letters, digits, dots, dashes)
      /                       → literal slash
      ([\\w.-]+)              → capture group 2: repo name
      /issues/                → literal "/issues/"
      (\\d+)                  → capture group 3: issue number (digits only)

    Args:
        url: The GitHub issue URL pasted by the user.

    Returns:
        A tuple of (owner, repo_name, issue_number)

    Raises:
        ValueError: If the URL doesn't match the expected GitHub issue format.
    """
    # re.search() scans the string for the first location where the pattern matches
    pattern = r"https://github\.com/([\w.\-]+)/([\w.\-]+)/issues/(\d+)"
    match = re.search(pattern, url.strip())

    if not match:
        raise ValueError(
            f"Invalid GitHub issue URL: '{url}'\n"
            "Expected format: https://github.com/owner/repo/issues/123"
        )

    owner = match.group(1)        # First capture group → owner
    repo_name = match.group(2)    # Second capture group → repo name
    issue_number = int(match.group(3))  # Third capture group → issue number

    logger.info("Parsed URL → owner=%s, repo=%s, issue=#%d", owner, repo_name, issue_number)
    return owner, repo_name, issue_number


# ── 2. Fetch Issue Details from GitHub API ────────────────────────────────────

def fetch_issue_details(owner: str, repo_name: str, issue_number: int) -> IssueModel:
    """
    Fetches the title, body, and labels of a GitHub issue using PyGithub.

    CONCEPT — PyGithub:
    Instead of manually making HTTP requests to GitHub's REST API, PyGithub
    gives us a clean Python interface. We just call methods and get Python
    objects back.

    HOW AUTHENTICATION WORKS:
    GitHub's API allows ~60 requests/hour unauthenticated.
    With our PAT (Personal Access Token), we get 5,000 requests/hour.

    Args:
        owner: GitHub username or org name.
        repo_name: The repository name.
        issue_number: The issue number (integer).

    Returns:
        An IssueModel populated with data from the GitHub API.

    Raises:
        GithubException: If the repo/issue doesn't exist or PAT lacks access.
    """
    settings = get_settings()

    # Authenticate with our PAT — this is sent as a Bearer token in API headers
    gh = Github(settings.GITHUB_PAT)

    try:
        logger.info("Fetching issue #%d from %s/%s...", issue_number, owner, repo_name)

        # gh.get_repo() makes a GET request to:
        #   https://api.github.com/repos/{owner}/{repo_name}
        repo = gh.get_repo(f"{owner}/{repo_name}")

        # repo.get_issue() makes a GET request to:
        #   https://api.github.com/repos/{owner}/{repo_name}/issues/{issue_number}
        issue = repo.get_issue(number=issue_number)

        # Extract label names from the label objects
        labels = [label.name for label in issue.labels]

        logger.info("Fetched issue: '%s' | Labels: %s", issue.title, labels)

        # Build and return our typed IssueModel
        return IssueModel(
            url=f"https://github.com/{owner}/{repo_name}/issues/{issue_number}",
            repo_full_name=f"{owner}/{repo_name}",
            issue_number=issue_number,
            title=issue.title,
            body=issue.body or "",   # body can be None for empty issues
            labels=labels,
        )

    except GithubException as e:
        # GithubException has a .status and .data attribute
        # e.g. 404 = not found, 403 = no permission
        logger.error("GitHub API error: %s - %s", e.status, e.data)
        raise ValueError(
            f"Could not fetch issue #{issue_number} from {owner}/{repo_name}.\n"
            f"GitHub API error {e.status}: {e.data.get('message', 'Unknown error')}\n"
            "Make sure your GITHUB_PAT has 'repo' scope and can access this repository."
        ) from e


# ── 3. Clone or Pull the Repository ───────────────────────────────────────────

def clone_or_pull_repo(owner: str, repo_name: str) -> str:
    """
    Clones the repository if it doesn't exist locally.
    If it already exists, does a `git pull` to get the latest code.

    CONCEPT — GitPython:
    GitPython lets us run Git commands from Python code.
      git.Repo.clone_from(url, path) ≡ running `git clone <url> <path>` in terminal
      repo.remotes.origin.pull()     ≡ running `git pull` in terminal

    WHY CLONE LOCALLY?
    The agent needs to READ the source files to understand the codebase,
    and WRITE patched files before committing. We can't do this directly
    on GitHub — we need a local copy.

    Args:
        owner: GitHub username or org name.
        repo_name: The repository name.

    Returns:
        The absolute path to the cloned repository on disk.
        e.g. "./cloned_repos/owner_repo_name"
    """
    settings = get_settings()

    # Build a safe folder name: "owner/my-repo" → "owner_my-repo"
    safe_folder = f"{owner}_{repo_name}"
    local_path = os.path.join(settings.REPOS_CLONE_PATH, safe_folder)
    # e.g. "./cloned_repos/muzaffar_gitfix-ai"

    # The clone URL includes the PAT for authentication:
    # https://<PAT>@github.com/owner/repo.git
    # This way Git can push/pull without prompting for a password.
    clone_url = f"https://{settings.GITHUB_PAT}@github.com/{owner}/{repo_name}.git"

    if os.path.exists(local_path):
        # Repo already cloned — just pull latest changes
        logger.info("Repo already exists at %s. Pulling latest changes...", local_path)
        try:
            existing_repo = git.Repo(local_path)
            existing_repo.remotes.origin.pull()
            logger.info("Successfully pulled latest changes.")
        except git.GitCommandError as e:
            # If pull fails (e.g. merge conflict), log a warning but continue
            # The existing code is still usable
            logger.warning("Git pull failed (non-fatal): %s", str(e))
    else:
        # Repo not cloned yet — clone it fresh
        logger.info("Cloning %s/%s into %s ...", owner, repo_name, local_path)
        try:
            git.Repo.clone_from(clone_url, local_path)
            logger.info("Successfully cloned repository.")
        except git.GitCommandError as e:
            logger.error("Git clone failed: %s", str(e))
            raise ValueError(
                f"Failed to clone {owner}/{repo_name}.\n"
                "Check that your GITHUB_PAT has 'repo' scope."
            ) from e

    return os.path.abspath(local_path)
    # os.path.abspath() converts "./cloned_repos/..." to a full absolute path
    # e.g. "/Users/muzaffar/Desktop/gitfix-ai/backend/cloned_repos/owner_repo"
