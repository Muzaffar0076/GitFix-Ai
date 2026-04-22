"""
app/github/pr_creator.py
─────────────────────────
Handles everything that happens AFTER the patch is verified:
  1. Create a new git branch  → "gitfix/issue-42"
  2. Write the patched file to disk
  3. Stage + commit the changes
  4. Push the branch to GitHub
  5. Open a Pull Request linking back to the original issue

CONCEPT — Why a new branch instead of committing directly to main?
  In real software development, you NEVER push directly to main/master.
  Instead:
    main ──── gitfix/issue-42 ──── (patch committed here)
                                         ↓
                              Pull Request opened
                                         ↓
                         Human reviews → merges to main

  This protects the main branch from broken code and gives humans
  a chance to review the AI's fix before it goes live.

FLOW:
  PatchModel + IssueModel
      ↓
  create_branch()
      ↓
  commit_patch()
      ↓
  push_branch()
      ↓
  open_pull_request()
      ↓
  Returns PR URL (string)
"""

import os

import git
from github import Github, GithubException

from app.core.config import get_settings
from app.core.constants import BRANCH_PREFIX
from app.core.logger import logger
from app.models.issue import IssueModel
from app.models.patch import PatchModel


# ── 1. Create a New Git Branch ────────────────────────────────────────────────

def create_branch(repo_path: str, issue_number: int) -> str:
    """
    Creates a new local git branch for the fix.

    CONCEPT — Git Branches:
    A branch is an independent line of development. Think of it like
    a parallel universe copy of your code where you can make changes
    without affecting the original.

    Branch naming convention: "gitfix/issue-42"
    - "gitfix/" prefix → makes it clear this branch was created by our AI
    - "issue-42"       → links the branch to the GitHub issue number

    Args:
        repo_path: Absolute path to the cloned repository.
        issue_number: The GitHub issue number being fixed.

    Returns:
        The branch name string, e.g. "gitfix/issue-42"
    """
    branch_name = f"{BRANCH_PREFIX}{issue_number}"
    # BRANCH_PREFIX = "gitfix/issue-" from constants.py
    # Result: "gitfix/issue-42"

    try:
        repo = git.Repo(repo_path)

        # Check if branch already exists (in case of a retry)
        existing_branches = [b.name for b in repo.branches]

        if branch_name in existing_branches:
            # Branch exists → just check it out (switch to it)
            repo.git.checkout(branch_name)
            logger.info("Switched to existing branch: %s", branch_name)
        else:
            # Branch doesn't exist → create it and switch to it
            # git.checkout("-b", branch_name) ≡ `git checkout -b gitfix/issue-42`
            repo.git.checkout("-b", branch_name)
            logger.info("Created new branch: %s", branch_name)

        return branch_name

    except git.GitCommandError as e:
        logger.error("Failed to create branch %s: %s", branch_name, str(e))
        raise ValueError(f"Could not create branch '{branch_name}': {str(e)}") from e


# ── 2. Write Patch + Stage + Commit ──────────────────────────────────────────

def commit_patch(repo_path: str, patch: PatchModel, issue_number: int) -> None:
    """
    Writes the patched file to disk, stages it, and creates a git commit.

    CONCEPT — Git Staging Area (Index):
    Git has 3 zones:
      Working Directory → where your files live on disk
      Staging Area      → files you've marked "ready to commit" with `git add`
      Commit History    → permanent snapshots saved with `git commit`

    This function:
      1. Writes the patched code to disk (Working Directory)
      2. `git add` → moves it to Staging Area
      3. `git commit` → saves it permanently to history

    Args:
        repo_path: Absolute path to the cloned repository.
        patch: The PatchModel containing file_path and patched_code.
        issue_number: Used to write a descriptive commit message.
    """
    try:
        repo = git.Repo(repo_path)

        # ── Step 1: Write patched code to disk ────────────────────────────────
        # patch.file_path is relative (e.g. "src/auth/login.py")
        # We need the full absolute path to write to disk
        full_file_path = os.path.join(repo_path, patch.file_path)

        # Make sure the directory exists (in case of new files)
        os.makedirs(os.path.dirname(full_file_path), exist_ok=True)

        with open(full_file_path, "w", encoding="utf-8") as f:
            f.write(patch.patched_code)

        logger.info("Wrote patched file to: %s", patch.file_path)

        # ── Step 2: Stage the file (git add) ──────────────────────────────────
        # repo.index.add() ≡ `git add <file_path>`
        repo.index.add([patch.file_path])
        logger.info("Staged: %s", patch.file_path)

        # ── Step 3: Commit with a descriptive message ──────────────────────────
        commit_message = (
            f"fix: resolve issue #{issue_number} via GitFix AI\n\n"
            f"Modified file: {patch.file_path}\n"
            f"Automated fix generated by GitFix AI agent."
        )

        # repo.index.commit() ≡ `git commit -m "..."`
        repo.index.commit(
            commit_message,
            author=git.Actor("GitFix AI", "gitfix-ai@noreply.github.com"),
            committer=git.Actor("GitFix AI", "gitfix-ai@noreply.github.com"),
        )
        # git.Actor sets the name/email shown in the commit history

        logger.info("Committed fix for issue #%d", issue_number)

    except git.GitCommandError as e:
        logger.error("Failed to commit patch: %s", str(e))
        raise ValueError(f"Git commit failed: {str(e)}") from e


# ── 3. Push Branch to GitHub ──────────────────────────────────────────────────

def push_branch(repo_path: str, branch_name: str) -> None:
    """
    Pushes the local branch to the remote (GitHub).

    CONCEPT — Remote vs Local:
    Your cloned repo exists in TWO places:
      Local  → your machine (at repo_path)
      Remote → GitHub (called "origin")

    `git push origin <branch>` uploads your local branch to GitHub
    so you can open a Pull Request from it.

    Args:
        repo_path: Absolute path to the cloned repository.
        branch_name: Name of the branch to push (e.g. "gitfix/issue-42")
    """
    try:
        repo = git.Repo(repo_path)

        # repo.remotes.origin is the "origin" remote (GitHub)
        # push() ≡ `git push origin <branch_name>`
        origin = repo.remotes.origin
        push_result = origin.push(refspec=f"{branch_name}:{branch_name}")

        # Check for push errors
        for result in push_result:
            if result.flags & result.ERROR:
                raise ValueError(f"Push failed: {result.summary}")

        logger.info("Pushed branch '%s' to GitHub.", branch_name)

    except git.GitCommandError as e:
        logger.error("Failed to push branch: %s", str(e))
        raise ValueError(f"Git push failed: {str(e)}") from e


# ── 4. Open a Pull Request ────────────────────────────────────────────────────

def open_pull_request(
    issue: IssueModel,
    branch_name: str,
    patch: PatchModel,
) -> str:
    """
    Opens a Pull Request on GitHub using the PyGithub API.

    CONCEPT — Pull Request:
    A PR is a formal request to merge code from one branch into another.
    It's where code review happens. By linking to the original issue
    (using "Closes #42"), GitHub will automatically close the issue
    when the PR is merged.

    Args:
        issue: The original IssueModel (contains repo name, issue number).
        branch_name: The branch containing the fix.
        patch: The PatchModel (used to describe what was changed in the PR body).

    Returns:
        The URL of the created Pull Request.
        e.g. "https://github.com/owner/repo/pull/5"
    """
    settings = get_settings()
    gh = Github(settings.GITHUB_PAT)

    try:
        repo = gh.get_repo(issue.repo_full_name)

        # Get the default branch (usually "main" or "master")
        default_branch = repo.default_branch
        # We merge our fix branch INTO the default branch

        # Build a descriptive PR title and body
        pr_title = f"fix: {issue.title} (via GitFix AI)"

        pr_body = f"""## 🤖 Automated Fix by GitFix AI

This Pull Request was automatically generated by **GitFix AI** to resolve the issue below.

### Original Issue
**#{issue.issue_number}:** {issue.title}

### What Was Changed
- **File modified:** `{patch.file_path}`
- **Fix summary:** The agent analyzed the issue description and relevant source code, then generated a minimal patch to resolve the reported bug.

### Diff Summary
```diff
{patch.diff if patch.diff else "Patch applied — diff not available"}
```

---
Closes #{issue.issue_number}

> ⚠️ Please review this automated fix carefully before merging.
"""

        # repo.create_pull() ≡ POST /repos/{owner}/{repo}/pulls
        pr = repo.create_pull(
            title=pr_title,
            body=pr_body,
            head=branch_name,      # branch WITH the fix
            base=default_branch,   # branch to merge INTO (main/master)
        )

        logger.info("Opened PR #%d: %s", pr.number, pr.html_url)
        return pr.html_url

    except GithubException as e:
        logger.error("Failed to create PR: %s - %s", e.status, e.data)
        raise ValueError(
            f"Could not open PR: GitHub API error {e.status}: "
            f"{e.data.get('message', 'Unknown error')}"
        ) from e


# ── 5. Master Function: Do Everything ────────────────────────────────────────

def create_fix_pr(
    repo_path: str,
    issue: IssueModel,
    patch: PatchModel,
) -> str:
    """
    Orchestrates the full PR creation pipeline:
      branch → commit → push → open PR

    This is the single function the orchestrator calls.
    It wraps all 4 steps above into one clean call.

    Args:
        repo_path: Local path to the cloned repo.
        issue: The IssueModel (metadata about the bug).
        patch: The PatchModel (the AI-generated fix).

    Returns:
        The Pull Request URL string.
    """
    logger.info("Starting PR creation for issue #%d...", issue.issue_number)

    # Step 1: Create branch
    branch_name = create_branch(repo_path, issue.issue_number)

    # Step 2: Write + commit the patch
    commit_patch(repo_path, patch, issue.issue_number)

    # Step 3: Push branch to GitHub
    push_branch(repo_path, branch_name)

    # Step 4: Open the Pull Request
    pr_url = open_pull_request(issue, branch_name, patch)

    logger.info("PR creation complete: %s", pr_url)
    return pr_url
