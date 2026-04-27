"""
app/agent/patch_applier.py
───────────────────────────
Applies the LLM-generated patch to the cloned repository.

WHAT THIS FILE DOES:
  1. Takes the PatchModel (from llm/client.py) — has file_path + patched_code
  2. Validates the target file exists in the repo
  3. Reads the original file content (for generating the diff)
  4. Writes the patched code to disk
  5. Generates a unified diff (shows exactly what changed)
  6. Returns the updated PatchModel with the diff included

CONCEPT — What is a Diff?
  A diff shows the exact lines that changed between two versions of a file.
  This is what you see in GitHub Pull Requests — the green (+) and red (-) lines.

  Example diff:
    --- a/src/auth/login.py
    +++ b/src/auth/login.py
    @@ -91,7 +91,7 @@
         hashed = hashlib.sha256(password.encode()).hexdigest()
    -    return user if user.password == hashed else None
    +    return user if secrets.compare_digest(user.password, hashed) else None

  Red line (-) = what was there before
  Green line (+) = what the LLM replaced it with

CONCEPT — Why Generate a Diff?
  1. We include it in the Pull Request description so humans can see what changed
  2. It's useful for debugging if the LLM makes a bad fix
  3. It proves the fix actually changed something

FLOW:
  PatchModel (file_path + patched_code) + repo_path
      ↓
  validate_patch()    → check the file exists, code is non-empty
      ↓
  read original file  → needed for diff generation
      ↓
  write patched file  → overwrite with LLM's fix
      ↓
  generate_diff()     → compute what changed
      ↓
  return updated PatchModel (now has diff filled in)
"""

import difflib
import os

from app.core.logger import logger
from app.models.patch import PatchModel


# ── 1. Validate the Patch ─────────────────────────────────────────────────────

def validate_patch(patch: PatchModel, repo_path: str) -> None:
    """
    Checks that the patch is safe to apply before touching any files.

    CONCEPT — Defensive Programming:
    Always validate inputs before acting on them. This prevents:
      - Writing to files outside the repo (path traversal attack)
      - Overwriting the file with empty content (LLM hallucination)
      - Trying to patch a file that doesn't exist

    Args:
        patch:     The PatchModel with file_path and patched_code.
        repo_path: Absolute path to the cloned repository root.

    Raises:
        ValueError: If the patch is invalid or unsafe.
    """
    # ── Check 1: file_path must not be empty
    if not patch.file_path or not patch.file_path.strip():
        raise ValueError("LLM returned an empty file path — cannot apply patch.")

    # ── Check 2: patched_code must not be empty
    if not patch.patched_code or not patch.patched_code.strip():
        raise ValueError("LLM returned empty patched code — refusing to overwrite file.")

    # ── Check 3: Build the full path and check it's inside the repo
    full_path = os.path.realpath(os.path.join(repo_path, patch.file_path))
    repo_real = os.path.realpath(repo_path)

    if not full_path.startswith(repo_real):
        # Path traversal attack: LLM tried to write to "../../etc/passwd" or similar
        raise ValueError(
            f"Security: file_path '{patch.file_path}' resolves outside the repo. "
            "Refusing to apply patch."
        )

    # ── Check 4: The file must already exist in the repo
    # We don't create new files — we only fix existing ones
    if not os.path.exists(full_path):
        raise ValueError(
            f"File '{patch.file_path}' does not exist in the repository. "
            "The LLM may have hallucinated a file path."
        )

    logger.info("Patch validated: target file '%s' exists and path is safe.", patch.file_path)


# ── 2. Generate a Unified Diff ────────────────────────────────────────────────

def generate_diff(original: str, patched: str, file_path: str) -> str:
    """
    Generates a unified diff showing what changed between original and patched code.

    CONCEPT — difflib:
    Python's built-in `difflib` module computes differences between sequences.
    `unified_diff()` produces the standard format used by Git and GitHub.

    Args:
        original:  The original file content (before the fix).
        patched:   The new file content (after the LLM fix).
        file_path: Used in the diff header (e.g. "src/auth/login.py").

    Returns:
        A string containing the unified diff, or a message if nothing changed.
    """
    original_lines = original.splitlines(keepends=True)
    patched_lines = patched.splitlines(keepends=True)
    # keepends=True → preserves the \n at end of each line (required by difflib)

    diff_lines = list(difflib.unified_diff(
        original_lines,
        patched_lines,
        fromfile=f"a/{file_path}",  # the "before" label
        tofile=f"b/{file_path}",    # the "after" label
        lineterm="",
    ))

    if not diff_lines:
        return "No changes detected — the LLM returned identical content."

    return "".join(diff_lines)


# ── 3. Apply the Patch ────────────────────────────────────────────────────────

def apply_patch(patch: PatchModel, repo_path: str) -> PatchModel:
    """
    Applies the LLM patch to the repository and returns updated PatchModel.

    This is the main function called by the orchestrator.

    Steps:
      1. Validate the patch (safety checks)
      2. Read the original file (for diff)
      3. Write the patched code to disk
      4. Generate the diff
      5. Return updated PatchModel with diff

    Args:
        patch:     PatchModel from llm/client.py (has file_path + patched_code).
        repo_path: Absolute path to the cloned repository.

    Returns:
        Updated PatchModel with the `diff` field populated.
    """
    # Step 1: Validate before touching anything
    validate_patch(patch, repo_path)

    full_path = os.path.join(repo_path, patch.file_path)

    # Step 2: Read original content for diff generation
    logger.info("Reading original file: %s", patch.file_path)
    with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
        original_content = f.read()

    # Step 3: Write the patched code to disk
    logger.info("Writing patched code to: %s", patch.file_path)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(patch.patched_code)

    # Step 4: Generate the diff
    diff = generate_diff(original_content, patch.patched_code, patch.file_path)

    lines_changed = diff.count("\n+") + diff.count("\n-")
    logger.info(
        "Patch applied successfully. ~%d lines changed in '%s'.",
        lines_changed,
        patch.file_path,
    )

    # Step 5: Return updated PatchModel with diff filled in
    # We create a new PatchModel so the original is not mutated
    return PatchModel(
        file_path=patch.file_path,
        original_code=original_content,
        patched_code=patch.patched_code,
        diff=diff,
    )
