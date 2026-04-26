"""
app/llm/client.py
──────────────────
Handles all communication with the Groq LLM API.

CONCEPT — How LLMs Work (simplified):
  You send a "conversation" (list of messages) to the LLM.
  The LLM reads it and generates a response.

  Two types of messages:
    "system"  → Instructions to the LLM: "You are a bug-fixing agent..."
    "user"    → The actual input: "Here is the issue and the code..."

  The LLM reads both and returns a "assistant" message with the fix.

CONCEPT — Why Groq?
  Groq provides free, extremely fast inference for open-source LLMs.
  We use llama-3.3-70b-versatile — a 70 billion parameter model.
  It's very capable at understanding and generating code.

CONCEPT — Temperature:
  Controls how "creative" or "deterministic" the LLM is.
  0.0 = always picks the most likely next word (deterministic, focused)
  1.0 = more random/creative
  For bug fixing we use 0.2 — we want precision, not creativity.

FLOW:
  IssueModel + relevant code context (from retriever.py)
      ↓
  build_prompt()      → constructs system + user messages
      ↓
  call_llm()          → sends to Groq API, gets raw response
      ↓
  parse_llm_response()→ extracts file_path + patched_code from response
      ↓
  Returns PatchModel
"""

from groq import Groq

from app.core.config import get_settings
from app.core.constants import LLM_TEMPERATURE, MAX_LLM_TOKENS
from app.core.logger import logger
from app.models.issue import IssueModel
from app.models.patch import PatchModel


# ── System Prompt ─────────────────────────────────────────────────────────────
# This is the "personality" and "instructions" we give the LLM.
# It's sent with EVERY request, before the user message.

SYSTEM_PROMPT = """You are GitFix AI — an expert software engineer specializing in debugging and fixing code.

Your job:
1. Read a GitHub issue description carefully
2. Study the relevant source code provided
3. Identify the root cause of the bug
4. Generate a minimal, correct fix

Output format (you MUST follow this exactly):
---FILE_PATH---
<relative path to the file you are fixing, e.g. src/auth/login.py>
---PATCHED_CODE---
<the complete fixed file content — not just the changed lines, the FULL file>
---END---

Rules:
- Fix ONLY what is necessary. Do not refactor unrelated code.
- Preserve all existing comments, imports, and formatting style.
- The patched code must be the complete file, not just a snippet.
- If the fix requires changes to multiple files, fix the most critical one only.
- Do not explain yourself — just output the fix in the exact format above.
"""


# ── 1. Build the Prompt ───────────────────────────────────────────────────────

def build_prompt(issue: IssueModel, context: str) -> str:
    """
    Constructs the user message sent to the LLM.

    CONCEPT — Prompt Engineering:
    The way you phrase your prompt directly affects the LLM's output quality.
    We give the LLM:
      - The issue title and body (what's broken)
      - The relevant code context (where to look)
      - Clear instructions on output format

    Args:
        issue:   The IssueModel with title, body, labels.
        context: Formatted code chunks from retriever.py's format_chunks_for_prompt()

    Returns:
        A formatted string to send as the "user" message.
    """
    prompt = f"""## GitHub Issue to Fix

**Repository:** {issue.repo_full_name}
**Issue #{issue.issue_number}:** {issue.title}
**Labels:** {', '.join(issue.labels) if issue.labels else 'None'}

### Issue Description
{issue.body if issue.body else 'No description provided.'}

---

## Relevant Source Code

The following code files are most likely related to this issue:

{context}

---

## Your Task

Analyze the issue and the code above. Identify the bug and generate the fix.
Follow the output format exactly as specified in your instructions.
"""
    return prompt


# ── 2. Call the Groq API ──────────────────────────────────────────────────────

def call_llm(prompt: str) -> str:
    """
    Sends the prompt to Groq and returns the raw LLM response text.

    CONCEPT — Groq SDK:
    The Groq SDK works very similarly to OpenAI's SDK.
    You create a client with your API key, then call:
      client.chat.completions.create(
          model="llama-3.3-70b-versatile",
          messages=[...],
          ...
      )

    The response has this structure:
      response.choices[0].message.content  → the LLM's text response

    CONCEPT — Messages List:
    LLMs expect a list of message objects, each with a "role" and "content":
      [
        {"role": "system",  "content": "You are a bug-fixing agent..."},
        {"role": "user",    "content": "Here is the issue and code..."},
      ]
    The LLM reads this as a conversation and generates the next message.

    Args:
        prompt: The user message (built by build_prompt()).

    Returns:
        Raw text response from the LLM.
    """
    settings = get_settings()

    # Create Groq client — authenticated with our API key
    client = Groq(api_key=settings.GROQ_API_KEY)

    logger.info(
        "Calling Groq API (model=%s, temp=%.1f)...",
        settings.LLM_MODEL,
        LLM_TEMPERATURE,
    )

    try:
        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=MAX_LLM_TOKENS,
        )

        raw_text = response.choices[0].message.content
        # response.choices  → list of possible completions (we always use [0])
        # .message.content  → the actual text the LLM generated

        logger.info(
            "LLM responded (%d chars, %d tokens used).",
            len(raw_text),
            response.usage.total_tokens,
        )

        return raw_text

    except Exception as e:
        logger.error("Groq API call failed: %s", str(e))
        raise ValueError(f"LLM call failed: {str(e)}") from e


# ── 3. Parse the LLM Response ─────────────────────────────────────────────────

def parse_llm_response(raw_response: str, issue: IssueModel) -> PatchModel:
    """
    Extracts the file path and patched code from the LLM's raw text output.

    CONCEPT — Why Parse?
    We told the LLM to use a specific format:
        ---FILE_PATH---
        src/auth/login.py
        ---PATCHED_CODE---
        def login(...):
            ...
        ---END---

    This function finds those markers and extracts the content between them.
    If the format is wrong, we raise a clear error.

    Args:
        raw_response: The raw text string from call_llm().
        issue:        The original IssueModel (for building the PatchModel).

    Returns:
        A PatchModel with file_path and patched_code populated.
    """
    try:
        # Split on our custom markers
        parts = raw_response.split("---FILE_PATH---")
        if len(parts) < 2:
            raise ValueError("LLM response missing ---FILE_PATH--- marker")

        remainder = parts[1]  # Everything after ---FILE_PATH---

        path_and_rest = remainder.split("---PATCHED_CODE---")
        if len(path_and_rest) < 2:
            raise ValueError("LLM response missing ---PATCHED_CODE--- marker")

        file_path = path_and_rest[0].strip()
        # .strip() removes leading/trailing whitespace and newlines

        code_and_end = path_and_rest[1].split("---END---")
        patched_code = code_and_end[0].strip()

        if not file_path:
            raise ValueError("LLM returned empty file path")
        if not patched_code:
            raise ValueError("LLM returned empty patched code")

        logger.info("Parsed LLM response → file: %s (%d chars)", file_path, len(patched_code))

        return PatchModel(
            file_path=file_path,
            patched_code=patched_code,
            diff="",  # Will be generated by patch_applier.py after writing
        )

    except ValueError:
        raise  # re-raise our specific errors as-is
    except Exception as e:
        logger.error("Failed to parse LLM response: %s\nRaw:\n%s", str(e), raw_response[:500])
        raise ValueError(
            f"Could not parse LLM output. Raw response preview:\n{raw_response[:300]}"
        ) from e


# ── 4. Master Function: Issue + Context → PatchModel ─────────────────────────

def generate_patch(issue: IssueModel, context: str) -> PatchModel:
    """
    Full pipeline: build prompt → call LLM → parse response → return PatchModel.

    This is the single function the orchestrator calls.

    Args:
        issue:   The IssueModel (title, body, repo info).
        context: Formatted relevant code from retriever.format_chunks_for_prompt().

    Returns:
        A PatchModel with the AI-generated fix.
    """
    logger.info("Generating patch for issue #%d: %s", issue.issue_number, issue.title)

    # Step 1: Build the user message
    prompt = build_prompt(issue, context)
    logger.info("Prompt built (%d chars)", len(prompt))

    # Step 2: Call Groq
    raw_response = call_llm(prompt)

    # Step 3: Parse the response
    patch = parse_llm_response(raw_response, issue)

    logger.info("Patch generated for file: %s", patch.file_path)
    return patch
