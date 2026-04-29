from app.agent.patch_applier import apply_patch
from app.core.constants import MAX_RETRIES
from app.llm.client import generate_patch
from app.models.issue import IssueModel
from app.models.patch import PatchModel


def generate_and_apply_with_retries(
    issue: IssueModel,
    base_context: str,
    repo_path: str,
    max_retries: int = MAX_RETRIES,
) -> tuple[PatchModel, int]:
    """
    Try patch generation + patch application multiple times.

    Returns:
        (applied_patch, attempts_used)

    Raises:
        RuntimeError if all attempts fail.
    """
    feedback = ""
    last_error = "Unknown error"

    for attempt in range(1, max_retries + 1):
        context = base_context
        if feedback:
            context = (
                f"{base_context}\n\n"
                "Previous attempt failed. Please revise your fix using this error:\n"
                f"{feedback}\n"
                "Return a corrected single-file full patch."
            )

        try:
            patch = generate_patch(issue, context)
            applied_patch = apply_patch(patch, repo_path)
            return applied_patch, attempt
        except Exception as exc:
            last_error = str(exc)
            feedback = last_error

    raise RuntimeError(
        f"Patch generation/application failed after {max_retries} attempts. "
        f"Last error: {last_error}"
    )
