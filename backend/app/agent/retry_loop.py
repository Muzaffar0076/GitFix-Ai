from app.agent.patch_applier import apply_patch
from app.core.constants import MAX_RETRIES
from app.llm.client import generate_patch
from app.models.issue import IssueModel
from app.models.patch import PatchModel
from app.sandbox.test_executor import execute_test_suite


def generate_apply_test_with_retries(
    issue: IssueModel,
    base_context: str,
    repo_path: str,
    max_retries: int = MAX_RETRIES,
) -> tuple[PatchModel, int, dict]:
    """
    Try patch generation + patch application + test execution multiple times.

    Returns:
        (applied_patch, attempts_used, test_result)

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

            test_result = execute_test_suite(repo_path)
            if test_result["passed"]:
                return applied_patch, attempt, test_result

            last_error = (
                "Test suite failed after applying patch.\n"
                f"Runner: {test_result['runner']}\n"
                f"Exit code: {test_result['exit_code']}\n"
                f"Logs:\n{test_result['logs'][:3000]}"
            )
            feedback = last_error
        except Exception as exc:
            last_error = str(exc)
            feedback = last_error

    raise RuntimeError(
        f"Patch generation/apply/test failed after {max_retries} attempts. "
        f"Last error: {last_error}"
    )
