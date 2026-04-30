from typing import TypedDict

from app.core.logger import logger
from app.sandbox.docker_runner import run_tests_with_fallback


class TestExecutionResult(TypedDict):
    passed: bool
    exit_code: int
    logs: str
    runner: str


def execute_test_suite(
    repo_path: str,
    test_command: str = "pytest -q",
    timeout_seconds: int = 300,
) -> TestExecutionResult:
    """
    Execute repository tests and return normalized output for the agent.
    """
    result = run_tests_with_fallback(
        repo_path=repo_path,
        command=test_command,
        timeout_seconds=timeout_seconds,
    )

    logs = result["stdout"]
    if result["stderr"]:
        logs = f"{logs}\n{result['stderr']}".strip()

    logger.info(
        "Test execution finished via %s (exit_code=%d).",
        result["runner"],
        result["exit_code"],
    )

    return {
        "passed": result["success"],
        "exit_code": result["exit_code"],
        "logs": logs,
        "runner": result["runner"],
    }
