import os
import shutil
import subprocess
from typing import TypedDict

import docker

from app.core.logger import logger


class TestRunResult(TypedDict):
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    runner: str


def run_tests_in_docker(
    repo_path: str,
    command: str = "pytest -q",
    image: str = "python:3.11-slim",
    timeout_seconds: int = 300,
) -> TestRunResult:
    """
    Run tests inside a disposable Docker container.
    """
    client = docker.from_env()

    command_parts = command.split()
    workdir = "/workspace"

    container = client.containers.run(
        image=image,
        command=command_parts,
        working_dir=workdir,
        volumes={os.path.abspath(repo_path): {"bind": workdir, "mode": "rw"}},
        detach=True,
        stderr=True,
        stdout=True,
    )

    try:
        result = container.wait(timeout=timeout_seconds)
        exit_code = int(result.get("StatusCode", 1))
        logs = container.logs(stdout=True, stderr=True).decode("utf-8", errors="ignore")
        return {
            "success": exit_code == 0,
            "exit_code": exit_code,
            "stdout": logs,
            "stderr": "" if exit_code == 0 else logs,
            "runner": "docker",
        }
    finally:
        container.remove(force=True)


def run_tests_locally(
    repo_path: str,
    command: str = "pytest -q",
    timeout_seconds: int = 300,
) -> TestRunResult:
    """
    Fallback runner when Docker is unavailable in local dev.
    """
    command_parts = command.split()
    completed = subprocess.run(
        command_parts,
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return {
        "success": completed.returncode == 0,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "runner": "local",
    }


def run_tests_with_fallback(
    repo_path: str,
    command: str = "pytest -q",
    timeout_seconds: int = 300,
) -> TestRunResult:
    """
    Prefer Docker sandbox. Fallback to local execution when unavailable.
    """
    if shutil.which("docker") is None:
        logger.warning("Docker CLI not found. Falling back to local test execution.")
        return run_tests_locally(repo_path, command=command, timeout_seconds=timeout_seconds)

    try:
        return run_tests_in_docker(
            repo_path=repo_path,
            command=command,
            timeout_seconds=timeout_seconds,
        )
    except Exception as exc:
        logger.warning("Docker test execution failed (%s). Falling back to local runner.", exc)
        return run_tests_locally(repo_path, command=command, timeout_seconds=timeout_seconds)
