import os
import shutil
import subprocess
import sys
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
    Fallback runner when Docker is unavailable.
    Tries to find pytest in the current virtual environment.
    """
    # 1. Try to find the pytest executable in the same folder as the current python
    python_bin_dir = os.path.dirname(sys.executable)
    pytest_path = os.path.join(python_bin_dir, "pytest")

    # If 'pytest' is the command, replace it with the full path
    command_parts = command.split()
    if command_parts[0] == "pytest":
        if os.path.exists(pytest_path):
            command_parts[0] = pytest_path
        else:
            logger.warning("Could not find pytest at %s. Falling back to system 'pytest'.", pytest_path)

    # 2. Add repo_path and repo_path/src to PYTHONPATH
    env = os.environ.copy()
    src_path = os.path.join(repo_path, "src")
    extra_paths = [repo_path]
    if os.path.isdir(src_path):
        extra_paths.insert(0, src_path)  # Prioritize src/

    current_pythonpath = env.get("PYTHONPATH", "")
    new_pythonpath = os.pathsep.join(extra_paths)
    env["PYTHONPATH"] = f"{new_pythonpath}{os.pathsep}{current_pythonpath}" if current_pythonpath else new_pythonpath

    try:
        completed = subprocess.run(
            command_parts,
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,  # Use the modified environment
        )
        return {
            "success": completed.returncode == 0,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "runner": "local",
        }
    except FileNotFoundError as exc:
        return {
            "success": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": f"Local test runner failed: {str(exc)}. Is pytest installed in the venv?",
            "runner": "local",
        }
    except Exception as exc:
        return {
            "success": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": f"An unexpected error occurred: {str(exc)}",
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
