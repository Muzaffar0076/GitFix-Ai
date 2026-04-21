"""
app/core/logger.py
───────────────────
Shared logger instance for the entire application.

USAGE (in any other file):
  from app.core.logger import logger
  logger.info("Starting agent...")
  logger.error("Docker failed: %s", error_message)

LOG LEVELS (low → high severity):
  DEBUG    → Fine-grained details, only for development
  INFO     → Normal operations ("Cloning repo...", "PR created")
  WARNING  → Something unexpected but not fatal
  ERROR    → A failure that needs attention
  CRITICAL → App cannot continue
"""

import logging
import sys

from app.core.config import get_settings


def _build_logger() -> logging.Logger:
    """
    Creates and configures the application logger.
    Called once at module load time.
    """
    settings = get_settings()

    # 1. Create a named logger — using the app name makes it easy to identify
    #    in logs when multiple libraries are also logging.
    log = logging.getLogger("gitfix_ai")

    # 2. Set the log level from the .env variable (e.g., "INFO" → logging.INFO)
    #    getattr turns the string "INFO" into the integer constant logging.INFO
    log.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    # 3. Create a handler — this controls WHERE logs are sent.
    #    StreamHandler sends them to stdout (your terminal).
    handler = logging.StreamHandler(sys.stdout)

    # 4. Create a formatter — this controls HOW each log line looks.
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(filename)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        # Example output:
        # 2026-04-22 01:00:00 | INFO     | orchestrator.py | Cloning repo...
    )

    handler.setFormatter(formatter)

    # 5. Attach the handler to the logger.
    #    Guard against duplicate handlers if this module is reloaded.
    if not log.handlers:
        log.addHandler(handler)

    return log


# This is the single logger instance imported by all other modules.
logger = _build_logger()
