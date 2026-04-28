from pydantic import BaseModel, HttpUrl
from fastapi import APIRouter, HTTPException, status

from app.agent.orchestrator import run_fix_pipeline
from app.core.logger import logger
from app.models.run_log import RunLog

router = APIRouter(prefix="/api", tags=["dashboard"])


class FixRequest(BaseModel):
    issue_url: HttpUrl


@router.post("/fix", response_model=RunLog, status_code=status.HTTP_200_OK)
def start_fix(request: FixRequest) -> RunLog:
    """
    Step-1 API entry point: accepts a GitHub issue URL and runs the pipeline.
    """
    try:
        return run_fix_pipeline(str(request.issue_url))
    except Exception as exc:
        logger.exception("Failed to run fix pipeline for URL: %s", request.issue_url)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run fix pipeline: {exc}",
        ) from exc
