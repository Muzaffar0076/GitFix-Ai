import asyncio
from uuid import uuid4

from pydantic import BaseModel, HttpUrl
from fastapi import APIRouter, HTTPException, status

from app.agent.orchestrator import run_fix_pipeline
from app.api.websockets.log_stream import log_stream_manager
from app.core.logger import logger
from app.models.event_log import EventLog
from app.models.run_log import RunStatus
from app.models.run_log import RunLog

router = APIRouter(prefix="/api", tags=["dashboard"])
RUN_STORE: dict[str, RunLog] = {}


class FixRequest(BaseModel):
    issue_url: HttpUrl


@router.post("/fix", response_model=RunLog, status_code=status.HTTP_200_OK)
async def start_fix(request: FixRequest) -> RunLog:
    """
    Step-1 API entry point: accepts a GitHub issue URL and runs the pipeline.
    """
    try:
        run_id = str(uuid4())
        run = RunLog(
            run_id=run_id,
            issue_url=str(request.issue_url),
            status=RunStatus.RUNNING,
        )
        RUN_STORE[run_id] = run
        loop = asyncio.get_running_loop()

        def on_event(event: EventLog) -> None:
            run.events.append(event)
            log_stream_manager.broadcast_from_thread(loop, run_id, event)

        run = await asyncio.to_thread(
            run_fix_pipeline,
            str(request.issue_url),
            run_id,
            on_event,
        )
        RUN_STORE[run.run_id] = run
        await log_stream_manager.broadcast_message(
            run.run_id,
            {"type": "run_complete", "run_id": run.run_id, "status": run.status},
        )
        return run
    except Exception as exc:
        logger.exception("Failed to run fix pipeline for URL: %s", request.issue_url)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run fix pipeline: {exc}",
        ) from exc


@router.get("/runs/{run_id}", response_model=RunLog, status_code=status.HTTP_200_OK)
def get_run_status(run_id: str) -> RunLog:
    """
    Step-2 API endpoint: fetch an existing run log by run_id.
    """
    run = RUN_STORE.get(run_id)
    if not run:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Run '{run_id}' not found",
        )
    return run
