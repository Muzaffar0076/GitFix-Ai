from typing import List, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, HttpUrl
from sqlalchemy.orm import Session
from sqlalchemy import select

from app.core.database import get_session
from app.api.routes.auth import get_current_user
from app.models.db_models import User, RunLog, EventLog
from app.models.run_log import RunStatus
from app.models.event_log import LogLevel, AgentStage

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

# Pydantic models for API responses (distinct from DB models)
class EventResponse(BaseModel):
    timestamp: datetime
    level: str
    stage: str
    message: str
    data: dict

class RunResponse(BaseModel):
    run_id: str
    issue_url: str
    status: str
    created_at: datetime
    finished_at: Optional[datetime] = None
    attempts: int = 0
    pr_url: Optional[str] = None
    error: Optional[str] = None

class CreateRunRequest(BaseModel):
    issue_url: str

def _run_pipeline_in_background(run_db_id: int, issue_url: str):
    """
    Background task to execute the autonomous pipeline and update DB.
    """
    from app.core.database import SessionLocal
    
    db = SessionLocal()
    try:
        # 1. Update status to RUNNING
        run = db.get(RunLog, run_db_id)
        if not run:
            return
        run.status = "RUNNING"
        db.commit()

        # 2. Define a callback that writes to the DB
        def db_event_callback(event_obj):
            event = EventLog(
                run_id=run_db_id,
                level=event_obj.level.value if hasattr(event_obj.level, "value") else str(event_obj.level),
                stage=event_obj.stage.value if hasattr(event_obj.stage, "value") else str(event_obj.stage),
                message=event_obj.message,
                data=event_obj.data or {}
            )
            db.add(event)
            db.commit()

        # 3. Run pipeline
        from app.agent.orchestrator import run_fix_pipeline
        result_run_log = run_fix_pipeline(
            issue_url=issue_url,
            run_id=run.run_id,
            event_callback=db_event_callback
        )

        # 4. Final update
        run = db.get(RunLog, run_db_id)
        if result_run_log.status == "SUCCESS":
            run.status = "SUCCESS"
            run.pr_url = result_run_log.pr_url
            run.patch = result_run_log.patch.model_dump() if result_run_log.patch else None
            run.issue = result_run_log.issue.model_dump() if result_run_log.issue else None
        else:
            run.status = "FAILED"
            run.error = result_run_log.error or "Pipeline failed to generate a valid fix."
        
        run.finished_at = datetime.now(timezone.utc)
        db.commit()

    except Exception as e:
        run = db.get(RunLog, run_db_id)
        if run:
            run.status = "ERROR"
            run.error = str(e)
            run.finished_at = datetime.now(timezone.utc)
            db.commit()
    finally:
        db.close()

@router.post("/runs", response_model=RunResponse)
def create_run(
    request: CreateRunRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    # Create the run record
    new_run = RunLog(
        issue_url=str(request.issue_url),
        user_id=current_user.id,
        status="PENDING"
    )
    db.add(new_run)
    db.commit()
    db.refresh(new_run)

    # Start the background task
    background_tasks.add_task(_run_pipeline_in_background, new_run.id, new_run.issue_url)

    return RunResponse(
        run_id=new_run.run_id,
        issue_url=new_run.issue_url,
        status=new_run.status,
        created_at=new_run.created_at
    )

@router.get("/runs", response_model=List[RunResponse])
def list_runs(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    statement = select(RunLog).where(RunLog.user_id == current_user.id).order_by(RunLog.created_at.desc())
    runs = db.execute(statement).scalars().all()
    
    return [
        RunResponse(
            run_id=r.run_id,
            issue_url=r.issue_url,
            status=r.status,
            created_at=r.created_at,
            finished_at=r.finished_at,
            attempts=r.attempts,
            pr_url=r.pr_url,
            error=r.error
        ) for r in runs
    ]

@router.get("/runs/{run_id}", response_model=RunResponse)
def get_run_details(
    run_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    statement = select(RunLog).where(RunLog.run_id == run_id, RunLog.user_id == current_user.id)
    run = db.execute(statement).scalar_one_or_none()
    
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    return RunResponse(
        run_id=run.run_id,
        issue_url=run.issue_url,
        status=run.status,
        created_at=run.created_at,
        finished_at=run.finished_at,
        attempts=run.attempts,
        pr_url=run.pr_url,
        error=run.error
    )

@router.get("/runs/{run_id}/events", response_model=List[EventResponse])
def get_run_events(
    run_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_session)
):
    # First verify the run belongs to the user
    run_stmt = select(RunLog).where(RunLog.run_id == run_id, RunLog.user_id == current_user.id)
    run = db.execute(run_stmt).scalar_one_or_none()
    
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    
    # Return events (already ordered by timestamp in the model relationship)
    return [
        EventResponse(
            timestamp=e.timestamp,
            level=e.level,
            stage=e.stage,
            message=e.message,
            data=e.data
        ) for e in run.events
    ]
