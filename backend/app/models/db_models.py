from __future__ import annotations
from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    """
    Standard SQLAlchemy Declarative Base.
    """
    pass

class User(Base):
    """
    Represents a registered user who can trigger and view runs.
    """
    __tablename__ = "user"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship to RunLog: One user can have many runs
    runs: Mapped[List["RunLog"]] = relationship("RunLog", back_populates="user")

class RunLog(Base):
    """
    Stores the full record of one agent run in the database.
    """
    __tablename__ = "runlog"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36), unique=True, index=True, default=lambda: str(uuid4()))
    issue_url: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="PENDING")
    
    # Store complex metadata as JSON
    issue: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    patch: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    pr_url: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Relationship to User
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("user.id"), nullable=True)
    user: Mapped[Optional["User"]] = relationship("User", back_populates="runs")
    
    # Relationship to EventLog
    events: Mapped[List["EventLog"]] = relationship(
        "EventLog", 
        back_populates="run", 
        cascade="all, delete-orphan",
        order_by="EventLog.timestamp"
    )

class EventLog(Base):
    """
    Stores individual log events for a specific run.
    """
    __tablename__ = "eventlog"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    level: Mapped[str] = mapped_column(String(20), default="INFO")
    stage: Mapped[str] = mapped_column(String(40), default="PARSING")
    message: Mapped[str] = mapped_column(Text)
    data: Mapped[dict] = mapped_column(JSON, default=dict)
    
    # Relationship to RunLog
    run_id: Mapped[Optional[int]] = mapped_column(ForeignKey("runlog.id"), nullable=True)
    run: Mapped[Optional["RunLog"]] = relationship("RunLog", back_populates="events")
