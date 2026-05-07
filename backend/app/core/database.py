from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.core.config import get_settings
from app.models.db_models import Base

settings = get_settings()

# Connect to the database. 
# check_same_thread=False is required ONLY for SQLite
connect_args = {"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}

engine = create_engine(
    settings.DATABASE_URL, 
    echo=False, 
    connect_args=connect_args
)

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """
    Creates all tables in the database based on our SQLAlchemy models.
    """
    Base.metadata.create_all(bind=engine)

def get_session():
    """
    FastAPI dependency that provides a database session for a single request.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
