from secrets import token_urlsafe
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import select
import bcrypt

from app.core.config import get_settings
from app.core.database import get_session
from app.models.db_models import User


router = APIRouter(prefix="/api/auth", tags=["auth"])
security = HTTPBearer(auto_error=False)

# We keep ACTIVE_TOKENS in memory for now (sessions), but link them to DB users
ACTIVE_TOKENS: dict[str, str] = {}


class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=40)
    password: str = Field(min_length=6, max_length=128)


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=40)
    password: str = Field(min_length=6, max_length=128)


class LoginResponse(BaseModel):
    token: str
    username: str


def hash_password(password: str) -> str:
    """
    Hashes a plain-text password using bcrypt.
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a plain-text password against a hashed bcrypt password.
    """
    return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))


def _seed_default_user(session: Session) -> None:
    settings = get_settings()
    # Check if admin already exists
    statement = select(User).where(User.username == settings.AUTH_USERNAME)
    existing_user = session.execute(statement).scalar_one_or_none()
    
    if not existing_user:
        admin = User(
            username=settings.AUTH_USERNAME,
            hashed_password=hash_password(settings.AUTH_PASSWORD)
        )
        session.add(admin)
        session.commit()


def _create_session(username: str) -> LoginResponse:
    token = token_urlsafe(32)
    ACTIVE_TOKENS[token] = username
    return LoginResponse(token=token, username=username)


def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing bearer token",
        )

    token = credentials.credentials
    if token not in ACTIVE_TOKENS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    return token


def is_valid_token(token: str | None) -> bool:
    """
    Checks if a token exists in ACTIVE_TOKENS.
    Used by WebSockets where we can't easily use Depends(verify_token).
    """
    return token in ACTIVE_TOKENS


def get_current_user(
    token: str = Depends(verify_token),
    session: Session = Depends(get_session)
) -> User:
    username = ACTIVE_TOKENS[token]
    statement = select(User).where(User.username == username)
    user = session.execute(statement).scalar_one_or_none()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user


@router.post("/register", response_model=LoginResponse, status_code=status.HTTP_201_CREATED)
def register(
    request: RegisterRequest, 
    session: Session = Depends(get_session)
) -> LoginResponse:
    username = request.username.strip()

    # Check if user already exists
    statement = select(User).where(User.username == username)
    if session.execute(statement).scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already registered",
        )

    new_user = User(
        username=username,
        hashed_password=hash_password(request.password)
    )
    session.add(new_user)
    session.commit()
    
    return _create_session(username)


@router.post("/login", response_model=LoginResponse)
def login(
    request: LoginRequest, 
    session: Session = Depends(get_session)
) -> LoginResponse:
    _seed_default_user(session)
    username = request.username.strip()
    
    statement = select(User).where(User.username == username)
    user = session.execute(statement).scalar_one_or_none()

    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    return _create_session(username)


@router.get("/me")
def current_user_info(user: User = Depends(get_current_user)) -> dict[str, str]:
    return {"username": user.username}


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(token: str = Depends(verify_token)) -> None:
    ACTIVE_TOKENS.pop(token, None)
