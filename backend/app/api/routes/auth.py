from secrets import compare_digest, token_urlsafe

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.core.config import get_settings


router = APIRouter(prefix="/api/auth", tags=["auth"])
security = HTTPBearer(auto_error=False)
USERS: dict[str, str] = {}
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


def _seed_default_user() -> None:
    settings = get_settings()
    USERS.setdefault(settings.AUTH_USERNAME, settings.AUTH_PASSWORD)


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
    return bool(token and token in ACTIVE_TOKENS)


@router.post("/register", response_model=LoginResponse, status_code=status.HTTP_201_CREATED)
def register(request: RegisterRequest) -> LoginResponse:
    _seed_default_user()
    username = request.username.strip()

    if username in USERS:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already registered",
        )

    USERS[username] = request.password
    return _create_session(username)


@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest) -> LoginResponse:
    _seed_default_user()
    username = request.username.strip()
    stored_password = USERS.get(username)
    password_ok = stored_password is not None and compare_digest(
        request.password,
        stored_password,
    )

    if not password_ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    return _create_session(username)


@router.get("/me")
def current_user(token: str = Depends(verify_token)) -> dict[str, str]:
    return {"username": ACTIVE_TOKENS[token]}


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(token: str = Depends(verify_token)) -> None:
    ACTIVE_TOKENS.pop(token, None)
