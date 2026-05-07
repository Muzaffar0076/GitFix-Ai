from secrets import compare_digest, token_urlsafe

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from app.core.config import get_settings


router = APIRouter(prefix="/api/auth", tags=["auth"])
security = HTTPBearer(auto_error=False)
ACTIVE_TOKENS: set[str] = set()


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    username: str


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


@router.post("/login", response_model=LoginResponse)
def login(request: LoginRequest) -> LoginResponse:
    settings = get_settings()
    username_ok = compare_digest(request.username, settings.AUTH_USERNAME)
    password_ok = compare_digest(request.password, settings.AUTH_PASSWORD)

    if not username_ok or not password_ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )

    token = token_urlsafe(32)
    ACTIVE_TOKENS.add(token)
    return LoginResponse(token=token, username=request.username)


@router.get("/me")
def current_user(_: str = Depends(verify_token)) -> dict[str, str]:
    return {"username": get_settings().AUTH_USERNAME}


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(token: str = Depends(verify_token)) -> None:
    ACTIVE_TOKENS.discard(token)
