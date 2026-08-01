from __future__ import annotations

from fastapi import HTTPException, Request

from .config import Settings


def check_client_auth(request: Request, settings: Settings) -> None:
    token = settings.bridge_token
    if not token:
        return

    auth = request.headers.get("authorization", "")
    x_api_key = request.headers.get("x-api-key", "")
    if auth.lower().startswith("bearer ") and auth.split(" ", 1)[1] == token:
        return
    if x_api_key == token:
        return
    raise HTTPException(status_code=401, detail={"error": "unauthorized"})


def assert_body_size(body: bytes, settings: Settings) -> None:
    if len(body) > settings.max_body_bytes:
        raise HTTPException(
            status_code=413,
            detail={
                "error": "request_body_too_large",
                "body_bytes": len(body),
                "max_body_bytes": settings.max_body_bytes,
            },
        )
