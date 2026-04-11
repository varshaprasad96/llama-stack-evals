"""
Mock custom auth server for Llama Stack experiments.

Maps bearer tokens of the form "token-{tenant}-{user_idx}" to tenant identities.
Implements the custom auth provider protocol expected by Llama Stack:
  POST /auth with {"api_key": "...", "request": {"path": "...", "headers": {...}}}
  Returns {"principal": "...", "attributes": {"namespaces": [...]}}

Usage:
    python scripts/auth_server.py [--port 9999]
"""

import argparse
import logging

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mock Auth Server")


class AuthRequestContext(BaseModel):
    path: str = ""
    headers: dict[str, str] = {}
    params: dict[str, list[str]] = {}


class AuthRequest(BaseModel):
    api_key: str
    request: AuthRequestContext = AuthRequestContext()


class AuthResponse(BaseModel):
    principal: str
    attributes: dict[str, list[str]] | None = None
    message: str | None = None


# Valid tenants for the experiment
VALID_TENANTS = {"finance", "engineering", "legal"}


@app.post("/auth")
async def authenticate(request: AuthRequest) -> AuthResponse:
    """
    Parse bearer token and return tenant identity.

    Token format: "token-{tenant}-{user_idx}"
    Example: "token-finance-0" -> principal="finance-user-0", namespaces=["finance"]
    """
    token = request.api_key

    try:
        parts = token.split("-")
        if len(parts) != 3 or parts[0] != "token":
            raise ValueError(f"Invalid token format: {token}")

        tenant = parts[1]
        user_idx = parts[2]

        if tenant not in VALID_TENANTS:
            raise ValueError(f"Unknown tenant: {tenant}")

        principal = f"{tenant}-user-{user_idx}"
        attributes = {
            "namespaces": [tenant],
            "roles": ["user"],
        }

        logger.debug(f"Authenticated {principal} for {request.request.path}")

        return AuthResponse(
            principal=principal,
            attributes=attributes,
            message=f"Authenticated as {principal}",
        )

    except (ValueError, IndexError) as e:
        logger.warning(f"Auth failed for token '{token}': {e}")
        return AuthResponse(
            principal="anonymous",
            attributes={"namespaces": [], "roles": []},
            message=f"Authentication failed: {e}",
        )


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock auth server")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
