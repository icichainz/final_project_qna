"""Self-registration endpoints, mounted ahead of chainlit's SPA catch-all.

GET  /register  — Chainlit-matched, responsive account-creation page
POST /register  — JSON {username, password} -> 200 | 400 | 409 | 403 | 429

Disable with ALLOW_SIGNUP=0. Login stays chainlit's own page; a small
custom.js adds a "Create an account" link there.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from gcf_qna.app import accounts

router = APIRouter()

_TEMPLATE = Path(__file__).with_name("templates") / "register.html"
_PAGE_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self'; style-src 'self'; script-src 'self'; "
        "connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}


def _signup_enabled() -> bool:
    return os.getenv("ALLOW_SIGNUP", "1") == "1"


@router.get("/ssa-sw.js", include_in_schema=False)
async def service_worker():
    """Serve the service worker at the root so it can control the full app."""
    path = Path(__file__).resolve().parents[3] / "public" / "ssa-sw.js"
    return FileResponse(
        path,
        media_type="application/javascript",
        headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"},
    )


@router.get("/register", response_class=HTMLResponse)
async def register_page():
    if not _signup_enabled():
        return HTMLResponse("Sign-up is disabled.", status_code=403)
    return HTMLResponse(_TEMPLATE.read_text(encoding="utf-8"), headers=_PAGE_HEADERS)


@router.post("/register")
async def register_submit(request: Request):
    if not _signup_enabled():
        return JSONResponse({"detail": "Sign-up is disabled."}, status_code=403)
    # Behind caddy every request.client is the proxy container; use the
    # forwarded client IP so the signup throttle is per-visitor, not global.
    fwd = request.headers.get("x-forwarded-for", "")
    remote = (fwd.split(",")[0].strip() if fwd
              else request.client.host if request.client else "?")
    if not accounts.signup_allowed(remote):
        return JSONResponse({"detail": "Too many attempts — try again later."}, status_code=429)
    try:
        body = await request.json()
        username, password = str(body["username"]), str(body["password"])
    except Exception:
        return JSONResponse({"detail": "username and password are required."}, status_code=400)
    # scrypt (~100 ms of CPU by design) + a sqlite INSERT: blocking work that
    # would otherwise stall every streaming answer in the process
    err = await run_in_threadpool(accounts.create_account, username, password)
    if err == "This username is not available.":
        return JSONResponse({"detail": err}, status_code=409)
    if err:
        return JSONResponse({"detail": err}, status_code=400)
    return JSONResponse({"ok": True})


def mount() -> None:
    """Insert our routes BEFORE chainlit's /{full_path:path} SPA catch-all."""
    from chainlit.server import app
    app.router.routes[:0] = router.routes
