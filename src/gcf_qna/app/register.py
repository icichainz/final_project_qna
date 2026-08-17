"""Self-registration endpoints, mounted ahead of chainlit's SPA catch-all.

GET  /register  — minimal account-creation page (theme-aware, self-contained)
POST /register  — JSON {username, password} -> 200 | 400 | 409 | 403 | 429

Disable with ALLOW_SIGNUP=0. Login stays chainlit's own page; a small
custom.js adds a "Create an account" link there.
"""
from __future__ import annotations

import os

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from gcf_qna.app import accounts

router = APIRouter()

_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Create account</title>
<style>
  :root { color-scheme: light dark; }
  body { margin:0; min-height:100vh; display:flex; align-items:center; justify-content:center;
         font-family: ui-sans-serif, system-ui, sans-serif;
         background:#f4f4f5; color:#18181b; }
  @media (prefers-color-scheme: dark){ body { background:#18181b; color:#e4e4e7; } }
  form { display:flex; flex-direction:column; gap:.8rem; width:min(20rem, 90vw);
         padding:2rem; border-radius:12px; background:rgba(128,128,128,.08);
         border:1px solid rgba(128,128,128,.25); }
  h1 { font-size:1.25rem; margin:0 0 .4rem; }
  input { padding:.6rem .7rem; border-radius:8px; border:1px solid rgba(128,128,128,.4);
          background:transparent; color:inherit; font-size:1rem; }
  button { padding:.65rem; border-radius:8px; border:none; background:#4f46e5; color:#fff;
           font-size:1rem; cursor:pointer; }
  .msg { min-height:1.2rem; font-size:.85rem; }
  .msg.err { color:#dc2626; } .msg.ok { color:#16a34a; }
  a { color:inherit; font-size:.85rem; }
</style></head><body>
<form id="f">
  <h1>Create an account</h1>
  <input name="username" placeholder="Username" autocomplete="username" required>
  <input name="password" type="password" placeholder="Password (min 8 chars)"
         autocomplete="new-password" minlength="8" required>
  <button>Create account</button>
  <div class="msg" id="m"></div>
  <a href="/">Back to sign in</a>
</form>
<script>
document.getElementById('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const m = document.getElementById('m');
  m.className = 'msg';
  const r = await fetch('/register', {method:'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({username: fd.get('username'), password: fd.get('password')})});
  const data = await r.json().catch(() => ({}));
  if (r.ok) {
    m.className='msg ok'; m.textContent='Account created — signing you in…';
    // auto-login so the user never retypes credentials
    const body = new URLSearchParams();
    body.set('username', fd.get('username'));
    body.set('password', fd.get('password'));
    const lr = await fetch('/login', {method:'POST', body}).catch(() => null);
    if (lr && lr.ok) { location.href = '/'; }
    else { m.textContent = 'Account created — please sign in.';
           setTimeout(() => location.href = '/', 1200); }
  }
  else { m.className='msg err'; m.textContent = data.detail || 'Registration failed.'; }
});
</script></body></html>"""


def _signup_enabled() -> bool:
    return os.getenv("ALLOW_SIGNUP", "1") == "1"


@router.get("/register", response_class=HTMLResponse)
async def register_page():
    if not _signup_enabled():
        return HTMLResponse("Sign-up is disabled.", status_code=403)
    return HTMLResponse(_PAGE)


@router.post("/register")
async def register_submit(request: Request):
    if not _signup_enabled():
        return JSONResponse({"detail": "Sign-up is disabled."}, status_code=403)
    remote = request.client.host if request.client else "?"
    if not accounts.signup_allowed(remote):
        return JSONResponse({"detail": "Too many attempts — try again later."}, status_code=429)
    try:
        body = await request.json()
        username, password = str(body["username"]), str(body["password"])
    except Exception:
        return JSONResponse({"detail": "username and password are required."}, status_code=400)
    err = accounts.create_account(username, password)
    if err == "This username is not available.":
        return JSONResponse({"detail": err}, status_code=409)
    if err:
        return JSONResponse({"detail": err}, status_code=400)
    return JSONResponse({"ok": True})


def mount() -> None:
    """Insert our routes BEFORE chainlit's /{full_path:path} SPA catch-all."""
    from chainlit.server import app
    app.router.routes[:0] = router.routes
