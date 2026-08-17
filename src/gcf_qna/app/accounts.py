"""Self-registered accounts: scrypt-hashed credentials in the app database.

Stdlib-only (hashlib.scrypt + per-user salt); credentials live in their own
table beside chainlit's thread tables. Env APP_USERS accounts keep working
and always take precedence at login.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import sqlite3
import time
from typing import Optional

from gcf_qna import config

USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.@-]{2,63}$")
MIN_PASSWORD_LEN = 8
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2**14, 8, 1


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(config.APP_DB, timeout=10)
    con.execute("PRAGMA busy_timeout=10000")
    return con


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    h = hashlib.scrypt(password.encode("utf-8"), salt=salt,
                       n=_SCRYPT_N, r=_SCRYPT_R, p=_SCRYPT_P)
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${h.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, n, r, p, salt_hex, hash_hex = stored.split("$")
        if algo != "scrypt":
            return False
        h = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt_hex),
                           n=int(n), r=int(r), p=int(p))
        return hmac.compare_digest(h.hex(), hash_hex)
    except (ValueError, KeyError):
        return False


def create_account(username: str, password: str) -> Optional[str]:
    """Create an account; returns an error message or None on success."""
    username = username.strip()
    if not USERNAME_RE.match(username):
        return "Username: 3-64 chars, letters/digits/._@- (must start alphanumeric)."
    if len(password) < MIN_PASSWORD_LEN:
        return f"Password must be at least {MIN_PASSWORD_LEN} characters."
    from gcf_qna.app.chainlit_app import _parse_users  # env accounts are reserved
    if username in _parse_users():
        return "This username is not available."
    con = _connect()
    try:
        con.execute(
            'INSERT INTO credentials ("identifier", "passwordHash", "createdAt") '
            "VALUES (?, ?, ?)",
            (username, hash_password(password),
             time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        con.commit()
        return None
    except sqlite3.IntegrityError:
        return "This username is not available."
    finally:
        con.close()


def check_login(username: str, password: str) -> bool:
    con = _connect()
    try:
        row = con.execute(
            'SELECT "passwordHash" FROM credentials WHERE "identifier" = ?',
            (username.strip(),)).fetchone()
    except sqlite3.OperationalError:   # table not created yet
        return False
    finally:
        con.close()
    return bool(row) and verify_password(password, row[0])


# -- tiny in-memory signup throttle (per remote address) ---------------------
_attempts: dict = {}
SIGNUP_LIMIT, SIGNUP_WINDOW_S = 10, 3600


def signup_allowed(remote: str) -> bool:
    now = time.time()
    times = [t for t in _attempts.get(remote, []) if now - t < SIGNUP_WINDOW_S]
    if len(times) >= SIGNUP_LIMIT:
        _attempts[remote] = times
        return False
    times.append(now)
    _attempts[remote] = times
    return True
