from fastapi import FastAPI
from fastapi.testclient import TestClient

from gcf_qna.app import accounts
from gcf_qna.app.register import router


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_registration_page_matches_chainlit_login_structure(monkeypatch):
    monkeypatch.delenv("ALLOW_SIGNUP", raising=False)

    response = _client().get("/register")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert 'class="auth-layout"' in response.text
    assert 'id="registration-form"' in response.text
    assert 'name="confirm-password"' in response.text
    assert "/public/register.css" in response.text
    assert "/public/register.js" in response.text
    assert "/public/brand/ssa-chatbot-login-background.jpg" in response.text
    assert "Already have an account?" in response.text


def test_registration_page_respects_disabled_flag(monkeypatch):
    monkeypatch.setenv("ALLOW_SIGNUP", "0")

    response = _client().get("/register")

    assert response.status_code == 403
    assert response.text == "Sign-up is disabled."


def test_registration_submit_preserves_account_contract(monkeypatch):
    monkeypatch.delenv("ALLOW_SIGNUP", raising=False)
    monkeypatch.setattr(accounts, "signup_allowed", lambda remote: remote == "203.0.113.4")
    captured = {}

    def create_account(username, password):
        captured.update(username=username, password=password)
        return None

    monkeypatch.setattr(accounts, "create_account", create_account)

    response = _client().post(
        "/register",
        headers={"x-forwarded-for": "203.0.113.4, 10.0.0.1"},
        json={"username": "researcher", "password": "safe-passphrase"},
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert captured == {"username": "researcher", "password": "safe-passphrase"}


def test_registration_submit_maps_reserved_username_to_conflict(monkeypatch):
    monkeypatch.delenv("ALLOW_SIGNUP", raising=False)
    monkeypatch.setattr(accounts, "signup_allowed", lambda remote: True)
    monkeypatch.setattr(
        accounts, "create_account", lambda username, password: "This username is not available."
    )

    response = _client().post(
        "/register", json={"username": "existing", "password": "safe-passphrase"}
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "This username is not available."}


def test_registration_submit_keeps_throttle(monkeypatch):
    monkeypatch.delenv("ALLOW_SIGNUP", raising=False)
    monkeypatch.setattr(accounts, "signup_allowed", lambda remote: False)

    response = _client().post(
        "/register", json={"username": "researcher", "password": "safe-passphrase"}
    )

    assert response.status_code == 429


def test_service_worker_remains_root_scoped():
    response = _client().get("/ssa-sw.js")

    assert response.status_code == 200
    assert response.headers["service-worker-allowed"] == "/"
    assert response.headers["cache-control"] == "no-cache"
