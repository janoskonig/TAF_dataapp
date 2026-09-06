"""A főalkalmazás hozzáférési szabályai: a klinikai kódhoz kötött oldalak és a
munkamenet-süti beállításai (a meghívó-link e-mailből is működjön)."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
os.environ.setdefault("FOLLOWUP_ACCESS_CODE", "teszt-kod")

import main  # noqa: E402


@pytest.fixture
def client():
    main.app.testing = True
    return main.app.test_client()


def test_session_cookie_is_lax_so_invite_links_from_email_keep_the_session():
    assert main.app.config["SESSION_COOKIE_SAMESITE"] == "Lax"
    assert main.app.config["SESSION_COOKIE_HTTPONLY"] is True


@pytest.mark.parametrize("path", ["/results", "/student_exam"])
def test_results_and_student_exam_require_clinical_login(client, path):
    response = client.get(path)
    assert response.status_code == 302
    assert response.headers["Location"] == f"/followup/login?next={path}"


def test_student_exam_submission_requires_clinical_login(client):
    response = client.post("/submit_student_exam", data={})
    assert response.status_code == 302 and response.headers["Location"].startswith("/followup/login")


def test_student_exam_opens_after_clinical_login(client):
    with client.session_transaction() as session:
        session["followup_authenticated"] = True
    assert client.get("/student_exam").status_code == 200


def test_expert_invite_link_sets_session_cookie_without_code(monkeypatch):
    monkeypatch.setenv("EXPERT_ACCESS_CODE", "Kata")
    from expert_priors import create_expert_blueprint
    from flask import Flask, Blueprint
    import test_expert_priors as helpers

    row = helpers.response_row(consent_confirmed=False, invited_at="2026-09-06")
    app, _ = helpers.build_app([helpers.SCHEMA_OK, helpers.token_lookup(row), helpers.token_lookup(row)])
    app.config.update(SESSION_COOKIE_SAMESITE="Lax")
    client = app.test_client()
    response = client.get("/expert/meghivo/tok")
    assert response.status_code == 302 and response.headers["Location"] == "/expert"
    cookie = response.headers.get("Set-Cookie", "")
    assert "session=" in cookie and "SameSite=Lax" in cookie
    assert client.get("/expert").status_code == 200   # a start oldal nyílik, nem a kódkérő
