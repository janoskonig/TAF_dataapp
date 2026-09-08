import csv
import io
from pathlib import Path

import pytest
from flask import Blueprint, Flask
from psycopg2.extras import Json

from expert_priors import (
    BACKGROUND_CSV_COLUMNS,
    ITEM_CODES,
    ITEMS,
    PRIOR_CSV_COLUMNS,
    completeness_errors,
    create_expert_blueprint,
    item_progress,
    parse_field_name,
    prior_rows,
)

RESPONSE_COLUMNS = [
    "id", "expert_code", "expert_name", "expert_affiliation", "token", "status", "consent_confirmed",
    "background", "calibration", "items", "closing", "form_version", "submitted_at", "created_at", "updated_at",
    "invited_at", "opened_at", "invite_note", "invite_email", "invite_sent_at", "reminder_sent_at", "invite_deadline",
]
SCHEMA_OK = ("to_regclass('public.expert_prior_responses')", ["responses_table", "name_column"], [("expert_prior_responses", True)])
INSERT_OK = ("INSERT INTO expert_prior_responses", ["id"], [(41,)])
NEXT_CODE_OK = ("SUBSTRING(expert_code FROM", ["next_number"], [(7,)])


def response_row(**overrides):
    row = {
        "id": 7, "expert_code": "SZ07", "expert_name": "Dr. Teszt Elek", "expert_affiliation": "SE Fogpótlástani Klinika",
        "token": "tok", "status": "draft", "consent_confirmed": True,
        "background": {}, "calibration": {"felkeszites_kesz": "1"}, "items": {}, "closing": {}, "form_version": "v1.0",
        "submitted_at": None, "created_at": "2026-09-06 10:00", "updated_at": "2026-09-06 10:05",
        "invited_at": None, "opened_at": None, "invite_note": None,
        "invite_email": None, "invite_sent_at": None, "reminder_sent_at": None, "invite_deadline": None,
    }
    row.update(overrides)
    return row


def token_lookup(row):
    return ("WHERE token = %s", RESPONSE_COLUMNS, [row])


def id_lookup(row):
    return ("WHERE id = %s", RESPONSE_COLUMNS, [row])


def listing(rows):
    return ("FROM expert_prior_responses", RESPONSE_COLUMNS, rows)


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.description = None
        self._rows = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self, sql, params=()):
        self.connection.executions.append((sql, params))
        self.description = None
        self._rows = []
        for matcher, columns, rows in self.connection.script:
            if matcher in sql:
                self.description = [(column,) for column in columns]
                self._rows = [
                    tuple(row[column] for column in columns) if isinstance(row, dict) else row
                    for row in rows
                ]
                break

    def fetchall(self):
        return list(self._rows)

    def fetchone(self):
        return self._rows[0] if self._rows else None


class FakeConnection:
    def __init__(self, script):
        self.script = script
        self.executions = []
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def build_app(script, mail_sender=None):
    connections = []

    def factory():
        connection = FakeConnection(script)
        connections.append(connection)
        return connection

    app = Flask(__name__, root_path=str(Path(__file__).parent))
    app.secret_key = "test-secret"
    app.testing = True
    clinical = Blueprint("followup", __name__, url_prefix="/followup")

    @clinical.route("/login")
    def login():
        return "klinikai belépés"

    app.register_blueprint(clinical)
    app.register_blueprint(create_expert_blueprint(connection_factory=factory, mail_sender=mail_sender))
    return app, connections


def executed_sql(connections):
    return [sql for connection in connections for sql, _ in connection.executions]


def auth_expert(client):
    with client.session_transaction() as session:
        session["expert_authenticated"] = True
        session["expert_csrf"] = "csrf-test"


def auth_admin(client):
    with client.session_transaction() as session:
        session["followup_authenticated"] = True
        session["expert_csrf"] = "csrf-test"


def complete_form():
    form = {
        "csrf_token": "csrf-test",
        "bg__evek_gyakorlat": "22",
        "bg__fogsorok_szama_kat": "500-1000",
        "cal__alap_siker_100": "75",
        "cal__anatomia_sulya_pct": "40",
        "cl__onertekeles": "4",
    }
    for code in ITEM_CODES:
        form[f"item__{code}__irany"] = "B_kedvezotlenebb"
        form[f"item__{code}__p_irany"] = "90"
        form[f"item__{code}__siker_A"] = "85"
        form[f"item__{code}__siker_A_min"] = "75"
        form[f"item__{code}__siker_A_max"] = "92"
        form[f"item__{code}__siker_B"] = "60"
        form[f"item__{code}__siker_B_min"] = "45"
        form[f"item__{code}__siker_B_max"] = "72"
    form["item__F5__sub__lokalizacio"] = "frontalis"
    return form


@pytest.fixture(autouse=True)
def no_access_code(monkeypatch):
    monkeypatch.delenv("EXPERT_ACCESS_CODE", raising=False)


def test_item_codes_match_r_prior_template():
    with open(Path(__file__).parent / "predict_expert_priorok.csv", encoding="utf-8-sig") as handle:
        template_codes = {row["tetel"] for row in csv.DictReader(handle)}
    assert set(ITEM_CODES) <= template_codes | {"A2"}
    assert "A2" not in ITEM_CODES and "A1" in ITEM_CODES
    assert len(ITEMS) == 16


def test_parse_field_name_keeps_sub_question_suffix():
    assert parse_field_name("item__F5__sub__lokalizacio") == ("item", "F5", "sub__lokalizacio")
    assert parse_field_name("bg__oktat") == ("bg", None, "oktat")
    with pytest.raises(ValueError):
        parse_field_name("csrf_token")


def test_start_page_is_open_without_access_code():
    app, _ = build_app([SCHEMA_OK])
    client = app.test_client()
    response = client.get("/expert")
    assert response.status_code == 200
    assert "Kezdjük" in response.get_data(as_text=True)
    assert "Kilépés" not in response.get_data(as_text=True)
    login = client.get("/expert/login")
    assert login.status_code == 302 and login.headers["Location"].endswith("/expert")


def test_optional_access_code_guards_start_page(monkeypatch):
    monkeypatch.setenv("EXPERT_ACCESS_CODE", "szakerto-kod")
    app, _ = build_app([SCHEMA_OK])
    client = app.test_client()
    response = client.get("/expert")
    assert response.status_code == 302
    assert "/expert/login" in response.headers["Location"]


def test_login_with_correct_code_opens_session(monkeypatch):
    monkeypatch.setenv("EXPERT_ACCESS_CODE", "szakerto-kod")
    app, _ = build_app([SCHEMA_OK])
    client = app.test_client()
    with client.session_transaction() as session:
        session["expert_csrf"] = "csrf-test"
    response = client.post("/expert/login", data={"csrf_token": "csrf-test", "access_code": "szakerto-kod"})
    assert response.status_code == 302
    with client.session_transaction() as session:
        assert session.get("expert_authenticated") is True


def test_start_response_inserts_row_and_assigns_expert_code():
    app, connections = build_app([SCHEMA_OK, INSERT_OK, NEXT_CODE_OK])
    client = app.test_client()
    auth_expert(client)
    response = client.post("/expert/start", data={"csrf_token": "csrf-test", "consent": "on", "expert_name": " Dr. Teszt Elek ", "expert_affiliation": "SE"})
    assert response.status_code == 302
    assert "/expert/urlap/" in response.headers["Location"]
    statements = [(sql, params) for connection in connections for sql, params in connection.executions]
    assert any("INSERT INTO expert_prior_responses" in sql and params[1] == "Dr. Teszt Elek" and params[2] == "SE" for sql, params in statements)
    assert any("SET expert_code = %s" in sql and params[0] == "SZ07" for sql, params in statements)
    assert connections[-1].committed


def test_start_without_consent_or_name_is_refused():
    app, connections = build_app([SCHEMA_OK, SCHEMA_OK])
    client = app.test_client()
    auth_expert(client)
    response = client.post("/expert/start", data={"csrf_token": "csrf-test", "expert_name": "Dr. Teszt Elek"})
    assert response.status_code == 302
    response = client.post("/expert/start", data={"csrf_token": "csrf-test", "consent": "on", "expert_name": "  "})
    assert response.status_code == 302
    assert not any("INSERT" in sql for sql in executed_sql(connections))


def test_form_page_renders_all_items():
    app, _ = build_app([SCHEMA_OK, token_lookup(response_row())])
    client = app.test_client()
    auth_expert(client)
    html = client.get("/expert/urlap/tok").get_data(as_text=True)
    for code in ITEM_CODES:
        assert f'id="item-{code}"' in html
    assert 'id="itemsDone">0</span> / 16 adottság' in html
    assert 'name="bg__diploma_ev"' in html and 'name="bg__szakvizsga"' in html
    assert 'type="range" id="cal__felso_also_arany"' in html and 'data-needs-jaw' in html


def test_autosave_writes_sub_question_into_item_json():
    app, connections = build_app([token_lookup(response_row())])
    client = app.test_client()
    auth_expert(client)
    response = client.post(
        "/expert/urlap/tok/mentes",
        data={"csrf_token": "csrf-test", "field": "item__F5__sub__lokalizacio", "value": "frontalis"},
    )
    assert response.status_code == 200
    assert response.get_json() == {"ok": True}
    update = [(sql, params) for connection in connections for sql, params in connection.executions if "jsonb_set" in sql]
    assert len(update) == 1
    sql, params = update[0]
    assert params[0] == "F5"
    assert isinstance(params[2], Json)
    assert params[2].adapted == {"sub_lokalizacio": "frontalis"}


def test_autosave_rejects_certainty_outside_scale():
    app, connections = build_app([token_lookup(response_row())])
    client = app.test_client()
    auth_expert(client)
    response = client.post(
        "/expert/urlap/tok/mentes",
        data={"csrf_token": "csrf-test", "field": "item__F1__p_irany", "value": "73"},
    )
    assert response.status_code == 400
    assert not any("UPDATE" in sql for sql in executed_sql(connections))


def test_autosave_refuses_submitted_response():
    app, _ = build_app([token_lookup(response_row(status="submitted"))])
    client = app.test_client()
    auth_expert(client)
    response = client.post(
        "/expert/urlap/tok/mentes",
        data={"csrf_token": "csrf-test", "field": "bg__oktat", "value": "igen"},
    )
    assert response.status_code == 409


def test_submit_incomplete_form_is_rejected_with_error_list():
    app, connections = build_app([token_lookup(response_row())])
    client = app.test_client()
    auth_expert(client)
    response = client.post("/expert/urlap/tok/bekuldes", data={"csrf_token": "csrf-test", "item__F1__irany": "B_kedvezotlenebb"})
    assert response.status_code == 400
    html = response.get_data(as_text=True)
    assert "F1 · A felső gerinc magassága: mennyire biztos" in html
    assert "Sikeres fogsorok aránya 100 betegből" in html
    assert not any("status = 'submitted'" in sql for sql in executed_sql(connections))


def test_submit_complete_form_marks_response_submitted():
    app, connections = build_app([token_lookup(response_row())])
    client = app.test_client()
    auth_expert(client)
    response = client.post("/expert/urlap/tok/bekuldes", data=complete_form())
    assert response.status_code == 302
    assert response.headers["Location"].endswith("/expert/kesz/tok")
    submits = [(sql, params) for connection in connections for sql, params in connection.executions if "status = 'submitted'" in sql]
    assert len(submits) == 1
    items = submits[0][1][2].adapted
    assert items["A11"]["siker_A"] == 85 and items["A11"]["siker_A_max"] == 92 and items["A11"]["siker_B_min"] == 45
    assert items["F5"]["sub_lokalizacio"] == "frontalis"
    assert submits[0][1][0].adapted["evek_gyakorlat"] == 22


def test_export_requires_clinical_session():
    app, _ = build_app([SCHEMA_OK])
    client = app.test_client()
    auth_expert(client)
    response = client.get("/expert/admin/export/priorok.csv")
    assert response.status_code == 302
    assert "/followup/login" in response.headers["Location"]


def test_prior_export_matches_r_script_template():
    submitted = response_row(
        status="submitted",
        items={
            "F5": {"irany": "B_kedvezotlenebb", "p_irany": 80, "siker_A": 80, "siker_B": 55,
                   "siker_A_min": 70, "siker_A_max": 90, "siker_B_min": 40, "siker_B_max": 65,
                   "sub_lokalizacio": "frontalis", "megjegyzes": "ritka"},
            "F2": {"irany": "nem_monoton"},
        },
    )
    app, _ = build_app([listing([submitted])])
    client = app.test_client()
    auth_admin(client)
    response = client.get("/expert/admin/export/priorok.csv")
    assert response.status_code == 200
    text = response.get_data(as_text=True).lstrip("﻿")
    rows = list(csv.DictReader(io.StringIO(text)))
    assert list(rows[0].keys()) == PRIOR_CSV_COLUMNS
    assert len(rows) == len(ITEM_CODES)
    by_code = {row["tetel"]: row for row in rows}
    assert by_code["F5"]["irany"] == "B_kedvezotlenebb" and by_code["F5"]["p_irany"] == "80"
    assert by_code["F5"]["mechanizmus"] == ""
    assert "lokalizacio=a frontális gerincen" in by_code["F5"]["megjegyzes"]
    assert by_code["F2"]["alak"] == "optimum"
    assert by_code["A3"]["irany"] == ""
    assert by_code["F5"]["siker_A_min"] == "70" and by_code["F5"]["siker_B_max"] == "65" and by_code["F5"]["nagysag_nem_tudom"] == ""
    assert by_code["F1"]["polus_A_ertek"] == "10.0" and by_code["F1"]["polus_B_ertek"] == "5.0" and by_code["F1"]["polus_egyseg"] == "mm"
    assert by_code["F6"]["polus_M_ertek"] == "10.0" and by_code["A4"]["polus_A_ertek"] == ""


def test_background_export_columns():
    submitted = response_row(status="submitted", background={"evek_gyakorlat": 30, "fogsorok_szama_kat": ">3000"},
                             calibration={"alap_siker_100": 70, "anatomia_sulya_pct": 35}, closing={"rang_1": "A11", "onertekeles": "4"})
    app, _ = build_app([listing([submitted])])
    client = app.test_client()
    auth_admin(client)
    text = client.get("/expert/admin/export/hatter.csv").get_data(as_text=True).lstrip("﻿")
    rows = list(csv.DictReader(io.StringIO(text)))
    assert list(rows[0].keys()) == BACKGROUND_CSV_COLUMNS
    assert rows[0]["rang_1"] == "A11" and rows[0]["alap_siker_100"] == "70"
    assert rows[0]["nev"] == "Dr. Teszt Elek" and rows[0]["intezmeny"] == "SE Fogpótlástani Klinika"


def test_admin_pages_render():
    submitted = response_row(status="submitted", items={"A1": {"irany": "B_kedvezotlenebb", "p_irany": 95}})
    app, _ = build_app([SCHEMA_OK, id_lookup(submitted), listing([submitted, response_row(id=8, expert_code="SZ08", token="tok8")])])
    client = app.test_client()
    auth_admin(client)
    html = client.get("/expert/admin").get_data(as_text=True)
    assert "SZ07" in html and "SZ08" in html and "Tételenkénti gyorsösszesítés" in html
    assert "Dr. Teszt Elek" in html
    view = client.get("/expert/admin/7").get_data(as_text=True)
    assert "a B változat a rosszabb" in view and "Válasz törlése" in view


def test_admin_delete_requires_code_confirmation():
    row = response_row(status="submitted")
    app, connections = build_app([id_lookup(row)])
    client = app.test_client()
    auth_admin(client)
    refused = client.post("/expert/admin/7/torles", data={"csrf_token": "csrf-test", "confirm": "rossz"})
    assert refused.status_code == 302
    assert not any("DELETE" in sql for sql in executed_sql(connections))
    accepted = client.post("/expert/admin/7/torles", data={"csrf_token": "csrf-test", "confirm": "SZ07"})
    assert accepted.status_code == 302
    assert any("DELETE FROM expert_prior_responses" in sql for sql in executed_sql(connections))


def test_helpers_progress_and_completeness():
    items = {"F1": {"irany": "B_kedvezotlenebb", "p_irany": 90}, "F2": {"irany": "nem_monoton"}, "F3": {"irany": "A_kedvezotlenebb"}}
    assert item_progress(items) == 2
    problems = completeness_errors({"items": items, "background": {}, "calibration": {}, "closing": {}})
    assert any("F3" in problem and "mennyire biztos" in problem for problem in problems)
    rows = prior_rows([response_row(status="submitted", items=items)])
    assert len(rows) == 16


def test_simulated_rows_are_flagged_and_excluded_by_default():
    real = response_row(status="submitted", items={"A1": {"irany": "B_kedvezotlenebb", "p_irany": 95}})
    simulated = response_row(id=9, expert_code="SZIM01", token="tok9", status="submitted", form_version="v1.0-SZIMULACIO",
                             items={"A1": {"irany": "A_kedvezotlenebb", "p_irany": 99}})
    app, _ = build_app([SCHEMA_OK, listing([real, simulated])])
    client = app.test_client()
    auth_admin(client)
    html = client.get("/expert/admin").get_data(as_text=True)
    assert "szimulált" in html and "1</strong><span>beküldött" in html
    text = client.get("/expert/admin/export/priorok.csv").get_data(as_text=True)
    assert "SZIM01" not in text and "SZ07" in text
    text_all = client.get("/expert/admin/export/priorok.csv?szimulacio=1").get_data(as_text=True)
    assert "SZIM01" in text_all


def test_mechanism_codes_remain_readable_but_are_not_asked():
    from expert_priors import MECHANISM_CODES
    assert "retencio" in MECHANISM_CODES and "szivohatas" in MECHANISM_CODES
    app, _ = build_app([SCHEMA_OK, token_lookup(response_row())])
    client = app.test_client()
    auth_expert(client)
    html = client.get("/expert/urlap/tok").get_data(as_text=True)
    assert "__mechanizmus" not in html


def test_english_version_renders_and_is_remembered():
    app, _ = build_app([SCHEMA_OK, SCHEMA_OK, token_lookup(response_row())])
    client = app.test_client()
    auth_expert(client)
    html = client.get("/expert?lang=en").get_data(as_text=True)
    assert "What has experience taught you about complete dentures?" in html
    form = client.get("/expert/urlap/tok").get_data(as_text=True)
    assert "Sagittal relation of the jaws" in form and "Which variant is worse" in form
    assert 'name="bg__nyelv" value="en"' in form
    back = client.get("/expert?lang=hu").get_data(as_text=True)
    assert "Mit tanított Önnek a tapasztalat" in back


def test_a10_is_named_sagittal_relation_in_hungarian():
    from expert_priors import ITEMS_BY_CODE
    assert ITEMS_BY_CODE["A10"]["nev"] == "Az állcsontok sagittális relációja"
    assert "magasság" in ITEMS_BY_CODE["A1"]["nev"]


def test_admin_can_create_invitation_with_link():
    app, connections = build_app([SCHEMA_OK, INSERT_OK, NEXT_CODE_OK])
    client = app.test_client()
    auth_admin(client)
    response = client.post("/expert/admin/meghivo", data={"csrf_token": "csrf-test", "expert_name": "Dr. Meghívott Mária", "lang": "en", "invite_note": "maria@example.org"})
    assert response.status_code == 302 and "uj=SZ07" in response.headers["Location"]
    inserts = [(sql, params) for connection in connections for sql, params in connection.executions if "INSERT INTO expert_prior_responses" in sql]
    assert len(inserts) == 1
    params = inserts[0][1]
    assert params[1] == "Dr. Meghívott Mária" and params[4] is False and params[7] is True and params[8] == "maria@example.org"
    assert params[6].adapted == {"nyelv": "en", "szerep": "fogorvos"}
    assert params[9] is None and params[10] is None


def test_invite_link_logs_in_without_code_and_asks_for_consent(monkeypatch):
    monkeypatch.setenv("EXPERT_ACCESS_CODE", "szakerto-kod")
    invited = response_row(consent_confirmed=False, invited_at="2026-09-06 09:00", background={"nyelv": "en"})
    app, _ = build_app([SCHEMA_OK, token_lookup(invited)])
    client = app.test_client()
    response = client.get("/expert/meghivo/tok")
    assert response.status_code == 302 and response.headers["Location"].endswith("/expert")
    with client.session_transaction() as session:
        assert session.get("expert_authenticated") is True and session.get("expert_token") == "tok" and session.get("expert_lang") == "en"
    start = client.get("/expert").get_data(as_text=True)
    assert "Personal invitation" in start and 'value="Dr. Teszt Elek"' in start and 'name="invite_token" value="tok"' in start
    assert "Start a new questionnaire" not in start


def test_accepting_invitation_records_consent_and_opens_form():
    app, connections = build_app([SCHEMA_OK])
    client = app.test_client()
    with client.session_transaction() as session:
        session["expert_authenticated"] = True
        session["expert_csrf"] = "csrf-test"
        session["expert_token"] = "tok"
    response = client.post("/expert/start", data={"csrf_token": "csrf-test", "consent": "on", "expert_name": "Dr. Teszt Elek", "invite_token": "tok"})
    assert response.status_code == 302 and response.headers["Location"].endswith("/expert/urlap/tok")
    updates = [sql for sql in executed_sql(connections) if "consent_confirmed = TRUE" in sql and "opened_at" in sql]
    assert len(updates) == 1
    assert not any("INSERT" in sql for sql in executed_sql(connections))


def test_invalid_invite_link_shows_friendly_page():
    app, _ = build_app([SCHEMA_OK, ("WHERE token = %s", RESPONSE_COLUMNS, [])])
    client = app.test_client()
    response = client.get("/expert/meghivo/nincs-ilyen")
    assert response.status_code == 404
    assert "érvénytelen" in response.get_data(as_text=True)


def test_invitation_email_is_sent_through_sendgrid_sender():
    sent = []

    def fake_sender(to_email, to_name, subject, text, html):
        sent.append((to_email, to_name, subject, text, html))

    app, connections = build_app([SCHEMA_OK, INSERT_OK, NEXT_CODE_OK], mail_sender=fake_sender)
    client = app.test_client()
    auth_admin(client)
    response = client.post("/expert/admin/meghivo", data={
        "csrf_token": "csrf-test", "expert_name": "Dr. Meghívott Mária", "lang": "en",
        "invite_email": "maria@example.org", "send_now": "on",
    })
    assert response.status_code == 302
    assert len(sent) == 1
    to_email, to_name, subject, text, html = sent[0]
    assert to_email == "maria@example.org" and to_name == "Dr. Meghívott Mária"
    assert subject.startswith("A request for your experience")
    from expert_priors import deadline_iso, format_deadline
    assert "/expert/meghivo/" in text and f"within two weeks, by {format_deadline(deadline_iso(), 'en')}." in text and "Dear Dr. Meghívott Mária" in text
    assert "<a href=" in html
    assert any("invite_sent_at = CURRENT_TIMESTAMP" in sql and "invite_deadline = COALESCE(invite_deadline, %s)" in sql for sql in executed_sql(connections))
    deadline_params = [params for c in connections for sql, params in c.executions if "invite_deadline = COALESCE" in sql]
    assert deadline_params[0][0] == deadline_iso()


def test_reminder_uses_stored_email_and_language():
    sent = []
    row = response_row(consent_confirmed=False, invited_at="2026-09-06", invite_email="elek@example.org",
                       invite_sent_at="2026-09-06 10:00", background={"nyelv": "hu"})
    app, connections = build_app([id_lookup(row)], mail_sender=lambda *args: sent.append(args))
    client = app.test_client()
    auth_admin(client)
    response = client.post("/expert/admin/7/email", data={"csrf_token": "csrf-test"})
    assert response.status_code == 302
    assert len(sent) == 1 and sent[0][0] == "elek@example.org" and sent[0][2] == "Emlékeztető: PREDICT szakértői kérdőív"
    assert "Tisztelt Dr. Teszt Elek!" in sent[0][3]
    assert any("reminder_sent_at = CURRENT_TIMESTAMP" in sql for sql in executed_sql(connections))


def test_mail_failure_keeps_invitation_and_reports_error():
    from expert_priors import MailError

    def failing_sender(*_args):
        raise MailError("nincs kulcs")

    app, connections = build_app([SCHEMA_OK, INSERT_OK, NEXT_CODE_OK], mail_sender=failing_sender)
    client = app.test_client()
    auth_admin(client)
    response = client.post("/expert/admin/meghivo", data={"csrf_token": "csrf-test", "expert_name": "Dr. X", "invite_email": "x@example.org", "send_now": "on"}, follow_redirects=False)
    assert response.status_code == 302
    assert any("INSERT INTO expert_prior_responses" in sql for sql in executed_sql(connections))
    assert not any("invite_sent_at = CURRENT_TIMESTAMP" in sql for sql in executed_sql(connections))
    with client.session_transaction() as session:
        flashes = session.get("_flashes", [])
    assert any("nem ment el" in message for _, message in flashes)


def test_smtp_sender_requires_configuration(monkeypatch):
    import expert_priors
    from expert_priors import MailError, smtp_send
    for key in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "SMTP_PASS", "SMTP_SENDER_EMAIL", "SMTP_FROM"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(expert_priors, "SIBLING_ENV_CANDIDATES", [])
    assert expert_priors.mail_configured() is False
    with pytest.raises(MailError):
        smtp_send("a@example.org", "A", "s", "t", "<p>t</p>")


def test_smtp_sender_builds_shadematch_style_message(monkeypatch):
    import expert_priors
    monkeypatch.setenv("SMTP_HOST", "smtp.sendgrid.net")
    monkeypatch.setenv("SMTP_USER", "apikey")
    monkeypatch.setenv("SMTP_PASSWORD", "SG.test")
    monkeypatch.setenv("SMTP_SENDER_EMAIL", "predict@example.org")
    monkeypatch.setenv("EMAIL_FROM_NAME", "PREDICT")
    sent = {}

    class FakeServer:
        def __init__(self, host, port, timeout=None):
            sent["host"], sent["port"] = host, port
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def starttls(self):
            sent["tls"] = True
        def login(self, user, password):
            sent["login"] = (user, password)
        def send_message(self, message):
            sent["message"] = message

    monkeypatch.setattr(expert_priors.smtplib, "SMTP", FakeServer)
    expert_priors.smtp_send("kollega@example.org", "Dr. Kolléga", "Tárgy", "szöveg http://x", "<p>szöveg</p>")
    assert sent["host"] == "smtp.sendgrid.net" and sent["port"] == 587 and sent["tls"] is True
    assert sent["login"] == ("apikey", "SG.test")
    message = sent["message"]
    assert message["From"].endswith("<predict@example.org>") and "Dr. Kolléga" in message["To"]
    assert message["Auto-Submitted"] == "auto-generated" and message["Message-ID"]


def test_sendgrid_api_key_alone_configures_smtp_relay(monkeypatch):
    import expert_priors
    for key in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "SMTP_PASS", "SMTP_SENDER_EMAIL"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(expert_priors, "SIBLING_ENV_CANDIDATES", [])
    monkeypatch.setenv("SendGridAPI_Key", "SG.kulcs")
    monkeypatch.setenv("SMTP_FROM", "predict@example.org")
    settings = expert_priors.resolve_mail_settings()
    assert settings["host"] == "smtp.sendgrid.net" and settings["user"] == "apikey" and settings["password"] == "SG.kulcs"
    assert settings["sender"] == "predict@example.org" and expert_priors.mail_configured() is True
    monkeypatch.setenv("SMTP_PASS", "${SendGridAPI_Key}")
    assert expert_priors.resolve_mail_settings()["password"] == "SG.kulcs"


def test_pages_show_the_predict_logo_and_favicon():
    app, _ = build_app([SCHEMA_OK])
    client = app.test_client()
    auth_expert(client)
    html = client.get("/expert").get_data(as_text=True)
    assert 'class="brand-logo" src="/static/predict-mark.svg"' in html
    assert 'rel="icon" type="image/svg+xml" href="/static/predict-mark.svg"' in html


def test_email_html_carries_both_logos_when_urls_given():
    from expert_priors import build_email
    _, _, html = build_email("invite", "hu", "Dr. X", "https://x/expert/meghivo/a", None,
                             logo_url="https://x/static/predict-logo.png", partner_logo_url="https://x/static/semmelweis-logo.png")
    assert '<img src="https://x/static/predict-logo.png"' in html
    assert '<img src="https://x/static/semmelweis-logo.png" alt="Semmelweis Egyetem"' in html
    assert html.index("predict-logo.png") < html.index("semmelweis-logo.png") < html.index("Tisztelt Dr. X!")
    _, _, plain = build_email("invite", "hu", "Dr. X", "https://x/expert/meghivo/a")
    assert "<img" not in plain and "<table" not in plain


def test_sender_name_always_carries_predict(monkeypatch):
    import expert_priors
    monkeypatch.setenv("EMAIL_FROM_NAME", "Dr. König János")
    assert expert_priors.mail_from_name() == "Dr. König János (PREDICT)"
    monkeypatch.setenv("EMAIL_FROM_NAME", "Dr. König János · PREDICT-vizsgálat")
    assert expert_priors.mail_from_name() == "Dr. König János · PREDICT-vizsgálat"


def test_signature_literal_backslash_n_becomes_line_breaks(monkeypatch):
    from expert_priors import build_email
    monkeypatch.setenv("EXPERT_MAIL_SIGNATURE", "Dr. König János\\nFogpótlástani Klinika\\nkonig.janos@semmelweis.hu")
    _, text, html = build_email("invite", "hu", "Dr. X", "https://predict-study.hu/expert/meghivo/a")
    assert "\\n" not in text and "Tisztelettel,\nDr. König János\nFogpótlástani Klinika\nkonig.janos@semmelweis.hu" in text
    assert "Dr. König János<br>Fogpótlástani Klinika<br>konig.janos@semmelweis.hu" in html


def test_invitation_expands_acronym_once_and_mentions_bayes():
    from expert_priors import build_email
    from expert_texts import STUDY_ACRONYM
    for lang, word in (("hu", "Bayes-i"), ("en", "Bayesian")):
        _, text, _ = build_email("invite", lang, "Dr. X", "https://predict-study.hu/expert/meghivo/a")
        assert text.count(STUDY_ACRONYM[lang]) == 1 and word in text
        _, reminder, _ = build_email("reminder", lang, "Dr. X", "https://predict-study.hu/expert/meghivo/a")
        assert STUDY_ACRONYM[lang] not in reminder


def test_links_use_configured_public_base_url_not_request_host(monkeypatch):
    monkeypatch.setenv("APP_BASE_URL", "https://predict-study.hu/")
    render_host = "https://taf-hcax.onrender.com"

    def auth_admin_at_render_host(client):
        with client.session_transaction(base_url=render_host) as session:
            session["followup_authenticated"] = True
            session["expert_csrf"] = "csrf-test"

    sent = []
    app, connections = build_app([SCHEMA_OK, INSERT_OK, NEXT_CODE_OK], mail_sender=lambda *args: sent.append(args))
    client = app.test_client()
    auth_admin_at_render_host(client)
    response = client.post("/expert/admin/meghivo", base_url=render_host, data={
        "csrf_token": "csrf-test", "expert_name": "Dr. X", "invite_email": "x@example.org", "send_now": "on",
    })
    assert response.status_code == 302
    text, html = sent[0][3], sent[0][4]
    assert "https://predict-study.hu/expert/meghivo/" in text and "onrender.com" not in text
    assert '<img src="https://predict-study.hu/static/predict-logo.png"' in html
    assert '<img src="https://predict-study.hu/static/semmelweis-logo.png"' in html
    row = response_row(consent_confirmed=False, invited_at="2026-09-06")
    app, _ = build_app([SCHEMA_OK, listing([row])])
    client = app.test_client()
    auth_admin_at_render_host(client)
    page = client.get("/expert/admin", base_url=render_host).get_data(as_text=True)
    assert "https://predict-study.hu/expert/meghivo/" in page and "https://predict-study.hu/expert" in page
    assert "onrender.com" not in page


def test_start_page_offers_role_choice_and_stores_it():
    app, connections = build_app([SCHEMA_OK, INSERT_OK, NEXT_CODE_OK])
    client = app.test_client()
    auth_expert(client)
    page = client.get("/expert").get_data(as_text=True)
    assert 'name="szerep" value="fogtechnikus"' in page and "fogtechnikusként" in page
    assert "fogorvosoknak és fogtechnikusoknak" in page
    response = client.post("/expert/start", data={"csrf_token": "csrf-test", "consent": "on", "expert_name": "Kovács Fogtechnikus", "szerep": "fogtechnikus"})
    assert response.status_code == 302 and "/expert/urlap/" in response.headers["Location"]
    inserts = [params for connection in connections for sql, params in connection.executions if "INSERT INTO expert_prior_responses" in sql]
    assert inserts[0][6].adapted == {"nyelv": "hu", "szerep": "fogtechnikus"}


def test_technician_form_shows_technician_background_and_all_items():
    row = response_row(background={"nyelv": "hu", "szerep": "fogtechnikus"})
    app, _ = build_app([SCHEMA_OK, token_lookup(row)])
    client = app.test_client()
    auth_expert(client)
    page = client.get("/expert/urlap/tok").get_data(as_text=True)
    assert 'name="bg__szerep" value="fogtechnikus"' in page
    assert 'name="bg__kepesites_ev"' in page and 'name="bg__mester"' in page
    assert 'name="bg__diploma_ev"' not in page and 'name="bg__szakvizsga"' not in page
    assert "a laborból nem látszik" in page and "mesterfogtechnikusi" in page
    assert page.count('class="card section-card expert-item"') == len(ITEM_CODES)
    dentist = response_row(background={"nyelv": "hu"})
    app, _ = build_app([SCHEMA_OK, token_lookup(dentist)])
    client = app.test_client()
    auth_expert(client)
    page = client.get("/expert/urlap/tok").get_data(as_text=True)
    assert 'name="bg__diploma_ev"' in page and 'name="bg__kepesites_ev"' not in page and 'value="fogorvos"' in page


def test_invited_technician_sees_role_and_keeps_it_on_acceptance():
    row = response_row(consent_confirmed=False, invited_at="2026-09-06", background={"nyelv": "hu", "szerep": "fogtechnikus"})
    app, connections = build_app([SCHEMA_OK, token_lookup(row), token_lookup(row)])
    client = app.test_client()
    client.get("/expert/meghivo/tok")
    with client.session_transaction() as session:
        session["expert_csrf"] = "csrf-test"
    page = client.get("/expert").get_data(as_text=True)
    assert 'name="szerep" value="fogtechnikus"' in page and "fogtechnikusoknak" in page
    response = client.post("/expert/start", data={"csrf_token": "csrf-test", "consent": "on", "expert_name": "Kovács Fogtechnikus", "invite_token": "tok", "szerep": "fogtechnikus"})
    assert response.status_code == 302
    updates = [params for connection in connections for sql, params in connection.executions if "SET consent_confirmed = TRUE" in sql]
    assert updates and updates[0][2].adapted == {"szerep": "fogtechnikus"}


def test_technician_submission_keeps_role_and_accepts_technician_fields():
    row = response_row(background={"nyelv": "hu", "szerep": "fogtechnikus"})
    app, connections = build_app([SCHEMA_OK, token_lookup(row)])
    client = app.test_client()
    auth_expert(client)
    form = complete_form()
    form.update({"bg__kepesites_ev": "1998", "bg__mester": "igen", "bg__visszajelzes": "kozvetlen_beteg", "item__A5__irany": "nem_tudom"})
    form.pop("item__A5__p_irany")
    response = client.post("/expert/urlap/tok/bekuldes", data=form)
    assert response.status_code == 302
    updates = [params for connection in connections for sql, params in connection.executions if "status = 'submitted'" in sql]
    background = updates[0][0].adapted
    assert background["szerep"] == "fogtechnikus" and background["kepesites_ev"] == 1998 and background["mester"] == "igen"
    assert updates[0][2].adapted["A5"]["irany"] == "nem_tudom"


def test_exports_carry_the_role_column():
    from expert_priors import BACKGROUND_CSV_COLUMNS, PRIOR_CSV_COLUMNS, background_rows, prior_rows
    rows = [response_row(status="submitted", background={"nyelv": "hu", "szerep": "fogtechnikus", "kepesites_ev": 1998, "mester": "igen", "evek_gyakorlat": 20})]
    assert "szerep" in PRIOR_CSV_COLUMNS and BACKGROUND_CSV_COLUMNS[-3:] == ["szerep", "visszajelzes", "ajanlo_kod"]
    assert {r["szerep"] for r in prior_rows(rows)} == {"fogtechnikus"}
    background = background_rows(rows)[0]
    assert background["szerep"] == "fogtechnikus" and "képesítés: 1998" in background["megjegyzes"] and "mesterfogtechnikus: igen" in background["megjegyzes"]
    assert background_rows([response_row(status="submitted")])[0]["szerep"] == "fogorvos"


def test_admin_invite_with_role_sends_technician_letter_and_lists_badge():
    sent = []
    app, connections = build_app([SCHEMA_OK, INSERT_OK, NEXT_CODE_OK], mail_sender=lambda *args: sent.append(args))
    client = app.test_client()
    auth_admin(client)
    response = client.post("/expert/admin/meghivo", data={
        "csrf_token": "csrf-test", "expert_name": "Kovács Fogtechnikus", "szerep": "fogtechnikus",
        "invite_email": "kovacs@example.org", "send_now": "on",
    })
    assert response.status_code == 302
    inserts = [params for connection in connections for sql, params in connection.executions if "INSERT INTO expert_prior_responses" in sql]
    assert inserts[0][6].adapted == {"nyelv": "hu", "szerep": "fogtechnikus"}
    subject, text = sent[0][2], sent[0][3]
    assert "fogtechnikus kollégáknak" in subject and "a saját munkája" in text and "a laborból nem lehet megítélni" in text
    row = response_row(status="submitted", submitted_at="2026-09-06 12:00", background={"nyelv": "hu", "szerep": "fogtechnikus"})
    app, _ = build_app([SCHEMA_OK, listing([row])])
    client = app.test_client()
    auth_admin(client)
    page = client.get("/expert/admin").get_data(as_text=True)
    assert '<span class="badge badge-muted">fogtechnikus</span>' in page and "fogtechnikusok (1)" in page
    filtered = client.get("/expert/admin?szerep=fogorvos").get_data(as_text=True)
    assert "<strong>fogorvosok (0)</strong>" in filtered


def _submit(form_overrides, background=None):
    row = response_row(background=background or {"nyelv": "hu"})
    app, connections = build_app([SCHEMA_OK, token_lookup(row)])
    client = app.test_client()
    auth_expert(client)
    form = complete_form()
    form.update(form_overrides)
    response = client.post("/expert/urlap/tok/bekuldes", data=form)
    return response, connections


def test_submission_flags_numbers_that_contradict_the_marked_direction():
    response, _ = _submit({"item__A11__siker_A": "20", "item__A11__siker_A_min": "10", "item__A11__siker_A_max": "30",
                           "item__A11__siker_B": "90", "item__A11__siker_B_min": "80", "item__A11__siker_B_max": "95"})
    assert response.status_code == 400
    page = response.get_data(as_text=True)
    assert "A11" in page and "a számok szerint a(z) A változat a rosszabb" in page


def test_submission_requires_point_inside_its_range():
    response, _ = _submit({"item__F1__siker_A": "95"})   # tartomány 75–92
    assert response.status_code == 400 and "F1" in response.get_data(as_text=True)
    assert "legvalószínűbb szám nincs az alsó és a felső határ között" in response.get_data(as_text=True)


def test_no_difference_answer_with_large_gap_is_rejected():
    response, _ = _submit({"item__F7__irany": "nincs_kulonbseg", "item__F7__siker_A": "85", "item__F7__siker_B": "60"})
    assert response.status_code == 400 and "de a két szám 25 beteggel tér el" in response.get_data(as_text=True)


def test_optimum_answer_needs_middle_scenario_to_be_best():
    overrides = {"item__F2__irany": "nem_monoton", "item__F2__siker_A": "70", "item__F2__siker_M": "60", "item__F2__siker_B": "50",
                 "item__F2__siker_A_min": "60", "item__F2__siker_A_max": "80", "item__F2__siker_M_min": "50", "item__F2__siker_M_max": "70"}
    response, _ = _submit(overrides)
    page = response.get_data(as_text=True)
    assert response.status_code == 400 and "a közepes szám nem a legmagasabb" in page
    overrides["item__F2__siker_M"] = "80"; overrides["item__F2__siker_M_max"] = "90"
    response, connections = _submit(overrides)
    assert response.status_code == 302
    items = [params for c in connections for sql, params in c.executions if "status = 'submitted'" in sql][0][2].adapted
    assert items["F2"]["siker_M"] == 80 and items["F2"]["irany"] == "nem_monoton"


def test_magnitude_unknown_replaces_the_numbers():
    missing = {f"item__A5__{k}": "" for k in ("siker_A", "siker_A_min", "siker_A_max", "siker_B", "siker_B_min", "siker_B_max")}
    response, _ = _submit(missing)
    assert response.status_code == 400 and "vagy jelölje, hogy a nagyságot nem tudja megbecsülni" in response.get_data(as_text=True)
    response, connections = _submit({**missing, "item__A5__nagysag_nem_tudom": "1"})
    assert response.status_code == 302
    items = [params for c in connections for sql, params in c.executions if "status = 'submitted'" in sql][0][2].adapted
    assert items["A5"]["nagysag_nem_tudom"] is True and items["A5"]["siker_A"] is None


def test_directional_answer_without_ranges_is_incomplete():
    response, _ = _submit({"item__F3__siker_A_min": "", "item__F3__siker_A_max": ""})
    assert response.status_code == 400 and "kérjük az alsó és felső határt is" in response.get_data(as_text=True)


def test_technician_must_say_where_feedback_comes_from():
    background = {"nyelv": "hu", "szerep": "fogtechnikus"}
    response, _ = _submit({"bg__kepesites_ev": "1998"}, background=background)
    assert response.status_code == 400 and "Miből tudja meg, hogyan vált be a fogsor" in response.get_data(as_text=True)
    response, connections = _submit({"bg__kepesites_ev": "1998", "bg__visszajelzes": "rendszeres_fogorvosi"}, background=background)
    assert response.status_code == 302
    bg = [params for c in connections for sql, params in c.executions if "status = 'submitted'" in sql][0][0].adapted
    assert bg["visszajelzes"] == "rendszeres_fogorvosi"


def test_form_asks_per_pole_ranges_and_magnitude_checkbox():
    row = response_row()
    app, _ = build_app([SCHEMA_OK, token_lookup(row)])
    client = app.test_client()
    auth_expert(client)
    page = client.get("/expert/urlap/tok").get_data(as_text=True)
    assert 'name="item__F1__siker_A_min"' in page and 'name="item__F1__siker_B_max"' in page
    assert 'name="item__F2__siker_M"' in page and 'name="item__F1__siker_M"' not in page
    assert 'name="item__F1__nagysag_nem_tudom"' in page and "húsz hasonló becslésből tizenkilencszer" in page
    assert "kb. 10 mm" in page and "kb. 125°" in page and "közepes eltérés (kb. 10°)" in page
    assert 'name="item__F1__kulonbseg_min"' not in page
    assert "valóban ez az irány igaz, és nem a fordítottja" in page


def test_preparation_gate_and_practice_page():
    row = response_row(calibration={})
    app, connections = build_app([SCHEMA_OK, token_lookup(row), token_lookup(row), token_lookup(row)])
    client = app.test_client()
    auth_expert(client)
    response = client.get("/expert/urlap/tok")
    assert response.status_code == 302 and response.headers["Location"].endswith("/expert/urlap/tok/felkeszites")
    page = client.get("/expert/urlap/tok/felkeszites").get_data(as_text=True)
    assert "Kétféle bizonytalanság" in page and 'name="cal__gyak_1_pont"' in page and 'name="cal__gyak_irany"' in page
    assert 'data-truth="243"' in page and "Semmelweis Ignác" in page and "Tovább a kérdőívre" in page
    response = client.post("/expert/urlap/tok/felkeszites/kesz", data={"csrf_token": "csrf-test"})
    assert response.status_code == 302 and response.headers["Location"].endswith("/expert/urlap/tok")
    updates = [params for c in connections for sql, params in c.executions if "calibration = COALESCE(calibration" in sql]
    assert updates and updates[0][0].adapted == {"felkeszites_kesz": "1"}


def test_practice_answers_autosave_into_calibration():
    row = response_row(calibration={})
    app, connections = build_app([SCHEMA_OK, token_lookup(row)])
    client = app.test_client()
    auth_expert(client)
    response = client.post("/expert/urlap/tok/mentes", data={"csrf_token": "csrf-test", "field": "cal__gyak_2_pont", "value": "45"})
    assert response.status_code == 200
    sql = [sql for c in connections for sql, params in c.executions if "SET calibration = calibration ||" in sql]
    assert sql


REF_COUNT_0 = ("ajanlo_kod' = %s", ["n"], [(0,)])
REF_COUNT_3 = ("ajanlo_kod' = %s", ["n"], [(3,)])
EMAIL_FREE = ("lower(invite_email)", ["id"], [])
EMAIL_TAKEN = ("lower(invite_email)", ["id"], [(12,)])


def auth_expert_with_token(client, token="tok"):
    with client.session_transaction() as session:
        session["expert_authenticated"] = True
        session["expert_csrf"] = "csrf-test"
        session["expert_token"] = token


def test_reminder_repeats_the_stored_deadline_in_hungarian():
    sent = []
    row = response_row(consent_confirmed=False, invited_at="2026-09-06", invite_email="elek@example.org",
                       invite_sent_at="2026-09-06 10:00", invite_deadline="2026-09-20", background={"nyelv": "hu"})
    app, _ = build_app([id_lookup(row)], mail_sender=lambda *args: sent.append(args))
    client = app.test_client()
    auth_admin(client)
    client.post("/expert/admin/7/email", data={"csrf_token": "csrf-test"})
    text = sent[0][3]
    assert "2026. szeptember 20-ig ki tudná tölteni" in text and "két héten belül" not in text
    assert "ajánlotta" not in text


def test_hungarian_invitation_states_two_weeks_and_the_date():
    sent = []
    app, _ = build_app([SCHEMA_OK, INSERT_OK, NEXT_CODE_OK], mail_sender=lambda *args: sent.append(args))
    client = app.test_client()
    auth_admin(client)
    client.post("/expert/admin/meghivo", data={"csrf_token": "csrf-test", "expert_name": "Dr. X", "invite_email": "x@example.org", "send_now": "on"})
    from expert_priors import deadline_iso, format_deadline
    assert f"két héten belül, {format_deadline(deadline_iso(), 'hu')}-ig ki tudná tölteni" in sent[0][3]


def test_done_page_offers_referral_to_the_submitting_expert():
    row = response_row(status="submitted", submitted_at="2026-09-06 12:00")
    app, _ = build_app([token_lookup(row), REF_COUNT_0])
    client = app.test_client()
    auth_expert_with_token(client)
    page = client.get("/expert/kesz/tok").get_data(as_text=True)
    assert 'action="/expert/urlap/tok/ajanlas"' in page and "Még 3 kollégát ajánlhat." in page and 'name="ref_email"' in page
    app, _ = build_app([token_lookup(row), REF_COUNT_3])
    client = app.test_client()
    auth_expert_with_token(client)
    page = client.get("/expert/kesz/tok").get_data(as_text=True)
    assert "Elérte az ajánlható kollégák számát" in page and 'name="ref_email"' not in page
    app, _ = build_app([token_lookup(row)])
    client = app.test_client()
    auth_expert_with_token(client, token="masik")
    page = client.get("/expert/kesz/tok").get_data(as_text=True)
    assert 'name="ref_email"' not in page


def test_referral_creates_invitation_with_referrer_and_sends_letter():
    sent = []
    row = response_row(status="submitted", submitted_at="2026-09-06 12:00")
    app, connections = build_app([token_lookup(row), REF_COUNT_0, EMAIL_FREE, INSERT_OK, NEXT_CODE_OK], mail_sender=lambda *args: sent.append(args))
    client = app.test_client()
    auth_expert_with_token(client)
    response = client.post("/expert/urlap/tok/ajanlas", data={
        "csrf_token": "csrf-test", "ref_name": "Kovács Fogtechnikus", "ref_email": "Kovacs@Example.org",
        "ref_role": "fogtechnikus", "ref_lang": "hu", "ref_include_name": "on",
    })
    assert response.status_code == 302 and response.headers["Location"].endswith("/expert/kesz/tok")
    inserts = [params for c in connections for sql, params in c.executions if "INSERT INTO expert_prior_responses" in sql]
    assert inserts[0][6].adapted == {"nyelv": "hu", "szerep": "fogtechnikus", "ajanlo_kod": "SZ07", "ajanlo_nev": "Dr. Teszt Elek"}
    assert inserts[0][9] == "kovacs@example.org" and inserts[0][7] is True and inserts[0][8] == "ajánlotta: SZ07"
    to_email, to_name, subject, text, _ = sent[0]
    assert to_email == "kovacs@example.org" and to_name == "Kovács Fogtechnikus"
    assert "fogtechnikus kollégáknak" in subject and "Erre a felmérésre Dr. Teszt Elek kolléga ajánlotta Önt." in text
    assert any("invite_deadline = COALESCE" in sql for sql in executed_sql(connections))
    with client.session_transaction() as session:
        flashes = session.get("_flashes", [])
    assert any("Meghívót küldtünk Kovács Fogtechnikus részére." in message for _, message in flashes)


def test_referral_refuses_duplicates_limits_and_foreign_sessions():
    row = response_row(status="submitted", submitted_at="2026-09-06 12:00")
    sent = []
    app, connections = build_app([token_lookup(row), REF_COUNT_0, EMAIL_TAKEN], mail_sender=lambda *args: sent.append(args))
    client = app.test_client()
    auth_expert_with_token(client)
    client.post("/expert/urlap/tok/ajanlas", data={"csrf_token": "csrf-test", "ref_name": "X", "ref_email": "x@example.org"})
    assert not sent and not any("INSERT INTO" in sql for sql in executed_sql(connections))
    with client.session_transaction() as session:
        assert any("már meghívtuk" in message for _, message in session.get("_flashes", []))
    app, connections = build_app([token_lookup(row), REF_COUNT_3, EMAIL_FREE], mail_sender=lambda *args: sent.append(args))
    client = app.test_client()
    auth_expert_with_token(client)
    client.post("/expert/urlap/tok/ajanlas", data={"csrf_token": "csrf-test", "ref_name": "X", "ref_email": "x@example.org"})
    assert not sent and not any("INSERT INTO" in sql for sql in executed_sql(connections))
    with client.session_transaction() as session:
        assert any("Legfeljebb 3 kollégát" in message for _, message in session.get("_flashes", []))
    app, _ = build_app([token_lookup(row)])
    client = app.test_client()
    auth_expert_with_token(client, token="masik")
    assert client.post("/expert/urlap/tok/ajanlas", data={"csrf_token": "csrf-test", "ref_name": "X", "ref_email": "x@example.org"}).status_code == 403
    app, connections = build_app([token_lookup(row), REF_COUNT_0])
    client = app.test_client()
    auth_expert_with_token(client)
    client.post("/expert/urlap/tok/ajanlas", data={"csrf_token": "csrf-test", "ref_name": "", "ref_email": "nem-email"})
    assert not any("INSERT INTO" in sql for sql in executed_sql(connections))


def test_bulk_invitations_create_rows_skip_bad_ones_and_send_letters():
    sent = []
    app, connections = build_app([SCHEMA_OK, EMAIL_FREE, INSERT_OK, NEXT_CODE_OK], mail_sender=lambda *args: sent.append(args))
    client = app.test_client()
    auth_admin(client)
    response = client.post("/expert/admin/meghivo/tomeges", data={
        "csrf_token": "csrf-test", "send_now": "on",
        "nev[]": ["Dr. Első Anna", "", "Második Béla", "Harmadik Cili", "Dr. Első Anna"],
        "munkahely[]": ["SE Fogpótlástani Klinika", "", "Labor Kft.", "", ""],
        "email[]": ["anna@example.org", "", "bela@example.org", "nem-email", "ANNA@example.org"],
        "szerep[]": ["fogorvos", "fogorvos", "fogtechnikus", "fogorvos", "fogorvos"],
        "nyelv[]": ["hu", "hu", "hu", "hu", "en"],
    })
    assert response.status_code == 302
    inserts = [params for c in connections for sql, params in c.executions if "INSERT INTO expert_prior_responses" in sql]
    assert len(inserts) == 2
    assert inserts[0][1] == "Dr. Első Anna" and inserts[0][2] == "SE Fogpótlástani Klinika" and inserts[0][9] == "anna@example.org"
    assert inserts[1][1] == "Második Béla" and inserts[1][6].adapted["szerep"] == "fogtechnikus"
    assert len(sent) == 2 and sent[1][0] == "bela@example.org" and "fogtechnikus kollégáknak" in sent[1][2]
    with client.session_transaction() as session:
        flashes = [message for _, message in session.get("_flashes", [])]
    assert any("2 meghívó elkészült, 2 levél elment" in m for m in flashes)
    assert any("4. sor" in m and "hibás az e-mail-cím" in m for m in flashes)
    assert any("5. sor" in m and "már meghívva" in m for m in flashes)


def test_bulk_invitations_skip_addresses_already_invited():
    sent = []
    app, connections = build_app([SCHEMA_OK, EMAIL_TAKEN, INSERT_OK, NEXT_CODE_OK], mail_sender=lambda *args: sent.append(args))
    client = app.test_client()
    auth_admin(client)
    client.post("/expert/admin/meghivo/tomeges", data={"csrf_token": "csrf-test", "send_now": "on", "nev[]": ["Dr. X"], "munkahely[]": [""], "email[]": ["x@example.org"], "szerep[]": ["fogorvos"], "nyelv[]": ["hu"]})
    assert not sent and not any("INSERT INTO" in sql for sql in executed_sql(connections))
    with client.session_transaction() as session:
        assert any("0 meghívó elkészült" in message for _, message in session.get("_flashes", []))


def test_admin_page_shows_bulk_invitation_table():
    app, _ = build_app([SCHEMA_OK, listing([])])
    client = app.test_client()
    auth_admin(client)
    page = client.get("/expert/admin").get_data(as_text=True)
    assert 'action="/expert/admin/meghivo/tomeges"' in page and page.count('<input type="text" name="nev[]"') == 5 and 'id="bulkPaste"' in page
    assert 'name="invite_deadline"' not in page


def test_no_difference_uses_one_common_number_and_ignores_stale_poles():
    # „nincs érdemi különbség”: a közös K szám és határai kellenek; a rejtett A/B mezők régi értékei nem számítanak
    response, _ = _submit({"item__F7__irany": "nincs_kulonbseg"})
    assert response.status_code == 400 and "kérjük a „száz beteg” számokat" in response.get_data(as_text=True)
    response, connections = _submit({"item__F7__irany": "nincs_kulonbseg", "item__F7__siker_K": "75", "item__F7__siker_K_min": "65", "item__F7__siker_K_max": "85"})
    assert response.status_code == 302
    items = [params for c in connections for sql, params in c.executions if "status = 'submitted'" in sql][0][2].adapted
    assert items["F7"]["siker_K"] == 75 and items["F7"]["siker_K_min"] == 65 and items["F7"]["irany"] == "nincs_kulonbseg"
    response, _ = _submit({"item__F7__irany": "nincs_kulonbseg", "item__F7__siker_K": "90", "item__F7__siker_K_min": "65", "item__F7__siker_K_max": "85"})
    assert response.status_code == 400 and "K változatnál a legvalószínűbb szám nincs" in response.get_data(as_text=True)


def test_cannot_judge_answer_needs_no_numbers():
    missing = {f"item__A3__{k}": "" for k in ("siker_A", "siker_A_min", "siker_A_max", "siker_B", "siker_B_min", "siker_B_max")}
    response, _ = _submit({**missing, "item__A3__irany": "nem_tudom"})
    assert response.status_code == 302
    response, _ = _submit({"item__A3__irany": "nem_tudom", "item__A3__siker_A": "20", "item__A3__siker_B": "90"})   # régi számok, figyelmen kívül
    assert response.status_code == 302


def test_form_renders_common_row_and_hides_numbers_for_cannot_judge():
    row = response_row(items={"F1": {"irany": "nincs_kulonbseg"}, "F3": {"irany": "nem_tudom"}, "F2": {"irany": "nem_monoton"}})
    app, _ = build_app([SCHEMA_OK, token_lookup(row)])
    client = app.test_client()
    auth_expert(client)
    page = client.get("/expert/urlap/tok").get_data(as_text=True)
    import re
    f1 = re.search(r'data-magnitude-for="F1".*?data-hint-for="F1"', page, re.S).group(0)
    assert 'class="pole-row" data-pole="K"' in f1 and 'class="pole-row is-disabled" data-pole="A"' in f1 and 'name="item__F1__siker_K"' in f1
    assert 'name="item__F1__siker_A" min="0" max="100" value="" data-autosave disabled' in f1
    f3 = re.search(r'<div class="expert-q magnitude-block[^"]*" data-magnitude-for="F3"[^>]*>', page).group(0)
    assert "is-disabled" in f3
    f2 = re.search(r'data-magnitude-for="F2".*?data-hint-for="F2"', page, re.S).group(0)
    assert 'class="pole-row" data-pole="M"' in f2 and 'class="pole-row is-disabled" data-pole="K"' in f2
    f4 = re.search(r'data-magnitude-for="F4".*?data-hint-for="F4"', page, re.S).group(0)
    assert 'class="pole-row is-disabled" data-pole="K"' in f4 and 'class="pole-row" data-pole="A"' in f4
    assert "Mindkét változatnál" in page and "„Nem tudom megítélni” választásnál számot nem kérünk." in page


def test_prior_export_carries_the_common_number():
    from expert_priors import PRIOR_CSV_COLUMNS, prior_rows
    rows = [response_row(status="submitted", items={"F7": {"irany": "nincs_kulonbseg", "siker_K": 75, "siker_K_min": 65, "siker_K_max": 85}})]
    assert PRIOR_CSV_COLUMNS[-3:] == ["siker_K", "siker_K_min", "siker_K_max"]
    f7 = [r for r in prior_rows(rows) if r["tetel"] == "F7"][0]
    assert f7["siker_K"] == 75 and f7["siker_K_max"] == 85 and f7["siker_A"] == ""


def test_a10_has_no_middle_option_and_rejects_it():
    from expert_priors import ITEMS_BY_CODE, parse_item_field, FieldError
    assert ITEMS_BY_CODE["A10"]["optimum"] is False and "M" not in ITEMS_BY_CODE["A10"]
    with pytest.raises(FieldError):
        parse_item_field("A10", "irany", "nem_monoton")
    row = response_row()
    app, _ = build_app([SCHEMA_OK, token_lookup(row)])
    client = app.test_client()
    auth_expert(client)
    page = client.get("/expert/urlap/tok").get_data(as_text=True)
    assert 'id="item__A10__irany_O"' not in page and 'id="item__F2__irany_O"' in page
    assert "feszes ínnyel nem fedett, plicaszerű, mozgékony tuberculum" in page and "fedetlen" not in page


def test_ranking_is_per_jaw_and_exported():
    from expert_priors import BACKGROUND_CSV_COLUMNS, CLOSING_FIELDS, LOWER_CODES, UPPER_CODES, background_rows
    assert UPPER_CODES == ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"] and "TUB" in LOWER_CODES and "A10" in LOWER_CODES
    assert CLOSING_FIELDS["rang_felso_1"][1] == set(UPPER_CODES) and CLOSING_FIELDS["rang_also_3"][1] == set(LOWER_CODES)
    row = response_row()
    app, _ = build_app([SCHEMA_OK, token_lookup(row)])
    client = app.test_client()
    auth_expert(client)
    page = client.get("/expert/urlap/tok").get_data(as_text=True)
    assert 'name="cl__rang_felso_1"' in page and 'name="cl__rang_also_3"' in page and 'name="cl__rang_1"' not in page
    assert "Felső állcsont: melyik három adottság" in page and "Alsó állcsont: melyik három adottság" in page
    response, connections = _submit({"cl__rang_felso_1": "F1", "cl__rang_felso_2": "F5", "cl__rang_also_1": "A1", "cl__rang_also_2": "TUB"})
    assert response.status_code == 302
    closing = [params for c in connections for sql, params in c.executions if "status = 'submitted'" in sql][0][3].adapted
    assert closing["rang_felso_1"] == "F1" and closing["rang_also_2"] == "TUB"
    bad, _ = _submit({"cl__rang_felso_1": "A1"})   # alsó tétel a felső listában
    assert bad.status_code == 400
    exported = background_rows([response_row(status="submitted", closing={"rang_felso_1": "F1", "rang_also_1": "A1", "rang_1": "F5"})])[0]
    assert exported["rang_felso_1"] == "F1" and exported["rang_also_1"] == "A1" and exported["rang_1"] == "F5"
    assert "rang_also_3" in BACKGROUND_CSV_COLUMNS
