"""Protected, staged data-collection workflow for PREDICT baseline visits."""

from __future__ import annotations

import math
import os
import re
import secrets
import uuid
from datetime import date, datetime
from functools import wraps

from flask import (
    Blueprint,
    abort,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from psycopg2.extras import Json
from werkzeug.utils import secure_filename


SITUATION_VALUES = {"Kiváló", "Jó", "Átlagos", "Rossz", "Nagyon rossz"}
QUESTIONNAIRE_FIELDS = [
    "responsiveness_today_situation",
    "chewing_today_situation",
    *[f"OHIP_{i}" for i in range(1, 6)],
    *[f"GOHAI_{i}" for i in range(1, 13)],
]
UPPER_ANATOMY_FIELDS = ["F5", "F7", "F8"]
LOWER_ANATOMY_FIELDS = [
    "A1_Kaan",
    *[f"A{number}_{side}" for number in range(3, 10) for side in ("jobb", "bal")],
    "A11",
    "A12",
    "A13",
    "A14",
]
UPPER_MODEL_FIELDS = ["F1", "F2", "F3", "F4", "F6"]
LOWER_MODEL_FIELDS = ["A10", "A2_gerincelvonal", "A2_bukkalisathajlas", "A2_lingualisathajlas"]


def create_baseline_blueprint(
    connection_factory,
    hue_calculator,
    nas_uploader,
    upload_folder,
    allowed_file,
):
    bp = Blueprint("baseline", __name__, url_prefix="/baseline")

    def csrf_token():
        token = session.get("followup_csrf")
        if not token:
            token = secrets.token_urlsafe(32)
            session["followup_csrf"] = token
        return token

    def validate_csrf():
        expected = session.get("followup_csrf", "")
        supplied = request.form.get("csrf_token", "")
        if not expected or not secrets.compare_digest(expected, supplied):
            abort(400, description="Érvénytelen vagy lejárt űrlap. Töltsd újra az oldalt.")

    @bp.context_processor
    def inject_helpers():
        return {"baseline_csrf_token": csrf_token}

    @bp.after_request
    def protect_health_data(response):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    def require_access(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if not os.getenv("FOLLOWUP_ACCESS_CODE"):
                return render_template("baseline_setup.html"), 503
            if not session.get("followup_authenticated"):
                return redirect(url_for("followup.login", next=request.full_path))
            return view(*args, **kwargs)

        return wrapped

    def rows(sql, params=()):
        conn = connection_factory()
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
                columns = [description[0] for description in cursor.description]
                return [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            conn.close()

    def execute(sql, params=()):
        conn = connection_factory()
        try:
            with conn.cursor() as cursor:
                cursor.execute(sql, params)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def schema_ready():
        found = rows(
            """
            SELECT
                to_regclass('public.baseline_visits') AS visits_table,
                EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'patients'
                      AND column_name = 'paciens_telefonszam'
                ) AS phone_column
            """
        )
        return bool(found and found[0]["visits_table"] and found[0]["phone_column"])

    def require_schema():
        if not schema_ready():
            return render_template("baseline_setup.html", migration_missing=True), 503
        return None

    def study_code(patient_id):
        return f"PRED-{int(patient_id):04d}"

    def normalize_taj(raw):
        digits = "".join(character for character in str(raw or "") if character.isdigit())
        if len(digits) != 9:
            return None
        return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"

    def valid_taj_checksum(formatted_taj):
        digits = [int(character) for character in formatted_taj if character.isdigit()]
        weighted = sum(value * (3 if index % 2 == 0 else 7) for index, value in enumerate(digits[:8]))
        return weighted % 10 == digits[8]

    def required_anatomy(denture_type):
        fields = []
        if denture_type in {"upper", "both"}:
            fields.extend(UPPER_ANATOMY_FIELDS)
        if denture_type in {"lower", "both"}:
            fields.extend(LOWER_ANATOMY_FIELDS)
        return fields

    def required_model(denture_type):
        fields = []
        if denture_type in {"upper", "both"}:
            fields.extend(UPPER_MODEL_FIELDS)
        if denture_type in {"lower", "both"}:
            fields.extend(LOWER_MODEL_FIELDS)
        return fields

    def present(value):
        return value is not None and str(value).strip() != ""

    def decorate_visit(record):
        record = dict(record)
        record["study_code"] = study_code(record["patient_id"])
        questionnaire = record.get("questionnaire_data") or {}
        anatomy = record.get("anatomy_data") or {}
        model = record.get("model_data") or {}
        record["questionnaire_complete"] = all(present(questionnaire.get(field)) for field in QUESTIONNAIRE_FIELDS)
        record["anatomy_complete"] = all(present(anatomy.get(field)) for field in required_anatomy(record["denture_type"]))
        record["model_complete"] = all(present(model.get(field)) for field in required_model(record["denture_type"]))
        record["mai_complete"] = present(record.get("init_mai_huedegree")) and present(record.get("init_image_path"))
        record["ready"] = (
            bool(record.get("consent_confirmed"))
            and record["questionnaire_complete"]
            and record["anatomy_complete"]
            and record["model_complete"]
            and record["mai_complete"]
        )
        return record

    visit_query = """
        SELECT
            b.*,
            p."TAJ" AS taj,
            p."paciens_neve" AS patient_name,
            p."paciens_telefonszam" AS patient_phone,
            p."birthdate" AS birthdate,
            p."gender" AS gender,
            p."denture_type" AS denture_type
        FROM baseline_visits b
        JOIN patients p ON p."id" = b.patient_id
    """

    def get_visit(patient_id):
        found = rows(visit_query + " WHERE b.patient_id = %s", (patient_id,))
        if not found:
            abort(404)
        return decorate_visit(found[0])

    def ensure_editable(visit):
        if visit.get("status") == "completed":
            abort(409, description="A lezárt kezdővizit nem módosítható.")

    def save_draft(patient_id, column, values, completed_column):
        if column not in {"questionnaire_data", "anatomy_data", "model_data"}:
            raise ValueError("Nem engedélyezett piszkozatmező.")
        if completed_column not in {
            "questionnaire_completed_at",
            "anatomy_completed_at",
            "model_completed_at",
        }:
            raise ValueError("Nem engedélyezett időbélyegmező.")
        execute(
            f"""
            UPDATE baseline_visits
            SET {column} = %s,
                {completed_column} = CURRENT_TIMESTAMP,
                status = 'in_progress',
                updated_at = CURRENT_TIMESTAMP
            WHERE patient_id = %s AND status <> 'completed'
            """,
            (Json(values), patient_id),
        )

    @bp.get("")
    @require_access
    def dashboard():
        setup_response = require_schema()
        if setup_response:
            return setup_response
        all_visits = [decorate_visit(record) for record in rows(visit_query + " ORDER BY b.created_at DESC")]
        visits = list(all_visits)
        query = request.args.get("q", "").strip().lower()
        if query:
            digits = "".join(character for character in query if character.isdigit())
            visits = [
                visit
                for visit in visits
                if query in str(visit.get("patient_name") or "").lower()
                or query in visit["study_code"].lower()
                or (digits and digits in "".join(c for c in str(visit.get("taj") or "") if c.isdigit()))
                or (digits and digits in "".join(c for c in str(visit.get("patient_phone") or "") if c.isdigit()))
            ]
        stats = {
            "total": len(all_visits),
            "in_progress": sum(visit["status"] != "completed" for visit in all_visits),
            "ready": sum(visit["ready"] and visit["status"] != "completed" for visit in all_visits),
            "completed": sum(visit["status"] == "completed" for visit in all_visits),
        }
        return render_template("baseline_dashboard.html", visits=visits, stats=stats, query=query)

    @bp.route("/new", methods=["GET", "POST"])
    @require_access
    def register_patient():
        setup_response = require_schema()
        if setup_response:
            return setup_response
        if request.method == "POST":
            validate_csrf()
            taj = normalize_taj(request.form.get("taj"))
            if not taj:
                abort(400, description="A TAJ formátuma 000-000-000 legyen.")
            if not valid_taj_checksum(taj):
                abort(400, description="A TAJ ellenőrzőszáma hibás. Ellenőrizd a begépelt számot.")
            patient_name = request.form.get("patient_name", "").strip()
            phone = request.form.get("patient_phone", "").strip()
            collector = request.form.get("data_collector", "").strip()
            if not patient_name:
                abort(400, description="A beteg neve kötelező.")
            if phone and not re.fullmatch(r"[0-9+()\-\s/.]{6,30}", phone):
                abort(400, description="A telefonszám formátuma nem érvényes.")
            if not collector:
                abort(400, description="A kitöltő monogramja kötelező.")
            try:
                birthdate = datetime.strptime(request.form.get("birthdate", ""), "%Y-%m-%d").date()
                if birthdate > date.today() or birthdate.year < 1900:
                    raise ValueError
            except ValueError:
                abort(400, description="A születési dátum nem érvényes.")
            gender = request.form.get("gender")
            denture_type = request.form.get("denture_type")
            if gender not in {"Male", "Female"}:
                abort(400, description="A nem kiválasztása kötelező.")
            if denture_type not in {"lower", "upper", "both"}:
                abort(400, description="A készítendő fogsor kiválasztása kötelező.")
            if request.form.get("consent_confirmed") != "1":
                abort(400, description="A dokumentált beleegyezést meg kell erősíteni.")

            conn = connection_factory()
            try:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (taj,))
                    cursor.execute(
                        """
                        SELECT "id" FROM patients
                        WHERE regexp_replace(COALESCE("TAJ", ''), '[^0-9]', '', 'g') = %s
                        LIMIT 1
                        """,
                        (taj.replace("-", ""),),
                    )
                    existing = cursor.fetchone()
                    if existing:
                        conn.rollback()
                        flash("Ez a TAJ már szerepel az adatbázisban; új kezdővizit nem nyitható hozzá.", "error")
                        return redirect(url_for("baseline.register_patient"))
                    cursor.execute(
                        """
                        INSERT INTO patients (
                            "TAJ", "record_datetime", "birthdate", "gender", "denture_type",
                            "paciens_neve", "paciens_telefonszam", "data_uploader"
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        RETURNING "id"
                        """,
                        (taj, datetime.now(), birthdate, gender, denture_type, patient_name, phone or None, collector),
                    )
                    patient_id = cursor.fetchone()[0]
                    cursor.execute(
                        """
                        INSERT INTO baseline_visits (patient_id, consent_confirmed, data_collector)
                        VALUES (%s, TRUE, %s)
                        """,
                        (patient_id, collector),
                    )
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
            flash("A beteg rögzítve. Folytasd a kiindulási kérdőívvel.", "success")
            return redirect(url_for("baseline.patient", patient_id=patient_id))
        return render_template("baseline_register.html")

    @bp.get("/patient/<int:patient_id>")
    @require_access
    def patient(patient_id):
        setup_response = require_schema()
        if setup_response:
            return setup_response
        return render_template("baseline_visit.html", visit=get_visit(patient_id))

    @bp.route("/patient/<int:patient_id>/questionnaire", methods=["GET", "POST"])
    @require_access
    def questionnaire(patient_id):
        visit = get_visit(patient_id)
        ensure_editable(visit)
        if request.method == "POST":
            validate_csrf()
            values = {}
            for field in ("responsiveness_today_situation", "chewing_today_situation"):
                value = request.form.get(field, "")
                if value not in SITUATION_VALUES:
                    abort(400, description="Minden általános kérdés megválaszolása kötelező.")
                values[field] = value
            for index in range(1, 6):
                value = request.form.get(f"OHIP_{index}", "")
                if value not in {"0", "1", "2", "3", "4"}:
                    abort(400, description="Minden OHIP-kérdés megválaszolása kötelező.")
                values[f"OHIP_{index}"] = int(value)
            for index in range(1, 13):
                value = request.form.get(f"GOHAI_{index}", "")
                if value not in {"1", "2", "3", "4", "5"}:
                    abort(400, description="Minden GOHAI-kérdés megválaszolása kötelező.")
                values[f"GOHAI_{index}"] = int(value)
            save_draft(patient_id, "questionnaire_data", values, "questionnaire_completed_at")
            flash("A 19 kérdésből álló kiindulási kérdőív hiánytalanul elmentve.", "success")
            return redirect(url_for("baseline.patient", patient_id=patient_id))
        return render_template(
            "baseline_questionnaire.html",
            visit=visit,
            form_values=visit.get("questionnaire_data") or {},
        )

    @bp.route("/patient/<int:patient_id>/anatomy", methods=["GET", "POST"])
    @require_access
    def anatomy(patient_id):
        visit = get_visit(patient_id)
        ensure_editable(visit)
        fields = required_anatomy(visit["denture_type"])
        if request.method == "POST":
            validate_csrf()
            values = {}
            for field in fields:
                value = request.form.get(field, "")
                allowed = {"1", "2", "3", "4", "5"} if field == "A1_Kaan" else {"1", "2", "3"}
                if value not in allowed:
                    abort(400, description="Minden megjelenő anatómiai képlet értékelése kötelező.")
                values[field] = int(value)
            save_draft(patient_id, "anatomy_data", values, "anatomy_completed_at")
            flash("A klinikai anatómiai vizsgálat hiánytalanul elmentve.", "success")
            return redirect(url_for("baseline.patient", patient_id=patient_id))
        return render_template(
            "baseline_anatomy.html",
            visit=visit,
            form_values=visit.get("anatomy_data") or {},
        )

    @bp.route("/patient/<int:patient_id>/model", methods=["GET", "POST"])
    @require_access
    def model(patient_id):
        visit = get_visit(patient_id)
        ensure_editable(visit)
        current = dict(visit.get("model_data") or {})
        if request.method == "POST":
            validate_csrf()
            values = dict(current)
            numeric_fields = []
            if visit["denture_type"] in {"upper", "both"}:
                numeric_fields.extend(UPPER_MODEL_FIELDS)
            if visit["denture_type"] in {"lower", "both"}:
                numeric_fields.append("A10")
            for field in numeric_fields:
                raw = request.form.get(field, "").strip().replace(",", ".")
                try:
                    number = float(raw)
                    if not math.isfinite(number):
                        raise ValueError
                    if field in {"F1", "F2", "F3"} and number < 0:
                        raise ValueError
                    if field in {"F4", "F6", "A10"} and not -180 <= number <= 180:
                        raise ValueError
                except ValueError:
                    abort(400, description=f"A(z) {field} mérési értéke nem érvényes.")
                values[field] = number

            if visit["denture_type"] in {"lower", "both"}:
                upload_fields = {
                    "A2_gerincelvonal": ("stl_gerincelvonal", "A2_gerinc"),
                    "A2_bukkalisathajlas": ("stl_bukkalis", "A2_bukkalis"),
                    "A2_lingualisathajlas": ("stl_lingualis", "A2_lingualis"),
                }
                for target, (form_name, measurement_type) in upload_fields.items():
                    uploaded = request.files.get(form_name)
                    if uploaded and uploaded.filename:
                        safe_name = secure_filename(uploaded.filename)
                        if "." not in safe_name or safe_name.rsplit(".", 1)[1].lower() != "stl":
                            abort(400, description="Az A2 modellfájlok formátuma STL legyen.")
                        temp_name = f"baseline_model_{patient_id}_{uuid.uuid4().hex}_{safe_name}"
                        temp_path = os.path.join(upload_folder, temp_name)
                        uploaded.save(temp_path)
                        try:
                            values[target] = nas_uploader(temp_path, visit["study_code"], measurement_type)
                        finally:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                    if not present(values.get(target)):
                        abort(400, description="Mindhárom A2 STL-fájl feltöltése kötelező.")

            save_draft(patient_id, "model_data", values, "model_completed_at")
            flash("A modellanalízis adatai elmentve.", "success")
            return redirect(url_for("baseline.patient", patient_id=patient_id))
        return render_template("baseline_model.html", visit=visit, form_values=current)

    @bp.post("/patient/<int:patient_id>/mai")
    @require_access
    def save_mai(patient_id):
        validate_csrf()
        visit = get_visit(patient_id)
        ensure_editable(visit)
        image = request.files.get("image")
        if not image or not image.filename or not allowed_file(image.filename):
            abort(400, description="TIFF kép feltöltése szükséges.")
        safe_name = secure_filename(image.filename)
        temp_name = f"baseline_mai_{patient_id}_{uuid.uuid4().hex}_{safe_name}"
        temp_path = os.path.join(upload_folder, temp_name)
        image.save(temp_path)
        try:
            hue_degree = hue_calculator(temp_path)
            if hue_degree is None or not math.isfinite(float(hue_degree)):
                abort(400, description="A képből nem számítható érvényes MAI hue-degree.")
            image_path = nas_uploader(temp_path, visit["study_code"], "mai_initial")
            execute(
                """
                UPDATE baseline_visits
                SET init_mai_huedegree = %s,
                    init_image_path = %s,
                    mai_note = %s,
                    mai_completed_at = CURRENT_TIMESTAMP,
                    status = 'in_progress',
                    updated_at = CURRENT_TIMESTAMP
                WHERE patient_id = %s AND status <> 'completed'
                """,
                (hue_degree, image_path, request.form.get("mai_note", "").strip() or None, patient_id),
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        flash("A kiindulási MAI-kép és a hue-degree elmentve.", "success")
        return redirect(url_for("baseline.patient", patient_id=patient_id))

    @bp.post("/patient/<int:patient_id>/complete")
    @require_access
    def complete_visit(patient_id):
        validate_csrf()
        visit = get_visit(patient_id)
        ensure_editable(visit)
        missing = []
        if not visit.get("consent_confirmed"):
            missing.append("dokumentált beleegyezés")
        if not visit["questionnaire_complete"]:
            missing.append("kiindulási kérdőív")
        if not visit["anatomy_complete"]:
            missing.append("klinikai anatómia")
        if not visit["model_complete"]:
            missing.append("modellanalízis")
        if not visit["mai_complete"]:
            missing.append("kiindulási MAI")
        if missing:
            flash("A kezdővizit még nem zárható le. Hiányzik: " + ", ".join(missing) + ".", "error")
            return redirect(url_for("baseline.patient", patient_id=patient_id))

        questionnaire = visit["questionnaire_data"]
        anatomy_data = visit["anatomy_data"]
        model_data = visit["model_data"]
        patient_values = {
            **questionnaire,
            **anatomy_data,
            **{field: model_data.get(field) for field in UPPER_MODEL_FIELDS if field in model_data},
            "A10": model_data.get("A10"),
            "A2_gerincelvonal": model_data.get("A2_gerincelvonal"),
            "A2_bukkalisathajlas": model_data.get("A2_bukkalisathajlas"),
            "A2_lingualisathajlas": model_data.get("A2_lingualisathajlas"),
            "modellanalizis_megtortent": True,
            "init_mai_huedegree": visit["init_mai_huedegree"],
            "init_image_path": visit["init_image_path"],
        }
        allowed_columns = set(QUESTIONNAIRE_FIELDS + UPPER_ANATOMY_FIELDS + LOWER_ANATOMY_FIELDS + UPPER_MODEL_FIELDS + LOWER_MODEL_FIELDS + ["modellanalizis_megtortent", "init_mai_huedegree", "init_image_path"])
        patient_values = {key: value for key, value in patient_values.items() if key in allowed_columns and value is not None}
        assignments = ", ".join(f'"{column}" = %s' for column in patient_values)

        conn = connection_factory()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT status FROM baseline_visits WHERE patient_id = %s FOR UPDATE", (patient_id,))
                locked = cursor.fetchone()
                if not locked or locked[0] == "completed":
                    abort(409, description="A kezdővizit már le van zárva.")
                cursor.execute(
                    f'UPDATE patients SET {assignments} WHERE "id" = %s',
                    [*patient_values.values(), patient_id],
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("A betegrekord nem található.")
                cursor.execute(
                    """
                    UPDATE baseline_visits
                    SET status = 'completed', completed_at = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE patient_id = %s
                    """,
                    (patient_id,),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        flash("A kezdővizit hiánytalanul lezárva; az adatok bekerültek az elemzési rekordba.", "success")
        return redirect(url_for("baseline.patient", patient_id=patient_id))

    return bp
