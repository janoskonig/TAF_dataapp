"""Protected, separate data-collection workflow for PREDICT follow-up visits."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import re
import secrets
import uuid
from datetime import datetime
from functools import wraps

from flask import (
    Blueprint,
    Response,
    after_this_request,
    abort,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    send_file,
    url_for,
)
from werkzeug.utils import secure_filename
from urllib.parse import urlsplit


OHIP_FIELDS = [f"ohip_{i}_recall" for i in range(1, 6)]
GOHAI_FIELDS = [f"gohai_{i}_recall" for i in range(1, 13)]
ANCHOR_FIELDS = [
    "responsiveness_today_situation_recall",
    "responsiveness_change",
    "chewing_today_situation_recall",
    "chewing_change",
]
QUESTIONNAIRE_FIELDS = ANCHOR_FIELDS + OHIP_FIELDS + GOHAI_FIELDS

SITUATION_VALUES = {"Kiváló", "Jó", "Átlagos", "Rossz", "Nagyon rossz"}
CHANGE_VALUES = {
    "Sokat romlott",
    "Kicsit romlott",
    "Változatlan maradt",
    "Kicsit javult",
    "Sokat javult",
}
VISIT_STATUSES = {
    "not_contacted",
    "contacted",
    "scheduled",
    "arrived",
    "completed",
    "declined",
    "no_show",
    "unreachable",
}
VISIT_STATUS_LABELS = {
    "not_contacted": "Még nem kerestük",
    "contacted": "Elértük",
    "scheduled": "Időpont egyeztetve",
    "arrived": "Megjelent",
    "completed": "Vizit lezárva",
    "declined": "Visszautasította",
    "no_show": "Nem jelent meg",
    "unreachable": "Nem elérhető",
}


def normalise_taj(value):
    """Return the last nine TAJ digits, preserving leading zeroes."""
    digits = "".join(character for character in str(value or "") if character.isdigit())
    return digits[-9:].zfill(9) if digits else ""


def create_followup_blueprint(
    connection_factory,
    hue_calculator,
    nas_uploader,
    model_stl_inventory_loader,
    model_file_lister,
    model_file_uploader,
    model_file_downloader,
    upload_folder,
    allowed_file,
):
    bp = Blueprint("followup", __name__, url_prefix="/followup")

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
        return {
            "followup_csrf_token": csrf_token,
            "visit_status_labels": VISIT_STATUS_LABELS,
        }

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
            access_code = os.getenv("FOLLOWUP_ACCESS_CODE")
            if not access_code:
                return render_template("followup_setup.html"), 503
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

    def execute_transaction(statements):
        conn = connection_factory()
        try:
            with conn.cursor() as cursor:
                for sql, params in statements:
                    cursor.execute(sql, params)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def schema_ready():
        result = rows(
            """
            SELECT
                to_regclass('public.followup_visits') AS visits_table,
                to_regclass('public.followup_contact_attempts') AS contacts_table
            """
        )
        return bool(result and result[0]["visits_table"] and result[0]["contacts_table"])

    def require_schema():
        if not schema_ready():
            return render_template("followup_setup.html", migration_missing=True), 503
        return None

    def study_code(patient_id):
        return f"PRED-{int(patient_id):04d}"

    def mask_taj(value):
        digits = "".join(character for character in str(value or "") if character.isdigit())
        return f"***-***-{digits[-3:]}" if len(digits) >= 3 else "***-***-***"

    def present(value):
        return value is not None and str(value).strip() != ""

    def questionnaire_complete(record, prefix):
        return all(present(record.get(f"{prefix}{field}")) for field in QUESTIONNAIRE_FIELDS)

    def decorate_patient(record):
        record = dict(record)
        record["study_code"] = study_code(record["patient_id"])
        record["masked_taj"] = mask_taj(record.get("taj"))
        phone = str(record.get("patient_phone") or "").strip()
        phone_digits = "".join(character for character in phone if character.isdigit())
        record["phone_href"] = ("+" if phone.startswith("+") else "") + phone_digits if phone_digits else None
        new_complete = bool(record.get("questionnaire_completed_at")) and questionnaire_complete(
            record, "new_"
        )
        legacy_complete = questionnaire_complete(record, "legacy_")
        record["questionnaire_complete"] = new_complete or legacy_complete
        record["questionnaire_source"] = "new" if new_complete else "legacy" if legacy_complete else None
        record["f9_value"] = record.get("new_f9") or record.get("legacy_f9")
        record["f9_complete"] = present(record["f9_value"])
        record["mai_eligible"] = present(record.get("init_mai_huedegree"))
        record["mai_complete"] = present(record.get("new_final_mai_huedegree")) or present(
            record.get("legacy_final_mai_huedegree")
        )
        record["model_analysis_completed"] = bool(
            record.get("model_analysis_completed")
        )
        record["consent_confirmed"] = bool(record.get("consent_confirmed"))
        record["primary_ready"] = (
            record["questionnaire_complete"]
            and record["f9_complete"]
            and record["consent_confirmed"]
        )
        record["fully_ready"] = record["primary_ready"] and (
            record["mai_complete"] or not record["mai_eligible"]
        )
        return record

    def add_model_stl_status(records):
        inventory, source_available = model_stl_inventory_loader()
        decorated = []
        for record in records:
            record = dict(record)
            count = max(
                inventory.get(normalise_taj(record.get("taj")), 0),
                inventory.get(record["study_code"].upper(), 0),
            )
            record["model_stl_count"] = count if source_available else None
            record["model_stl_status"] = (
                "available" if count else "missing" if source_available else "unknown"
            )
            record["model_stl_source_available"] = source_available
            decorated.append(record)
        return decorated

    def model_file_keys(patient_record):
        return [normalise_taj(patient_record.get("taj")), patient_record["study_code"]]

    def format_file_size(size):
        size = int(size or 0)
        units = ("B", "KB", "MB", "GB")
        value = float(size)
        for unit in units:
            if value < 1024 or unit == units[-1]:
                decimals = 0 if unit == "B" else 1
                return f"{value:.{decimals}f} {unit}"
            value /= 1024

    def decorated_model_files(patient_record):
        files, source_available = model_file_lister(model_file_keys(patient_record))
        decorated = []
        for item in files:
            item = dict(item)
            item["file_id"] = hashlib.sha256(item["name"].encode("utf-8")).hexdigest()[:20]
            item["size_label"] = format_file_size(item.get("size"))
            decorated.append(item)
        return decorated, source_available

    legacy_select = ",\n".join(
        [
            'c."responsiveness_today_situation_recall" AS legacy_responsiveness_today_situation_recall',
            'c."responsiveness_change" AS legacy_responsiveness_change',
            'c."chewing_today_situation_recall" AS legacy_chewing_today_situation_recall',
            'c."chewing_change" AS legacy_chewing_change',
        ]
        + [f'c."OHIP_{i}_recall" AS legacy_ohip_{i}_recall' for i in range(1, 6)]
        + [f'c."GOHAI_{i}_recall" AS legacy_gohai_{i}_recall' for i in range(1, 13)]
    )
    new_select = ",\n".join([f"f.{field} AS new_{field}" for field in QUESTIONNAIRE_FIELDS])

    cohort_query = f"""
        WITH latest AS (
            SELECT DISTINCT ON ("TAJ") *
            FROM patients
            WHERE "TAJ" IS NOT NULL
            ORDER BY "TAJ", "id" DESC
        ), cohort AS (
            SELECT * FROM latest
            WHERE LOWER(TRIM("denture_type")) = 'both'
              AND "OHIP_1" IS NOT NULL
        )
        SELECT
            c."id" AS patient_id,
            c."TAJ" AS taj,
            COALESCE(
                NULLIF(TRIM(f.patient_display_name), ''),
                NULLIF(TRIM(c."paciens_neve"), '')
            ) AS patient_name,
            COALESCE(
                NULLIF(TRIM(f.patient_phone), ''),
                NULLIF(TRIM(c."paciens_telefonszam"), '')
            ) AS patient_phone,
            c."gender" AS gender,
            c."birthdate" AS birthdate,
            c."init_mai_huedegree" AS init_mai_huedegree,
            c."final_mai_huedegree" AS legacy_final_mai_huedegree,
            c."F9" AS legacy_f9,
            c."modellanalizis_megtortent" AS model_analysis_completed,
            {legacy_select},
            {new_select},
            f.visit_status,
            f.contact_attempted_at,
            f.appointment_at,
            f.contact_note,
            f.nonattendance_reason,
            f.consent_confirmed,
            f.data_collector,
            f.months_since_delivery,
            f.interim_adjustments,
            f.interim_events,
            f.f9 AS new_f9,
            f.final_mai_huedegree AS new_final_mai_huedegree,
            f.final_image_path,
            f.questionnaire_completed_at,
            f.mai_completed_at,
            f.completed_at
        FROM cohort c
        LEFT JOIN followup_visits f
          ON f.patient_id = c."id" AND f.visit_round = 1
    """

    def get_cohort():
        cohort = [
            decorate_patient(record)
            for record in rows(cohort_query + ' ORDER BY f.appointment_at NULLS LAST, c."id"')
        ]
        return add_model_stl_status(cohort)

    def get_patient(patient_id):
        found = rows(cohort_query + ' WHERE c."id" = %s', (patient_id,))
        if not found:
            abort(404)
        return add_model_stl_status([decorate_patient(found[0])])[0]

    def upsert_statement(patient_id, values):
        allowed = {
            "visit_status",
            "contact_attempted_at",
            "appointment_at",
            "patient_display_name",
            "patient_phone",
            "contact_note",
            "nonattendance_reason",
            "consent_confirmed",
            "data_collector",
            "months_since_delivery",
            "interim_adjustments",
            "interim_events",
            "f9",
            "questionnaire_completed_at",
            "final_mai",
            "final_mai_huedegree",
            "final_image_path",
            "mai_completed_at",
            "mai_note",
            "completed_at",
            *QUESTIONNAIRE_FIELDS,
        }
        if not values or not set(values).issubset(allowed):
            raise ValueError("Nem engedélyezett utánkövetési mező.")
        columns = ["patient_id", "visit_round", *values.keys()]
        placeholders = ", ".join(["%s"] * len(columns))
        updates = ", ".join(
            [
                (
                    f"{column} = followup_visits.{column} OR EXCLUDED.{column}"
                    if column == "consent_confirmed"
                    else (
                        f"{column} = COALESCE(EXCLUDED.{column}, "
                        f"followup_visits.{column})"
                    )
                )
                for column in values
            ]
        )
        sql = f"""
            INSERT INTO followup_visits ({', '.join(columns)})
            VALUES ({placeholders})
            ON CONFLICT (patient_id, visit_round) DO UPDATE SET
                {updates}, updated_at = CURRENT_TIMESTAMP
        """
        return sql, [patient_id, 1, *values.values()]

    def upsert_fields(patient_id, values):
        execute_transaction([upsert_statement(patient_id, values)])

    @bp.route("/login", methods=["GET", "POST"])
    def login():
        next_url = request.values.get("next", "")
        parsed_next = urlsplit(next_url)
        if not next_url.startswith("/") or parsed_next.netloc or next_url.startswith("//"):
            next_url = ""
        if request.method == "POST":
            validate_csrf()
            configured = os.getenv("FOLLOWUP_ACCESS_CODE")
            supplied = request.form.get("access_code", "")
            if configured and secrets.compare_digest(configured, supplied):
                session["followup_authenticated"] = True
                return redirect(next_url or url_for("followup.dashboard"))
            flash("Hibás hozzáférési kód.", "error")
        return render_template("followup_login.html", next_url=next_url)

    @bp.post("/logout")
    def logout():
        validate_csrf()
        session.pop("followup_authenticated", None)
        return redirect(url_for("followup.login"))

    @bp.get("")
    @require_access
    def dashboard():
        setup_response = require_schema()
        if setup_response:
            return setup_response
        all_patients = get_cohort()
        patients = list(all_patients)
        query = request.args.get("q", "").strip().lower()
        status_filter = request.args.get("status", "all")
        if query:
            query_digits = "".join(character for character in query if character.isdigit())
            patients = [
                patient
                for patient in patients
                if query in patient["study_code"].lower()
                or query in str(patient.get("patient_name") or "").lower()
                or (query_digits and query_digits in "".join(c for c in str(patient["taj"]) if c.isdigit()))
                or (
                    query_digits
                    and query_digits
                    in "".join(c for c in str(patient.get("patient_phone") or "") if c.isdigit())
                )
            ]
        if status_filter == "scheduled":
            patients = [patient for patient in patients if patient.get("visit_status") == "scheduled"]
        elif status_filter == "missing_questionnaire":
            patients = [patient for patient in patients if not patient["questionnaire_complete"]]
        elif status_filter == "missing_mai":
            patients = [patient for patient in patients if patient["mai_eligible"] and not patient["mai_complete"]]
        elif status_filter == "missing_stl":
            patients = [patient for patient in patients if patient["model_stl_status"] == "missing"]
        elif status_filter == "model_completed":
            patients = [patient for patient in patients if patient["model_analysis_completed"]]
        elif status_filter == "model_pending":
            patients = [
                patient
                for patient in patients
                if patient["model_stl_status"] == "available"
                and not patient["model_analysis_completed"]
            ]
        elif status_filter == "ready":
            patients = [patient for patient in patients if patient["fully_ready"]]

        stats = {
            "total": len(all_patients),
            "scheduled": sum(patient.get("visit_status") == "scheduled" for patient in all_patients),
            "questionnaire": sum(patient["questionnaire_complete"] for patient in all_patients),
            "mai": sum(patient["mai_complete"] for patient in all_patients),
            "model_stl": sum(
                patient["model_stl_status"] == "available" for patient in all_patients
            ),
            "model_stl_source_available": bool(
                all_patients and all_patients[0]["model_stl_source_available"]
            ),
            "model_completed": sum(
                patient["model_analysis_completed"] for patient in all_patients
            ),
            "primary_ready": sum(patient["primary_ready"] for patient in all_patients),
            "fully_ready": sum(patient["fully_ready"] for patient in all_patients),
        }
        return render_template(
            "followup_dashboard.html",
            patients=patients,
            stats=stats,
            query=query,
            status_filter=status_filter,
        )

    @bp.get("/blender-addon")
    @require_access
    def blender_addon():
        addon_path = os.path.join(current_app.root_path, "addon", "taf_addon.py")
        if not os.path.isfile(addon_path):
            abort(404)
        return render_template(
            "followup_blender_addon.html",
            addon_version="2.3.0",
            blend_upload_max_mb=current_app.config.get("BLEND_UPLOAD_MAX_MB"),
        )

    @bp.get("/blender-addon/download")
    @require_access
    def download_blender_addon():
        addon_path = os.path.join(current_app.root_path, "addon", "taf_addon.py")
        if not os.path.isfile(addon_path):
            abort(404)
        return send_file(
            addon_path,
            as_attachment=True,
            download_name="taf_addon.py",
            mimetype="text/x-python",
        )

    @bp.get("/patient/<int:patient_id>")
    @require_access
    def patient(patient_id):
        setup_response = require_schema()
        if setup_response:
            return setup_response
        patient_record = get_patient(patient_id)
        model_files, model_files_source_available = decorated_model_files(patient_record)
        return render_template(
            "followup_visit.html",
            patient=patient_record,
            model_files=model_files,
            model_files_source_available=model_files_source_available,
        )

    @bp.post("/patient/<int:patient_id>/model-files")
    @require_access
    def upload_model_files(patient_id):
        validate_csrf()
        patient_record = get_patient(patient_id)
        uploads = [item for item in request.files.getlist("model_files") if item.filename]
        if not uploads:
            abort(400, description="Legalább egy fájl kiválasztása szükséges.")
        uploaded_names = []
        for upload in uploads:
            temporary_path = os.path.join(upload_folder, f"model_{uuid.uuid4().hex}.upload")
            try:
                upload.save(temporary_path)
                uploaded_names.append(
                    model_file_uploader(
                        normalise_taj(patient_record.get("taj")),
                        temporary_path,
                        upload.filename,
                    )
                )
            except Exception:
                current_app.logger.exception("Model-analysis file upload failed")
                abort(502, description="A fájl NAS-ra feltöltése nem sikerült.")
            finally:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)
        flash(f"{len(uploaded_names)} modellanalízis-fájl feltöltve.", "success")
        return redirect(url_for("followup.patient", patient_id=patient_id))

    @bp.get("/patient/<int:patient_id>/model-files/<file_id>")
    @require_access
    def download_model_file(patient_id, file_id):
        patient_record = get_patient(patient_id)
        files, source_available = decorated_model_files(patient_record)
        if not source_available:
            abort(503, description="A NAS FTP-tárhely jelenleg nem érhető el.")
        selected = next((item for item in files if item["file_id"] == file_id), None)
        if not selected:
            abort(404)
        temporary_path = os.path.join(upload_folder, f"download_{uuid.uuid4().hex}")
        try:
            model_file_downloader(
                model_file_keys(patient_record), selected["name"], temporary_path
            )
        except FileNotFoundError:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
            abort(404)
        except Exception:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)
            current_app.logger.exception("Model-analysis file download failed")
            abort(502, description="A fájl NAS-ról letöltése nem sikerült.")

        @after_this_request
        def remove_temporary_file(response):
            try:
                os.remove(temporary_path)
            except OSError:
                pass
            return response

        return send_file(
            temporary_path,
            as_attachment=True,
            download_name=selected["name"],
        )

    @bp.post("/patient/<int:patient_id>/logistics")
    @require_access
    def save_logistics(patient_id):
        validate_csrf()
        patient_record = get_patient(patient_id)
        status = request.form.get("visit_status")
        if status not in VISIT_STATUSES:
            abort(400, description="Érvénytelen vizitstátusz.")
        if status == "completed" and patient_record.get("visit_status") != "completed":
            abort(400, description="A vizitet csak a hiányellenőrzéssel lehet lezárni.")
        appointment_raw = request.form.get("appointment_at", "").strip()
        appointment = None
        if appointment_raw:
            try:
                appointment = datetime.strptime(appointment_raw, "%Y-%m-%dT%H:%M")
            except ValueError:
                abort(400, description="Érvénytelen időpont.")
        patient_phone = request.form.get("patient_phone", "").strip()
        if patient_phone and not re.fullmatch(r"[0-9+()\-\s/.]{6,30}", patient_phone):
            abort(400, description="A telefonszám formátuma nem érvényes.")
        values = {
            "visit_status": status,
            "appointment_at": appointment,
            "patient_display_name": request.form.get("patient_display_name", "").strip() or None,
            "patient_phone": patient_phone or None,
            "contact_note": request.form.get("contact_note", "").strip() or None,
            "nonattendance_reason": request.form.get("nonattendance_reason", "").strip() or None,
        }
        if status != "not_contacted":
            values["contact_attempted_at"] = datetime.now()
        statements = [upsert_statement(patient_id, values)]
        if status not in {"not_contacted", "completed"}:
            statements.append(
                (
                    """
                    INSERT INTO followup_contact_attempts (patient_id, outcome, note)
                    VALUES (%s, %s, %s)
                    """,
                    (patient_id, status, values["contact_note"]),
                )
            )
        execute_transaction(statements)
        flash("A megkeresési és időpontadatok elmentve.", "success")
        return redirect(url_for("followup.patient", patient_id=patient_id))

    @bp.post("/patient/<int:patient_id>/intake")
    @require_access
    def save_intake(patient_id):
        validate_csrf()
        get_patient(patient_id)
        months_raw = request.form.get("months_since_delivery", "").strip().replace(",", ".")
        months = None
        if months_raw:
            try:
                months = float(months_raw)
                if months < 0:
                    raise ValueError
            except ValueError:
                abort(400, description="A fogsor átadása óta eltelt idő nem érvényes.")
        f9_raw = request.form.get("f9", "").strip()
        f9 = int(f9_raw) if f9_raw in {"1", "2", "3"} else None
        upsert_fields(
            patient_id,
            {
                "visit_status": "arrived",
                "consent_confirmed": request.form.get("consent_confirmed") == "1",
                "data_collector": request.form.get("data_collector", "").strip() or None,
                "months_since_delivery": months,
                "interim_adjustments": request.form.get("interim_adjustments", "").strip() or None,
                "interim_events": request.form.get("interim_events", "").strip() or None,
                "f9": f9,
            },
        )
        flash("A beléptetési adatok elmentve.", "success")
        return redirect(url_for("followup.patient", patient_id=patient_id))

    @bp.route("/patient/<int:patient_id>/questionnaire", methods=["GET", "POST"])
    @require_access
    def questionnaire(patient_id):
        setup_response = require_schema()
        if setup_response:
            return setup_response
        patient_record = get_patient(patient_id)
        if patient_record["questionnaire_source"] == "legacy" and request.method == "GET":
            flash("Ehhez a beteghez már tartozik teljes korábbi utánkövetési kérdőív.", "info")
            return redirect(url_for("followup.patient", patient_id=patient_id))

        if request.method == "POST":
            validate_csrf()
            values = {}
            for field in ANCHOR_FIELDS:
                value = request.form.get(field, "")
                allowed_values = SITUATION_VALUES if "today_situation" in field else CHANGE_VALUES
                if value not in allowed_values:
                    abort(400, description="Minden anchor-kérdés megválaszolása kötelező.")
                values[field] = value
            for field in OHIP_FIELDS:
                value = request.form.get(field, "")
                if value not in {"0", "1", "2", "3", "4"}:
                    abort(400, description="Minden OHIP-kérdés megválaszolása kötelező.")
                values[field] = int(value)
            for field in GOHAI_FIELDS:
                value = request.form.get(field, "")
                if value not in {"1", "2", "3", "4", "5"}:
                    abort(400, description="Minden GOHAI-kérdés megválaszolása kötelező.")
                values[field] = int(value)
            values["questionnaire_completed_at"] = datetime.now()
            upsert_fields(patient_id, values)
            flash("A kérdőív hiánytalanul elmentve.", "success")
            return redirect(url_for("followup.patient", patient_id=patient_id))

        form_values = {field: patient_record.get(f"new_{field}") for field in QUESTIONNAIRE_FIELDS}
        return render_template(
            "followup_questionnaire.html",
            patient=patient_record,
            form_values=form_values,
        )

    @bp.post("/patient/<int:patient_id>/questionnaire/draft")
    @require_access
    def save_questionnaire_draft(patient_id):
        validate_csrf()
        get_patient(patient_id)
        values = {}
        for field in QUESTIONNAIRE_FIELDS:
            if field not in request.form:
                continue
            value = request.form.get(field, "")
            if field in ANCHOR_FIELDS:
                allowed_values = SITUATION_VALUES if "today_situation" in field else CHANGE_VALUES
                if value not in allowed_values:
                    abort(400, description="Érvénytelen kérdőívválasz.")
                values[field] = value
            elif field in OHIP_FIELDS:
                if value not in {"0", "1", "2", "3", "4"}:
                    abort(400, description="Érvénytelen OHIP-válasz.")
                values[field] = int(value)
            else:
                if value not in {"1", "2", "3", "4", "5"}:
                    abort(400, description="Érvénytelen GOHAI-válasz.")
                values[field] = int(value)
        if not values:
            abort(400, description="Nincs menthető kérdőívválasz.")
        upsert_fields(patient_id, values)
        return jsonify({"saved": True})

    @bp.post("/patient/<int:patient_id>/mai")
    @require_access
    def save_mai(patient_id):
        validate_csrf()
        patient_record = get_patient(patient_id)
        if not patient_record["mai_eligible"]:
            abort(400, description="Kiindulási MAI nélkül utánkövetési MAI-pár nem képezhető.")
        image = request.files.get("image")
        if not image or not image.filename or not allowed_file(image.filename):
            abort(400, description="TIFF kép feltöltése szükséges.")
        safe_name = secure_filename(image.filename)
        temp_name = f"followup_{patient_id}_{uuid.uuid4().hex}_{safe_name}"
        temp_path = os.path.join(upload_folder, temp_name)
        image.save(temp_path)
        try:
            hue_degree = hue_calculator(temp_path)
            if hue_degree is None or not math.isfinite(float(hue_degree)):
                abort(400, description="A képből nem számítható érvényes MAI hue-degree.")
            image_path = nas_uploader(temp_path, patient_record["study_code"], "mai_followup")
            upsert_fields(
                patient_id,
                {
                    "final_mai_huedegree": hue_degree,
                    "final_image_path": image_path,
                    "mai_completed_at": datetime.now(),
                    "mai_note": request.form.get("mai_note", "").strip() or None,
                },
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        flash("Az utánkövetési MAI-kép és a hue-degree elmentve.", "success")
        return redirect(url_for("followup.patient", patient_id=patient_id))

    @bp.post("/patient/<int:patient_id>/complete")
    @require_access
    def complete_visit(patient_id):
        validate_csrf()
        patient_record = get_patient(patient_id)
        missing = []
        if not patient_record["consent_confirmed"]:
            missing.append("beleegyezés")
        if not patient_record["questionnaire_complete"]:
            missing.append("OHIP–GOHAI–anchor kérdőív")
        if not patient_record["f9_complete"]:
            missing.append("F9")
        if patient_record["mai_eligible"] and not patient_record["mai_complete"]:
            missing.append("utánkövetési MAI")
        if missing:
            flash("A vizit még nem zárható le. Hiányzik: " + ", ".join(missing) + ".", "error")
        else:
            upsert_fields(
                patient_id,
                {"visit_status": "completed", "completed_at": datetime.now()},
            )
            flash("A vizit hiánytalanul lezárva.", "success")
        return redirect(url_for("followup.patient", patient_id=patient_id))

    @bp.get("/export.csv")
    @require_access
    def export_csv():
        setup_response = require_schema()
        if setup_response:
            return setup_response
        anatomy_columns = [
            "A1_Kaan",
            "A4_jobb",
            "A4_bal",
            "A5_jobb",
            "A5_bal",
            "A6_jobb",
            "A6_bal",
            "A7_jobb",
            "A7_bal",
            "A8_jobb",
            "A8_bal",
            "A9_jobb",
            "A9_bal",
            "A11",
            "A12",
        ]
        baseline_columns = [
            *[f"OHIP_{i}" for i in range(1, 6)],
            *[f"GOHAI_{i}" for i in range(1, 13)],
            "init_mai_huedegree",
        ]
        selected_source = ",\n".join(
            [f'c."{column}" AS "{column}"' for column in anatomy_columns + baseline_columns]
        )
        selected_followup = ",\n".join(
            [
                f'COALESCE(f.{field}, c."{_legacy_column(field)}") AS {field}'
                for field in QUESTIONNAIRE_FIELDS
            ]
        )
        export_query = f"""
            WITH latest AS (
                SELECT DISTINCT ON ("TAJ") * FROM patients
                WHERE "TAJ" IS NOT NULL
                ORDER BY "TAJ", "id" DESC
            ), cohort AS (
                SELECT * FROM latest
                WHERE LOWER(TRIM("denture_type")) = 'both'
                  AND "OHIP_1" IS NOT NULL
            )
            SELECT
                c."id" AS source_patient_id,
                c."gender" AS gender,
                EXTRACT(YEAR FROM AGE(CURRENT_DATE, c."birthdate"))::INTEGER AS age,
                {selected_source},
                {selected_followup},
                COALESCE(f.f9, c."F9") AS f9,
                COALESCE(f.final_mai_huedegree, c."final_mai_huedegree") AS final_mai_huedegree,
                f.visit_status,
                f.appointment_at,
                f.consent_confirmed,
                f.data_collector,
                f.months_since_delivery,
                f.interim_adjustments,
                f.interim_events,
                f.questionnaire_completed_at,
                f.mai_completed_at,
                f.completed_at
            FROM cohort c
            LEFT JOIN followup_visits f
              ON f.patient_id = c."id" AND f.visit_round = 1
            ORDER BY c."id"
        """
        export_rows = rows(export_query)
        for record in export_rows:
            record["study_code"] = study_code(record.pop("source_patient_id"))
        output = io.StringIO()
        fieldnames = list(export_rows[0].keys()) if export_rows else ["study_code"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(export_rows)
        filename = f"PREDICT_followup_{datetime.now():%Y%m%d_%H%M}.csv"
        return Response(
            "\ufeff" + output.getvalue(),
            mimetype="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return bp


def _legacy_column(field):
    if field.startswith("ohip_"):
        number = field.split("_")[1]
        return f"OHIP_{number}_recall"
    if field.startswith("gohai_"):
        number = field.split("_")[1]
        return f"GOHAI_{number}_recall"
    return field
