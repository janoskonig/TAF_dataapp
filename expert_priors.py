"""Szakértői prior-felmérés: különálló, hozzáférési kóddal védett munkafelület.

A PREDICT anatómiai tételeihez több tapasztalt kolléga előzetes szakmai
meggyőződését (irány, bizonyosság, hatásnagyság) gyűjti össze. A válaszok az
``expert_prior_responses`` táblába kerülnek (JSONB-blokkok), és pontosan
abban a hosszú CSV-formátumban exportálhatók, amelyet a
``predict_bayes_feltaro.R`` szkript priorként beolvas.

Hozzáférés:
  * szakértő: kód nélkül, a link elég (a kitöltést kitalálhatatlan token
    azonosítja; opcionálisan ``EXPERT_ACCESS_CODE``-dal zárható);
  * vizsgálatvezető: a klinikai ``followup_authenticated`` munkamenet
    (válaszlista, megtekintés, CSV-export, törlés).
"""

from __future__ import annotations

import csv
import io
import json
import os
import secrets
from datetime import datetime, timezone
from functools import wraps
from zoneinfo import ZoneInfo
from urllib.parse import urlsplit

from flask import (
    Blueprint,
    Response,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from psycopg2.extras import Json

FORM_VERSION = "v1.0"
LOCAL_TZ = ZoneInfo("Europe/Budapest")


def to_local(value):
    """DB CURRENT_TIMESTAMP (UTC, tz nélkül) → budapesti helyi idő."""
    if not isinstance(value, datetime):
        return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(LOCAL_TZ)


def format_stamp(value, fmt="%Y-%m-%d %H:%M"):
    local = to_local(value)
    if isinstance(local, datetime):
        return local.strftime(fmt)
    if not value:
        return ""
    return str(value)[:10] if fmt == "%Y-%m-%d" else str(value)[:16]

# ---------------------------------------------------------------------------
# Tételregiszter — a kódok azonosak az R-szkript regiszterével
# ---------------------------------------------------------------------------
ITEMS = [
    {
        "kod": "F1", "nev": "Felső állcsontgerinc magassága (profilja)", "jaw": "Felső állcsont",
        "rogzit": "Modellanalízis: a gerincél relatív magassága a bukkális áthajláshoz képest (mm, betegátlag); klinikailag: magas / közepes / alacsony (sorvadt) gerinc.",
        "A": "magas, jól megtartott gerinc", "B": "alacsony, sorvadt gerinc", "alak": "monoton", "optimum": False,
        "kuszob": "Milyen gerincmagasság alatt tekinti a felső gerincet klinikailag „alacsonynak”? (mm)", "subs": [],
    },
    {
        "kod": "F2", "nev": "Felső alámenős területek nagysága", "jaw": "Felső állcsont",
        "rogzit": "Modellanalízis: az alámenős területek összesített térfogata (mm³), ívhosszra standardizálva is (F2/L³). Klinikailag: nincs / kis / nagy alámenősség.",
        "A": "kevés vagy kis alámenősség", "B": "nagy alámenősség", "alak": "optimum", "optimum": True,
        "kuszob": "Ha optimum van: körülbelül hol? (szavakkal, pl. „enyhe, egyenletes alámenősség”)", "subs": [],
    },
    {
        "kod": "F3", "nev": "Szájpadboltozat magassága", "jaw": "Felső állcsont",
        "rogzit": "Modellanalízis: a szájpadboltozat magassága (mm). Klinikailag: magas, boltozatos / közepes / lapos szájpad.",
        "A": "magas, boltozatos szájpad", "B": "lapos szájpad", "alak": "monoton", "optimum": False,
        "kuszob": "Milyen boltozatmagasság alatt tekinti a szájpadot „laposnak”? (mm)", "subs": [],
    },
    {
        "kod": "F4", "nev": "Felső állcsontgerinc alakja (ívszög)", "jaw": "Felső állcsont",
        "rogzit": "Modellanalízis: a felső gerincív szöge (°); nagyobb szög = négyzetesebb ív, kisebb szög = elkeskenyedő (V-alakú) ív.",
        "A": "négyzetes ív (nagy szög)", "B": "elkeskenyedő, V-alakú ív (kis szög)", "alak": "kuszobos", "optimum": False,
        "kuszob": "Van-e olyan szög, amely felett már nincs további előny (platózás)? (°)", "subs": [],
    },
    {
        "kod": "F5", "nev": "Lötyögő, csontmag nélküli gerinc", "jaw": "Felső állcsont",
        "rogzit": "Klinikai vizsgálat: nincs / van, „lötyögő” tuberek / van, frontális gerincen.",
        "A": "nincs lötyögő gerinc", "B": "van lötyögő gerinc (bárhol)", "alak": "monoton", "optimum": False,
        "kuszob": None,
        "subs": [{"kod": "lokalizacio", "kerdes": "Ha van: melyik lokalizáció rosszabb?",
                  "opciok": [("frontalis", "frontális"), ("tuberalis", "tuberális"), ("egyforma", "egyforma"), ("nem_tudom", "nem tudom")]}],
    },
    {
        "kod": "F6", "nev": "Interalveoláris vonal és rágósík szöge", "jaw": "Felső állcsont",
        "rogzit": "Modellanalízis: a felső és alsó gerincélvonalat összekötő egyenes és a rágósík szöge (°); 90° = a gerincélvonalak vertikálisan egybeesnek.",
        "A": "≈ 90° (egybeeső gerincélvonalak)", "B": "90°-tól jelentősen eltérő szög", "alak": "optimum", "optimum": True,
        "kuszob": "Mekkora eltérést tekint 90°-tól klinikailag jelentősnek? (°)",
        "subs": [{"kod": "irany_szamit", "kerdes": "Számít-e az eltérés iránya (alsó szélesebb vs. felső szélesebb)?",
                  "opciok": [("igen", "igen"), ("nem", "nem"), ("nem_tudom", "nem tudom")]}],
    },
    {
        "kod": "F7", "nev": "Torus palatinus", "jaw": "Felső állcsont",
        "rogzit": "Klinikai vizsgálat: nincs / plató alakú / orsó alakú.",
        "A": "nincs torus", "B": "van torus (plató vagy orsó)", "alak": "monoton", "optimum": False,
        "kuszob": None,
        "subs": [{"kod": "alak_rosszabb", "kerdes": "Ha van: melyik alak rosszabb?",
                  "opciok": [("orso", "orsó"), ("plato", "plató"), ("egyforma", "egyforma"), ("nem_tudom", "nem tudom")]}],
    },
    {
        "kod": "F8", "nev": "Antagonista fogazat (a felső fogpótláshoz)", "jaw": "Felső állcsont",
        "rogzit": "Klinikai vizsgálat: (1) nincs, most készül; (2) teljes lemezes fogpótlás / overdenture / részleges fémlemezes; (3) teljesen megtartott vagy rögzített fogpótlással helyreállított fogazat.",
        "A": "teljes lemezes vagy kivehető antagonista", "B": "megtartott vagy rögzített antagonista fogazat", "alak": "interakcio", "optimum": False,
        "kuszob": None,
        "subs": [{"kod": "kategoria_vagy_interakcio", "kerdes": "A kategória önmagában számít, vagy csak az erők iránya (pl. sorvadt maxilla + erőltetett ollóharapás)?",
                  "opciok": [("kategoria", "a kategória önmagában"), ("interakcio", "csak interakcióban"), ("mindketto", "mindkettő"), ("nem_tudom", "nem tudom")]}],
    },
    {
        "kod": "A1", "nev": "Alsó állcsontgerinc alakja Kaán szerint", "jaw": "Alsó állcsont",
        "rogzit": "Klinikai vizsgálat, 5 fokozat: (1) egészében megtartott; (2) elöl megtartott, oldalt lapos; (3) egészében lapos; (4) negatív; (5) mélyült negatív.",
        "A": "egészében megtartott gerinc (1)", "B": "mélyült negatív gerinc (5)", "alak": "telitodo", "optimum": False,
        "kuszob": None,
        "subs": [
            {"kod": "legnagyobb_ugras", "kerdes": "Hol a legnagyobb ugrás a sikerben?",
             "opciok": [("1_2", "1→2"), ("2_3", "2→3"), ("3_4", "3→4"), ("4_5", "4→5"), ("egyenletes", "egyenletes")]},
            {"kod": "negy_ot_kulonbseg", "kerdes": "Van-e érdemi különbség a 4-es és 5-ös fokozat között?",
             "opciok": [("igen", "igen"), ("nem", "nem"), ("nem_tudom", "nem tudom")]},
        ],
    },
    {
        "kod": "A2", "nev": "Alsó állcsontgerinc magassága (modellanalízis)", "jaw": "Alsó állcsont",
        "rogzit": "Modellanalízis: a gerincél előjeles magassága a bukkális–lingvális referenciaszinthez képest (mm, betegátlag); ugyanazt a gerincállapotot méri, mint A1, folytonosan.",
        "A": "magas alsó gerinc", "B": "alacsony (referenciaszint alatti) alsó gerinc", "alak": "monoton", "optimum": False,
        "kuszob": "Milyen gerincmagasság alatt tekinti az alsó gerincet klinikailag „sorvadtnak”? (mm)", "subs": [],
    },
    {
        "kod": "A3", "nev": "Buccinator tasak", "jaw": "Alsó állcsont",
        "rogzit": "Klinikai vizsgálat oldalanként: szájnyitáskor beszűkülő / szájnyitáskor kiszélesedő / lebenyezett felszínű.",
        "A": "szájnyitáskor beszűkülő", "B": "szájnyitáskor kiszélesedő", "alak": "nincs_irany", "optimum": False,
        "kuszob": None,
        "subs": [{"kod": "lebenyezett", "kerdes": "Hova sorolja a lebenyezett felszínű formát?",
                  "opciok": [("legkedvezobb", "a legkedvezőbb"), ("koztes", "köztes"), ("legkedvezotlenebb", "a legkedvezőtlenebb"), ("nem_tudom", "nem tudom")]}],
    },
    {
        "kod": "A4", "nev": "Torus mandibularis", "jaw": "Alsó állcsont",
        "rogzit": "Klinikai vizsgálat oldalanként: nincs / kis méretű / nagy méretű.",
        "A": "nincs torus", "B": "van torus (kicsi vagy nagy)", "alak": "monoton", "optimum": False,
        "kuszob": None,
        "subs": [{"kod": "nagy_vs_kicsi", "kerdes": "Ha van: mennyivel rosszabb a nagy a kicsinél?",
                  "opciok": [("alig", "alig"), ("kozepesen", "közepesen"), ("sokkal", "sokkal"), ("nem_tudom", "nem tudom")]}],
    },
    {
        "kod": "A5", "nev": "Lingualis tasak (a környező izmok ereje nyeléskor)", "jaw": "Alsó állcsont",
        "rogzit": "Klinikai vizsgálat oldalanként: nyeléskor a környező izmok (1) nem szűkítik a tasakot / (2) ujjunkat a mandibulához préselik / (3) ujjunkat kifelé préselik.",
        "A": "az izmok ujjunkat a mandibulához préselik", "B": "az izmok ujjunkat kifelé préselik", "alak": "monoton", "optimum": False,
        "kuszob": None,
        "subs": [{"kod": "nem_szukit_helye", "kerdes": "Hova sorolja a „nem szűkíti” változatot?",
                  "opciok": [("A_kozel", "közel az A-hoz"), ("kozepen", "középen"), ("B_kozel", "közel a B-hez"), ("nem_tudom", "nem tudom")]}],
    },
    {
        "kod": "TUB", "nev": "Tuberculum alveolare mandibulae (A6–A9 együtt)", "jaw": "Alsó állcsont",
        "rogzit": "Négy klinikai tétel oldalanként: A6 feszes ínyborítás (az egészet / elülső harmadát / egyáltalán nem); A7 alak (fordított körte / kicsi, elkülönülő / plicaszerű); A8 tuberculum–gerinc inklináció (nincs eltérés / jelentős eltérés); A9 alakváltozás nyitás–záráskor (nem változik / kissé / abszolút mozgékony).",
        "A": "feszes ínnyel fedett, jól formált, stabil tuberculum", "B": "fedetlen, plicaszerű, mozgékony tuberculum", "alak": "monoton", "optimum": False,
        "kuszob": None,
        "subs": [{"kod": "legfontosabb", "kerdes": "Melyik a legfontosabb a négy jellemző közül a siker szempontjából?",
                  "opciok": [("A6", "A6 feszes íny"), ("A7", "A7 alak"), ("A8", "A8 inklináció"), ("A9", "A9 mozgékonyság"), ("nem_tudom", "nem tudom")]}],
    },
    {
        "kod": "A10", "nev": "Állcsontreláció szöge (Angle-osztály)", "jaw": "Alsó állcsont",
        "rogzit": "Modellanalízis: a felső és alsó gerincélvonal legelülső pontjait összekötő egyenes és a rágósík szöge (°); a mandibula sagittalis helyzetét, gyakorlatilag az Angle-osztályt jellemzi.",
        "A": "Angle I (normális reláció)", "B": "Angle II vagy III (eltérő reláció)", "alak": "nincs_irany", "optimum": True,
        "kuszob": None,
        "subs": [{"kod": "melyik_rosszabb", "kerdes": "Ha az eltérés kedvezőtlen: melyik rosszabb?",
                  "opciok": [("angle_II", "Angle II"), ("angle_III", "Angle III"), ("egyforma", "egyforma"), ("nem_tudom", "nem tudom")]}],
    },
    {
        "kod": "A11", "nev": "Szublingvális tájék / szájfenék", "jaw": "Alsó állcsont",
        "rogzit": "Klinikai vizsgálat: (1) nem elődomborodó / (2) puhán elődomborodó / (3) tömött, elődomborodó szájfenék.",
        "A": "puhán elődomborodó szájfenék", "B": "tömött, elődomborodó szájfenék", "alak": "monoton", "optimum": False,
        "kuszob": None,
        "subs": [{"kod": "nem_elodomborodo_helye", "kerdes": "Hova sorolja a „nem elődomborodó” szájfeneket?",
                  "opciok": [("A_kozel", "közel az A-hoz"), ("kozepen", "középen"), ("B_kozel", "közel a B-hez"), ("nem_tudom", "nem tudom")]}],
    },
    {
        "kod": "A12", "nev": "Spinae mentales", "jaw": "Alsó állcsont",
        "rogzit": "Klinikai vizsgálat: nem tapintható / tapintható / nyomásra érzékeny.",
        "A": "nem tapintható", "B": "tapintható vagy nyomásra érzékeny", "alak": "monoton", "optimum": False,
        "kuszob": None,
        "subs": [{"kod": "erzekeny_vs_tapinthato", "kerdes": "Ha tapintható: mennyivel rosszabb a nyomásérzékeny a csak tapinthatónál?",
                  "opciok": [("alig", "alig"), ("kozepesen", "közepesen"), ("sokkal", "sokkal"), ("nem_tudom", "nem tudom")]}],
    },
]
ITEM_CODES = [item["kod"] for item in ITEMS]
ITEMS_BY_CODE = {item["kod"]: item for item in ITEMS}

IRANY_VALUES = {"A_kedvezotlenebb", "B_kedvezotlenebb", "nem_monoton", "nincs_kulonbseg", "nem_tudom"}
IRANY_LABELS = {
    "A_kedvezotlenebb": "az A pólus a kedvezőtlenebb",
    "B_kedvezotlenebb": "a B pólus a kedvezőtlenebb",
    "nem_monoton": "nem monoton: a közepes érték a legjobb (optimum)",
    "nincs_kulonbseg": "nincs érdemi különbség",
    "nem_tudom": "nem tudom megítélni",
}
DIRECTIONAL = {"A_kedvezotlenebb", "B_kedvezotlenebb"}
P_IRANY_VALUES = [50, 60, 70, 80, 90, 95, 99]
MECHANISMS = [
    ("retencio", "retenció (szívóhatás)"),
    ("stabilitas", "stabilitás (elmozdulás)"),
    ("alatamasztas", "alátámasztás / teherviselés"),
    ("fajdalom", "fájdalom, nyomásérzékenység"),
    ("technikai", "technikai kivitelezés (lenyomat, kiterjesztés)"),
    ("egyeb", "egyéb"),
]
MECHANISM_CODES = {code for code, _ in MECHANISMS}
ITEM_TEXT_FIELDS = {"kuszob", "megjegyzes", "mechanizmus_egyeb"}
ITEM_INT_FIELDS = {"siker_A": (0, 100), "siker_B": (0, 100), "kulonbseg_min": (0, 100), "kulonbseg_max": (0, 100)}

BACKGROUND_FIELDS = {
    "diploma_ev": ("int", 1950, 2100),
    "evek_gyakorlat": ("int", 0, 70),
    "fogsorok_szama_kat": ("choice", {"<100", "100-500", "500-1000", "1000-3000", ">3000"}),
    "oktat": ("choice", {"igen", "nem", "korabban"}),
    "tevekenyseg": ("choice", {"egyetemi", "maganpraxis", "mindketto", "egyeb"}),
    "szakvizsga": ("text", 200),
    "evi_fogsorok": ("int", 0, 2000),
}
CALIBRATION_FIELDS = {
    "alap_siker_100": ("int", 0, 100),
    "anatomia_sulya_pct": ("int", 0, 100),
    "felso_vagy_also": ("choice", {"felso", "also", "egyforman"}),
    "felso_also_arany": ("text", 40),
    "kizaro_kepletek": ("text", 2000),
    "regi_fogsor": ("choice", {"alig", "kozepesen", "erosen"}),
    "regi_fogsor_megjegyzes": ("text", 1000),
}
CLOSING_FIELDS = {
    "rang_1": ("choice", set(ITEM_CODES)),
    "rang_2": ("choice", set(ITEM_CODES)),
    "rang_3": ("choice", set(ITEM_CODES)),
    "rang_4": ("choice", set(ITEM_CODES)),
    "rang_5": ("choice", set(ITEM_CODES)),
    "legrosszabb_kombinacio": ("text", 1000),
    "ellensulyozo": ("text", 1000),
    "hianyzo_kepletek": ("text", 2000),
    "onertekeles": ("choice", {"1", "2", "3", "4", "5"}),
    "megjegyzes": ("text", 3000),
}
SECTION_SPECS = {"bg": BACKGROUND_FIELDS, "cal": CALIBRATION_FIELDS, "cl": CLOSING_FIELDS}
SECTION_COLUMNS = {"bg": "background", "cal": "calibration", "cl": "closing"}

PRIOR_CSV_COLUMNS = [
    "szakerto_id", "datum", "tetel", "irany", "p_irany", "siker_A", "siker_B",
    "kulonbseg_min", "kulonbseg_max", "alak", "mechanizmus", "kuszob", "megjegyzes",
]
BACKGROUND_CSV_COLUMNS = [
    "szakerto_id", "datum", "evek_gyakorlat", "fogsorok_szama_kat", "oktat", "alap_siker_100",
    "anatomia_sulya_pct", "rang_1", "rang_2", "rang_3", "rang_4", "rang_5", "hianyzo_kepletek", "megjegyzes",
    "nev", "intezmeny",
]


class FieldError(ValueError):
    """Érvénytelen űrlapérték."""


def _clean_text(value, limit):
    text = str(value or "").strip()
    return text[:limit]


def parse_scalar(spec, value):
    """Validate a background/calibration/closing value; '' → None (törlés)."""
    kind = spec[0]
    if value is None or str(value).strip() == "":
        return None
    if kind == "int":
        try:
            number = int(str(value).strip())
        except ValueError as err:
            raise FieldError("Egész számot kérünk.") from err
        if not spec[1] <= number <= spec[2]:
            raise FieldError(f"Az érték {spec[1]} és {spec[2]} közé kell essen.")
        return number
    if kind == "choice":
        text = str(value).strip()
        if text not in spec[1]:
            raise FieldError("Nem megengedett választás.")
        return text
    return _clean_text(value, spec[1])


def parse_item_field(item_code, field, value):
    """Validate one tételmező; returns (json_key, cleaned_value)."""
    item = ITEMS_BY_CODE.get(item_code)
    if item is None:
        raise FieldError("Ismeretlen tétel.")
    raw = "" if value is None else str(value).strip()
    if field == "irany":
        if raw == "":
            return field, None
        if raw not in IRANY_VALUES or (raw == "nem_monoton" and not item["optimum"]):
            raise FieldError("Nem megengedett irány.")
        return field, raw
    if field == "p_irany":
        if raw == "":
            return field, None
        try:
            number = int(raw)
        except ValueError as err:
            raise FieldError("A bizonyosság százalékos szám.") from err
        if number not in P_IRANY_VALUES:
            raise FieldError("A bizonyosság csak 50, 60, 70, 80, 90, 95 vagy 99 lehet.")
        return field, number
    if field in ITEM_INT_FIELDS:
        if raw == "":
            return field, None
        try:
            number = int(raw)
        except ValueError as err:
            raise FieldError("Egész számot kérünk (0–100).") from err
        lo, hi = ITEM_INT_FIELDS[field]
        if not lo <= number <= hi:
            raise FieldError("Az érték 0 és 100 közé kell essen.")
        return field, number
    if field == "mechanizmus":
        codes = [part for part in raw.split(";") if part]
        if any(code not in MECHANISM_CODES for code in codes):
            raise FieldError("Ismeretlen mechanizmus.")
        return field, sorted(set(codes))
    if field in ITEM_TEXT_FIELDS:
        return field, _clean_text(raw, 1000) or None
    if field.startswith("sub__"):
        sub_code = field[len("sub__"):]
        for sub in item["subs"]:
            if sub["kod"] == sub_code:
                if raw == "":
                    return f"sub_{sub_code}", None
                if raw not in {code for code, _ in sub["opciok"]}:
                    raise FieldError("Nem megengedett választás.")
                return f"sub_{sub_code}", raw
    raise FieldError("Ismeretlen tételmező.")


def parse_field_name(name):
    """'item__F1__irany' → ('item', 'F1', 'irany'); 'bg__oktat' → ('bg', None, 'oktat')."""
    parts = str(name or "").split("__", 2)
    if len(parts) == 3 and parts[0] == "item" and parts[1] and parts[2]:
        return "item", parts[1], parts[2]
    if len(parts) == 2 and parts[0] in SECTION_SPECS and parts[1]:
        return parts[0], None, parts[1]
    raise FieldError("Ismeretlen mező.")


def collect_form(form):
    """A teljes űrlap feldolgozása: {'background':{}, 'calibration':{}, 'closing':{}, 'items':{}}, errors."""
    result = {"background": {}, "calibration": {}, "closing": {}, "items": {code: {} for code in ITEM_CODES}}
    errors = []
    mechanisms = {}
    for name in form.keys():
        try:
            section, item_code, field = parse_field_name(name)
        except FieldError:
            continue
        try:
            if section == "item":
                if field == "mechanizmus":
                    mechanisms.setdefault(item_code, []).extend(form.getlist(name))
                    continue
                key, value = parse_item_field(item_code, field, form.get(name))
                result["items"][item_code][key] = value
            else:
                spec = SECTION_SPECS[section].get(field)
                if spec is None:
                    continue
                result[SECTION_COLUMNS[section]][field] = parse_scalar(spec, form.get(name))
        except FieldError as err:
            errors.append(f"{name}: {err}")
    for item_code, codes in mechanisms.items():
        try:
            _, value = parse_item_field(item_code, "mechanizmus", ";".join(codes))
            result["items"][item_code]["mechanizmus"] = value
        except FieldError as err:
            errors.append(f"item__{item_code}__mechanizmus: {err}")
    return result, errors


def completeness_errors(data):
    """A beküldés feltételei (hiánylista, emberi olvasásra)."""
    problems = []
    background = data.get("background") or {}
    calibration = data.get("calibration") or {}
    closing = data.get("closing") or {}
    items = data.get("items") or {}
    if background.get("evek_gyakorlat") is None:
        problems.append("A. Protetikai gyakorlat évei")
    if not background.get("fogsorok_szama_kat"):
        problems.append("A. Elkészített teljes fogsorok száma")
    if calibration.get("alap_siker_100") is None:
        problems.append("B1. Alap-sikerarány")
    if calibration.get("anatomia_sulya_pct") is None:
        problems.append("B2. Az anatómia súlya")
    for item in ITEMS:
        answers = items.get(item["kod"]) or {}
        label = f"{item['kod']} · {item['nev']}"
        if not answers.get("irany"):
            problems.append(f"{label}: irány")
        elif answers.get("irany") in DIRECTIONAL and answers.get("p_irany") is None:
            problems.append(f"{label}: bizonyosság")
        lo, hi = answers.get("kulonbseg_min"), answers.get("kulonbseg_max")
        if lo is not None and hi is not None and lo > hi:
            problems.append(f"{label}: a tartomány alsó határa nagyobb a felsőnél")
    if not closing.get("onertekeles"):
        problems.append("D5. Önértékelés")
    return problems


def item_progress(items):
    answered = 0
    for code in ITEM_CODES:
        answers = (items or {}).get(code) or {}
        if answers.get("irany") and (answers["irany"] not in DIRECTIONAL or answers.get("p_irany") is not None):
            answered += 1
    return answered


def derive_alak(item, answers):
    irany = answers.get("irany")
    if irany == "nem_monoton":
        return "optimum"
    if irany in {"nincs_kulonbseg", "nem_tudom"}:
        return "nincs_irany"
    return item["alak"]


def item_note(item, answers):
    parts = []
    for sub in item["subs"]:
        value = answers.get(f"sub_{sub['kod']}")
        if value:
            labels = dict(sub["opciok"])
            parts.append(f"{sub['kod']}={labels.get(value, value)}")
    if answers.get("mechanizmus_egyeb"):
        parts.append(f"egyéb mechanizmus: {answers['mechanizmus_egyeb']}")
    if answers.get("megjegyzes"):
        parts.append(answers["megjegyzes"])
    return " | ".join(parts)


def prior_rows(responses):
    """Hosszú formátum: egy sor / szakértő / tétel (a predict_expert_priorok.csv oszlopai)."""
    rows = []
    for response in responses:
        stamp = response.get("submitted_at") or response.get("updated_at")
        datum = format_stamp(stamp, "%Y-%m-%d") if stamp else ""
        items = response.get("items") or {}
        for item in ITEMS:
            answers = items.get(item["kod"]) or {}
            rows.append({
                "szakerto_id": response["expert_code"],
                "datum": datum,
                "tetel": item["kod"],
                "irany": answers.get("irany") or "",
                "p_irany": answers.get("p_irany") if answers.get("p_irany") is not None else "",
                "siker_A": answers.get("siker_A") if answers.get("siker_A") is not None else "",
                "siker_B": answers.get("siker_B") if answers.get("siker_B") is not None else "",
                "kulonbseg_min": answers.get("kulonbseg_min") if answers.get("kulonbseg_min") is not None else "",
                "kulonbseg_max": answers.get("kulonbseg_max") if answers.get("kulonbseg_max") is not None else "",
                "alak": derive_alak(item, answers) if answers.get("irany") else "",
                "mechanizmus": "; ".join(answers.get("mechanizmus") or []),
                "kuszob": answers.get("kuszob") or "",
                "megjegyzes": item_note(item, answers),
            })
    return rows


def background_rows(responses):
    rows = []
    for response in responses:
        stamp = response.get("submitted_at") or response.get("updated_at")
        datum = format_stamp(stamp, "%Y-%m-%d") if stamp else ""
        background = response.get("background") or {}
        calibration = response.get("calibration") or {}
        closing = response.get("closing") or {}
        notes = []
        for key, label in (
            ("felso_vagy_also", "B3"), ("felso_also_arany", "B3 arány"), ("kizaro_kepletek", "B4"),
            ("regi_fogsor", "B5"), ("regi_fogsor_megjegyzes", "B5 megj."),
        ):
            if calibration.get(key):
                notes.append(f"{label}: {calibration[key]}")
        for key, label in (
            ("legrosszabb_kombinacio", "D2"), ("ellensulyozo", "D3"), ("onertekeles", "D5"), ("megjegyzes", "megj."),
        ):
            if closing.get(key):
                notes.append(f"{label}: {closing[key]}")
        for key, label in (("diploma_ev", "diploma"), ("tevekenyseg", "tevékenység"), ("szakvizsga", "szakvizsga"), ("evi_fogsorok", "fogsor/év")):
            if background.get(key) not in (None, ""):
                notes.append(f"{label}: {background[key]}")
        rows.append({
            "szakerto_id": response["expert_code"],
            "datum": datum,
            "evek_gyakorlat": background.get("evek_gyakorlat", ""),
            "fogsorok_szama_kat": background.get("fogsorok_szama_kat", "") or "",
            "oktat": background.get("oktat", "") or "",
            "alap_siker_100": calibration.get("alap_siker_100", "") if calibration.get("alap_siker_100") is not None else "",
            "anatomia_sulya_pct": calibration.get("anatomia_sulya_pct", "") if calibration.get("anatomia_sulya_pct") is not None else "",
            **{f"rang_{i}": closing.get(f"rang_{i}", "") or "" for i in range(1, 6)},
            "hianyzo_kepletek": closing.get("hianyzo_kepletek", "") or "",
            "megjegyzes": " | ".join(notes),
            "nev": response.get("expert_name") or "",
            "intezmeny": response.get("expert_affiliation") or "",
        })
    return rows


def item_tally(responses):
    """Tételenkénti gyorsösszesítés az admin-nézethez (beküldött válaszok)."""
    tally = []
    for item in ITEMS:
        counts = {key: 0 for key in ("A_kedvezotlenebb", "B_kedvezotlenebb", "nem_monoton", "nincs_kulonbseg", "nem_tudom")}
        certainties = []
        differences = []
        for response in responses:
            answers = (response.get("items") or {}).get(item["kod"]) or {}
            irany = answers.get("irany")
            if irany in counts:
                counts[irany] += 1
            if irany in DIRECTIONAL and answers.get("p_irany") is not None:
                certainties.append(answers["p_irany"])
            if answers.get("siker_A") is not None and answers.get("siker_B") is not None:
                differences.append(abs(answers["siker_A"] - answers["siker_B"]))
        tally.append({
            "kod": item["kod"], "nev": item["nev"], "counts": counts,
            "p_irany_atlag": round(sum(certainties) / len(certainties), 1) if certainties else None,
            "kulonbseg_atlag": round(sum(differences) / len(differences), 1) if differences else None,
            "n_valasz": sum(counts.values()),
        })
    return tally


def write_csv(rows, columns):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return "\ufeff" + buffer.getvalue()


def create_expert_blueprint(connection_factory):
    bp = Blueprint("expert", __name__, url_prefix="/expert")

    # -- munkamenet, CSRF, fejlécek -------------------------------------------
    def csrf_token():
        token = session.get("expert_csrf")
        if not token:
            token = secrets.token_urlsafe(32)
            session["expert_csrf"] = token
        return token

    def validate_csrf():
        expected = session.get("expert_csrf", "")
        supplied = request.form.get("csrf_token", "")
        if not expected or not secrets.compare_digest(expected, supplied):
            abort(400, description="Érvénytelen vagy lejárt űrlap. Töltsd újra az oldalt.")

    @bp.context_processor
    def inject_helpers():
        return {
            "expert_csrf_token": csrf_token,
            "expert_code_required": access_code_required,
            "expert_items": ITEMS,
            "irany_labels": IRANY_LABELS,
            "p_irany_values": P_IRANY_VALUES,
            "mechanisms": MECHANISMS,
            "item_codes": ITEM_CODES,
        }

    @bp.after_request
    def protect(response):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    def access_code_required():
        """A szakértői oldal alapból nyitott (a link elég); kód csak akkor kell,
        ha az üzemeltető EXPERT_ACCESS_CODE-ot állít be."""
        return bool(os.getenv("EXPERT_ACCESS_CODE"))

    def require_expert(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if access_code_required() and not session.get("expert_authenticated"):
                return redirect(url_for("expert.login", next=request.full_path))
            return view(*args, **kwargs)

        return wrapped

    def require_admin(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if not session.get("followup_authenticated"):
                return redirect(url_for("followup.login", next=request.full_path))
            return view(*args, **kwargs)

        return wrapped

    # -- adatbázis ---------------------------------------------------------------
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
                to_regclass('public.expert_prior_responses') AS responses_table,
                EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'expert_prior_responses'
                      AND column_name = 'expert_name'
                ) AS name_column
            """
        )
        return bool(result and result[0]["responses_table"] and result[0]["name_column"])

    def require_schema():
        if not schema_ready():
            return render_template("expert_setup.html", migration_missing=True), 503
        return None

    RESPONSE_COLUMNS = (
        "id, expert_code, expert_name, expert_affiliation, token, status, consent_confirmed, "
        "background, calibration, items, closing, form_version, submitted_at, created_at, updated_at"
    )

    def get_response_by_token(token):
        found = rows(f"SELECT {RESPONSE_COLUMNS} FROM expert_prior_responses WHERE token = %s", [token])
        return found[0] if found else None

    def get_response_by_id(response_id):
        found = rows(f"SELECT {RESPONSE_COLUMNS} FROM expert_prior_responses WHERE id = %s", [response_id])
        return found[0] if found else None

    def list_responses(status=None):
        sql = f"SELECT {RESPONSE_COLUMNS} FROM expert_prior_responses"
        params = []
        if status:
            sql += " WHERE status = %s"
            params.append(status)
        sql += " ORDER BY id"
        return rows(sql, params)

    def decorate(response):
        response = dict(response)
        for key in ("background", "calibration", "items", "closing"):
            value = response.get(key)
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except ValueError:
                    value = {}
            response[key] = value or {}
        response["items_answered"] = item_progress(response["items"])
        response["items_total"] = len(ITEMS)
        for key in ("created_at", "updated_at", "submitted_at"):
            response[f"{key}_label"] = format_stamp(response.get(key)) if response.get(key) else ""
        return response

    # -- szakértői oldalak ---------------------------------------------------------
    @bp.route("/login", methods=["GET", "POST"])
    def login():
        if not access_code_required():
            return redirect(url_for("expert.start"))
        next_url = request.values.get("next", "")
        parsed = urlsplit(next_url)
        if not next_url.startswith("/") or parsed.netloc or next_url.startswith("//"):
            next_url = ""
        if request.method == "POST":
            validate_csrf()
            configured = os.getenv("EXPERT_ACCESS_CODE")
            supplied = request.form.get("access_code", "")
            if configured and secrets.compare_digest(configured, supplied):
                session["expert_authenticated"] = True
                return redirect(next_url or url_for("expert.start"))
            flash("Hibás hozzáférési kód.", "error")
        return render_template("expert_login.html", next_url=next_url)

    @bp.post("/logout")
    def logout():
        validate_csrf()
        session.pop("expert_authenticated", None)
        session.pop("expert_token", None)
        return redirect(url_for("expert.login"))

    @bp.get("")
    @require_expert
    def start():
        setup_response = require_schema()
        if setup_response:
            return setup_response
        current = None
        token = session.get("expert_token")
        if token:
            current = get_response_by_token(token)
            if current:
                current = decorate(current)
        return render_template("expert_start.html", current=current)

    @bp.post("/start")
    @require_expert
    def start_response():
        validate_csrf()
        setup_response = require_schema()
        if setup_response:
            return setup_response
        expert_name = _clean_text(request.form.get("expert_name"), 200)
        expert_affiliation = _clean_text(request.form.get("expert_affiliation"), 200) or None
        if request.form.get("consent") != "on":
            flash("A kitöltés megkezdéséhez a hozzájárulást meg kell jelölni.", "error")
            return redirect(url_for("expert.start"))
        if not expert_name:
            flash("A kitöltés megkezdéséhez a szakértő nevét meg kell adni.", "error")
            return redirect(url_for("expert.start"))
        token = secrets.token_urlsafe(24)
        conn = connection_factory()
        try:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO expert_prior_responses
                        (expert_code, expert_name, expert_affiliation, token, consent_confirmed, form_version)
                    VALUES (%s, %s, %s, %s, TRUE, %s)
                    RETURNING id
                    """,
                    ["SZ-új", expert_name, expert_affiliation, token, FORM_VERSION],
                )
                new_id = cursor.fetchone()[0]
                cursor.execute(
                    "UPDATE expert_prior_responses SET expert_code = %s WHERE id = %s",
                    [f"SZ{int(new_id):02d}", new_id],
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        session["expert_token"] = token
        return redirect(url_for("expert.form", token=token))

    @bp.get("/urlap/<token>")
    @require_expert
    def form(token):
        setup_response = require_schema()
        if setup_response:
            return setup_response
        response = get_response_by_token(token)
        if response is None:
            abort(404)
        response = decorate(response)
        session["expert_token"] = token
        if response["status"] == "submitted":
            return render_template("expert_view.html", response=response, admin=False)
        return render_template("expert_form.html", response=response, errors=[])

    @bp.post("/urlap/<token>/mentes")
    @require_expert
    def autosave(token):
        validate_csrf()
        response = get_response_by_token(token)
        if response is None:
            return jsonify({"ok": False, "error": "Ismeretlen kitöltés."}), 404
        if response["status"] != "draft":
            return jsonify({"ok": False, "error": "A kitöltés már beküldve."}), 409
        name = request.form.get("field", "")
        value = request.form.get("value", "")
        try:
            section, item_code, field = parse_field_name(name)
            if section == "item":
                key, cleaned = parse_item_field(item_code, field, value)
                statement = (
                    """
                    UPDATE expert_prior_responses
                    SET items = jsonb_set(items, ARRAY[%s], COALESCE(items -> %s, '{}'::jsonb) || %s::jsonb),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE token = %s AND status = 'draft'
                    """,
                    [item_code, item_code, Json({key: cleaned}), token],
                )
            else:
                spec = SECTION_SPECS[section].get(field)
                if spec is None:
                    raise FieldError("Ismeretlen mező.")
                cleaned = parse_scalar(spec, value)
                column = SECTION_COLUMNS[section]
                statement = (
                    f"""
                    UPDATE expert_prior_responses
                    SET {column} = {column} || %s::jsonb, updated_at = CURRENT_TIMESTAMP
                    WHERE token = %s AND status = 'draft'
                    """,
                    [Json({field: cleaned}), token],
                )
        except FieldError as err:
            return jsonify({"ok": False, "error": str(err)}), 400
        execute_transaction([statement])
        return jsonify({"ok": True})

    @bp.post("/urlap/<token>/bekuldes")
    @require_expert
    def submit(token):
        validate_csrf()
        response = get_response_by_token(token)
        if response is None:
            abort(404)
        if response["status"] != "draft":
            return redirect(url_for("expert.form", token=token))
        data, errors = collect_form(request.form)
        problems = completeness_errors(data)
        if errors or problems:
            merged = decorate({**response, **{k: data[k] for k in ("background", "calibration", "closing", "items")}})
            flash("A kitöltés még nem küldhető be: nézd át a hiányzó vagy hibás mezőket.", "error")
            return render_template("expert_form.html", response=merged, errors=errors + problems), 400
        execute_transaction([
            (
                """
                UPDATE expert_prior_responses
                SET background = %s::jsonb, calibration = %s::jsonb, items = %s::jsonb, closing = %s::jsonb,
                    status = 'submitted', submitted_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE token = %s AND status = 'draft'
                """,
                [Json(data["background"]), Json(data["calibration"]), Json(data["items"]), Json(data["closing"]), token],
            )
        ])
        session.pop("expert_token", None)
        return redirect(url_for("expert.done", token=token))

    @bp.get("/kesz/<token>")
    @require_expert
    def done(token):
        response = get_response_by_token(token)
        if response is None:
            abort(404)
        return render_template("expert_done.html", response=decorate(response))

    # -- vizsgálatvezetői (admin) oldalak ---------------------------------------------
    @bp.get("/admin")
    @require_admin
    def admin():
        setup_response = require_schema()
        if setup_response:
            return setup_response
        responses = [decorate(row) for row in list_responses()]
        submitted = [row for row in responses if row["status"] == "submitted"]
        return render_template(
            "expert_admin.html",
            responses=responses,
            submitted_count=len(submitted),
            tally=item_tally(submitted),
        )

    @bp.get("/admin/<int:response_id>")
    @require_admin
    def admin_view(response_id):
        response = get_response_by_id(response_id)
        if response is None:
            abort(404)
        return render_template("expert_view.html", response=decorate(response), admin=True)

    @bp.post("/admin/<int:response_id>/torles")
    @require_admin
    def admin_delete(response_id):
        validate_csrf()
        response = get_response_by_id(response_id)
        if response is None:
            abort(404)
        if request.form.get("confirm") != response["expert_code"]:
            flash("A törléshez írd be a szakértő kódját megerősítésül.", "error")
            return redirect(url_for("expert.admin_view", response_id=response_id))
        execute_transaction([("DELETE FROM expert_prior_responses WHERE id = %s", [response_id])])
        flash(f"{response['expert_code']} törölve.", "success")
        return redirect(url_for("expert.admin"))

    def csv_response(rows_, columns, filename):
        return Response(
            write_csv(rows_, columns),
            mimetype="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @bp.get("/admin/export/priorok.csv")
    @require_admin
    def export_priors():
        include_drafts = request.args.get("status") == "all"
        responses = [decorate(row) for row in list_responses(None if include_drafts else "submitted")]
        return csv_response(prior_rows(responses), PRIOR_CSV_COLUMNS, "predict_expert_priorok.csv")

    @bp.get("/admin/export/hatter.csv")
    @require_admin
    def export_background():
        include_drafts = request.args.get("status") == "all"
        responses = [decorate(row) for row in list_responses(None if include_drafts else "submitted")]
        return csv_response(background_rows(responses), BACKGROUND_CSV_COLUMNS, "predict_expert_hatter.csv")

    return bp
