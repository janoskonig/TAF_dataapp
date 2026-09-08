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
import smtplib
import uuid
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid
from datetime import datetime, timedelta, timezone
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

from expert_texts import (IRANY_LABELS as IRANY_LABELS_BY_LANG, ITEM_TEXT_EN, JAW_LABELS, LANGS, ROLE_LABELS, ROLES,
                          STUDY_ACRONYM, UI, ui_texts)

FORM_VERSION = "v2.0"   # v2.0 (2026-09-06): pólusonkénti tartomány, konkrét mérési pólusok, konzisztencia-ellenőrzés
# A simulate_expert_responses.py ezzel a verziójelöléssel szúr be próbasorokat;
# a listában jelölve jelennek meg, az összesítésből és az exportból alapból kimaradnak.
SIM_VERSION = "v1.0-SZIMULACIO"
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


# A kitöltési határidő nem szabad szöveg: a felkérő levél küldésének napjától
# számított EXPERT_DEADLINE_DAYS (alapból 14) nap; ISO-dátumként tárolódik, a
# levélben a nyelv szerint formázva. Az emlékeztető ugyanezt a dátumot ismétli.
DEADLINE_DAYS = int(os.getenv("EXPERT_DEADLINE_DAYS", "14") or 14)
HU_MONTHS = ["január", "február", "március", "április", "május", "június", "július", "augusztus", "szeptember", "október", "november", "december"]
EN_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]


def deadline_iso(days=None, now=None):
    base = (now or datetime.now(LOCAL_TZ)).date()
    return (base + timedelta(days=DEADLINE_DAYS if days is None else days)).isoformat()


def format_deadline(value, lang="hu"):
    """ISO-dátum → „2026. szeptember 20” (magyar, a -ig rag a levélben) / „20 September 2026”;
    a nem ISO (régi, szabad szöveges) érték változatlanul marad."""
    text = str(value or "").strip()
    try:
        day = datetime.strptime(text[:10], "%Y-%m-%d").date()
    except ValueError:
        return text
    if lang == "en":
        return f"{day.day} {EN_MONTHS[day.month - 1]} {day.year}"
    return f"{day.year}. {HU_MONTHS[day.month - 1]} {day.day}"


# A meghívottak maguk is ajánlhatnak kollégát (hólabda-toborzás): a rendszer
# személyes linkkel küldi a felkérést, az ajánló kódját a háttérben rögzíti.
REFERRAL_LIMIT = int(os.getenv("EXPERT_REFERRAL_LIMIT", "3") or 3)


def referrals_enabled():
    return os.getenv("EXPERT_REFERRALS", "1") != "0"

# ---------------------------------------------------------------------------
# Tételregiszter — a kódok azonosak az R-szkript regiszterével
# ---------------------------------------------------------------------------
ITEMS = [
    {
        "kod": "F1", "nev": "A felső gerinc magassága", "jaw": "Felső állcsont",
        "rogzit": "A gipszmintán mérjük, milyen magas a felső állcsontgerinc.",
        "A": "magas, jól megtartott gerinc (a mintán kb. 10 mm)", "B": "alacsony, sorvadt gerinc (a mintán kb. 5 mm)", "alak": "monoton", "optimum": False,
        "kuszob": "Milyen magasság alatt mondaná, hogy a felső gerinc alacsony? (mm)", "subs": [],
    },
    {
        "kod": "F2", "nev": "Alámenős területek a felső állcsonton", "jaw": "Felső állcsont",
        "rogzit": "A gipszmintán mérjük, mennyi alámenős terület van a felső állcsonton.",
        "A": "kevés vagy alig van alámenősség", "M": "közepes alámenősség", "B": "nagy, kifejezett alámenősség", "alak": "optimum", "optimum": True,
        "kuszob": "Ha a közepes a legjobb: nagyjából milyen alámenősség az ideális? (pár szóval)", "subs": [],
    },
    {
        "kod": "F3", "nev": "A szájpad magassága", "jaw": "Felső állcsont",
        "rogzit": "A gipszmintán mérjük a szájpadboltozat magasságát.",
        "A": "magas, boltozatos szájpad (kb. 25 mm)", "B": "lapos szájpad (kb. 17 mm)", "alak": "monoton", "optimum": False,
        "kuszob": "Milyen magasság alatt mondaná, hogy a szájpad lapos? (mm)", "subs": [],
    },
    {
        "kod": "F4", "nev": "A felső gerincív alakja", "jaw": "Felső állcsont",
        "rogzit": "A gipszmintán mérjük: a tuber – locus caninus – papilla incisiva pontok által bezárt szög, mindkét oldalon, a két oldal átlaga. A nagyobb szög szögletesebb, széles, a kisebb hegyesebb, V-alakú ívet jelent.",
        "A": "szögletes, széles ív (kb. 140°)", "B": "hegyes, V-alakú ív (kb. 125°)", "alak": "kuszobos", "optimum": False,
        "kuszob": "Van-e olyan szög, amely felett már nincs további előny? (°)", "subs": [],
    },
    {
        "kod": "F5", "nev": "Lötyögő gerinc a felső állcsonton", "jaw": "Felső állcsont",
        "rogzit": "Vizsgálatkor nézzük: nincs; lötyögő tuberek; lötyögő frontális gerinc.",
        "A": "nincs lötyögő gerinc", "B": "van lötyögő gerinc", "alak": "monoton", "optimum": False,
        "kuszob": None,
        "subs": [{"kod": "lokalizacio", "kerdes": "Ha van: melyik a rosszabb?",
                  "opciok": [("frontalis", "a frontális gerincen"), ("tuberalis", "a tubereken"), ("egyforma", "egyforma"), ("nem_tudom", "nem tudom")]}],
    },
    {
        "kod": "F6", "nev": "A felső és az alsó gerincél egymáshoz viszonyított helyzete", "jaw": "Felső állcsont",
        "rogzit": "A gipszmintán mérjük, hogy a felső és az alsó gerincél vonala függőlegesen egymás felett van-e (90°), vagy eltér egymástól.",
        "A": "a két gerincél egymás felett van (eltérés legfeljebb 5°)", "M": "közepes eltérés (kb. 10°)", "B": "nagy eltérés (20° vagy több)", "alak": "optimum", "optimum": True,
        "kuszob": "Mekkora eltérést tart már jelentősnek? (°)",
        "subs": [{"kod": "irany_szamit", "kerdes": "Számít-e, hogy melyik irányban tér el (az alsó szélesebb, vagy a felső)?",
                  "opciok": [("igen", "igen"), ("nem", "nem"), ("nem_tudom", "nem tudom")]}],
    },
    {
        "kod": "F7", "nev": "Torus palatinus", "jaw": "Felső állcsont",
        "rogzit": "Vizsgálatkor nézzük: nincs; plató alakú; orsó alakú.",
        "A": "nincs torus", "B": "van torus (plató vagy orsó alakú)", "alak": "monoton", "optimum": False,
        "kuszob": None,
        "subs": [{"kod": "alak_rosszabb", "kerdes": "Ha van: melyik alak a rosszabb?",
                  "opciok": [("orso", "az orsó alakú"), ("plato", "a plató alakú"), ("egyforma", "egyforma"), ("nem_tudom", "nem tudom")]}],
    },
    {
        "kod": "F8", "nev": "Mi van a felső fogsorral szemben (antagonista)", "jaw": "Felső állcsont",
        "rogzit": "Vizsgálatkor nézzük: nincs, most készül; kivehető fogpótlás (teljes fogsor, overdenture, fémlemezes); megtartott saját fogak vagy rögzített pótlás.",
        "A": "kivehető fogpótlás a szemközti állcsonton", "B": "megtartott saját fogak vagy rögzített pótlás", "alak": "interakcio", "optimum": False,
        "kuszob": None,
        "subs": [{"kod": "kategoria_vagy_interakcio", "kerdes": "Ön szerint maga az antagonista fajtája számít, vagy inkább az, hogy milyen irányból jönnek az erők (például sorvadt felső állcsontnál erőltetett ollóharapás)?",
                  "opciok": [("kategoria", "az antagonista fajtája"), ("interakcio", "az erők iránya"), ("mindketto", "mindkettő"), ("nem_tudom", "nem tudom")]}],
    },
    {
        "kod": "A1", "nev": "Az alsó gerinc alakja és magassága (Kaán szerint)", "jaw": "Alsó állcsont",
        "rogzit": "Vizsgálatkor öt fokozatot különböztetünk meg: (1) egészében megtartott; (2) elöl megtartott, oldalt lapos; (3) egészében lapos; (4) negatív; (5) mélyült negatív. A gipszmintán a gerinc magasságát is mérjük.",
        "A": "egészében megtartott gerinc (1)", "B": "mélyült negatív gerinc (5)", "alak": "telitodo", "optimum": False,
        "kuszob": "Milyen magasság alatt mondaná, hogy az alsó gerinc sorvadt? (mm)",
        "subs": [
            {"kod": "legnagyobb_ugras", "kerdes": "Melyik két fokozat között romlik a legtöbbet a kilátás?",
             "opciok": [("1_2", "1 és 2 között"), ("2_3", "2 és 3 között"), ("3_4", "3 és 4 között"), ("4_5", "4 és 5 között"), ("egyenletes", "nagyjából egyenletesen romlik")]},
            {"kod": "negy_ot_kulonbseg", "kerdes": "Van-e érdemi különbség a negatív és a mélyült negatív gerinc között?",
             "opciok": [("igen", "igen"), ("nem", "nem"), ("nem_tudom", "nem tudom")]},
        ],
    },
    {
        "kod": "A3", "nev": "A buccinator-tasak", "jaw": "Alsó állcsont",
        "rogzit": "Vizsgálatkor nézzük, mit csinál a tasak szájnyitáskor: beszűkül; kiszélesedik; vagy lebenyezett a felszíne.",
        "A": "szájnyitáskor beszűkülő tasak", "B": "szájnyitáskor kiszélesedő tasak", "alak": "nincs_irany", "optimum": False,
        "kuszob": None,
        "subs": [{"kod": "lebenyezett", "kerdes": "És a lebenyezett felszínű tasak?",
                  "opciok": [("legkedvezobb", "az a legjobb"), ("koztes", "köztes"), ("legkedvezotlenebb", "az a legrosszabb"), ("nem_tudom", "nem tudom")]}],
    },
    {
        "kod": "A4", "nev": "Torus mandibularis", "jaw": "Alsó állcsont",
        "rogzit": "Vizsgálatkor nézzük mindkét oldalon: nincs; kis méretű; nagy méretű.",
        "A": "nincs torus", "B": "van torus (kicsi vagy nagy)", "alak": "monoton", "optimum": False,
        "kuszob": None,
        "subs": [{"kod": "nagy_vs_kicsi", "kerdes": "Ha van: mennyivel rosszabb a nagy torus a kicsinél?",
                  "opciok": [("alig", "alig"), ("kozepesen", "közepesen"), ("sokkal", "sokkal"), ("nem_tudom", "nem tudom")]}],
    },
    {
        "kod": "A5", "nev": "A lingualis tasak nyelés közben", "jaw": "Alsó állcsont",
        "rogzit": "Vizsgálatkor az ujjunkat a tasakba tesszük, és a beteg nyel: az izmok az ujjat az állcsonthoz préselik; nem szűkítik a tasakot; vagy kifelé nyomják az ujjat.",
        "A": "az izmok az ujjat az állcsonthoz préselik", "B": "az izmok kifelé nyomják az ujjat", "alak": "monoton", "optimum": False,
        "kuszob": None,
        "subs": [{"kod": "nem_szukit_helye", "kerdes": "És ha az izmok nem szűkítik a tasakot?",
                  "opciok": [("A_kozel", "az inkább az A-hoz áll közel"), ("kozepen", "a kettő között van"), ("B_kozel", "az inkább a B-hez áll közel"), ("nem_tudom", "nem tudom")]}],
    },
    {
        "kod": "TUB", "nev": "A tuberculum alveolare mandibulae", "jaw": "Alsó állcsont",
        "rogzit": "Négy dolgot nézünk: borítja-e feszes íny; milyen az alakja (fordított körte, kicsi, plicaszerű); milyen a dőlése a gerinchez képest; mozog-e szájnyitáskor.",
        "A": "feszes ínnyel fedett, jó alakú, mozdulatlan tuberculum", "B": "feszes ínnyel nem fedett, plicaszerű, mozgékony tuberculum", "alak": "monoton", "optimum": False,
        "kuszob": None,
        "subs": [{"kod": "legfontosabb", "kerdes": "A négy közül melyik számít a legtöbbet?",
                  "opciok": [("A6", "a feszes ínyborítás"), ("A7", "az alakja"), ("A8", "a dőlése"), ("A9", "hogy mozog-e"), ("nem_tudom", "nem tudom")]}],
    },
    {
        "kod": "A10", "nev": "Az állcsontok sagittális relációja", "jaw": "Alsó állcsont",
        "rogzit": "A gipszmintán mérjük, előrébb vagy hátrébb áll-e az alsó állcsont a felsőhöz képest; lényegében az Angle-osztályt.",
        "A": "Angle I., szabályos helyzet", "B": "Angle II. vagy III., eltérő helyzet", "alak": "nincs_irany", "optimum": False,
        "kuszob": None,
        "subs": [{"kod": "melyik_rosszabb", "kerdes": "Ha az eltérés rossz: melyik a rosszabb?",
                  "opciok": [("angle_II", "az Angle II."), ("angle_III", "az Angle III."), ("egyforma", "egyforma"), ("nem_tudom", "nem tudom")]}],
    },
    {
        "kod": "A11", "nev": "A szájfenék", "jaw": "Alsó állcsont",
        "rogzit": "Vizsgálatkor nézzük: nem elődomborodó; puhán elődomborodó; tömött, elődomborodó.",
        "A": "puhán elődomborodó szájfenék", "B": "tömött, elődomborodó szájfenék", "alak": "monoton", "optimum": False,
        "kuszob": None,
        "subs": [{"kod": "nem_elodomborodo_helye", "kerdes": "És a nem elődomborodó szájfenék?",
                  "opciok": [("A_kozel", "az inkább az A-hoz áll közel"), ("kozepen", "a kettő között van"), ("B_kozel", "az inkább a B-hez áll közel"), ("nem_tudom", "nem tudom")]}],
    },
    {
        "kod": "A12", "nev": "Spinae mentales", "jaw": "Alsó állcsont",
        "rogzit": "Vizsgálatkor nézzük: nem tapintható; tapintható; nyomásra érzékeny.",
        "A": "nem tapintható", "B": "tapintható vagy nyomásra érzékeny", "alak": "monoton", "optimum": False,
        "kuszob": None,
        "subs": [{"kod": "erzekeny_vs_tapinthato", "kerdes": "Ha tapintható: mennyivel rosszabb, ha nyomásra érzékeny is?",
                  "opciok": [("alig", "alig"), ("kozepesen", "közepesen"), ("sokkal", "sokkal"), ("nem_tudom", "nem tudom")]}],
    },
]


def localized_items(lang="hu"):
    """A tételregiszter a kért nyelven (a kódok és válaszkódok változatlanok)."""
    if lang != "en":
        return ITEMS
    out = []
    for item in ITEMS:
        text = ITEM_TEXT_EN.get(item["kod"], {})
        copy = dict(item)
        for key in ("nev", "rogzit", "A", "M", "B", "kuszob"):
            if key in text:
                copy[key] = text[key]
        copy["jaw"] = JAW_LABELS["en"].get(item["jaw"], item["jaw"])
        subs = []
        for sub in item["subs"]:
            sub_text = text.get("subs", {}).get(sub["kod"], {})
            labels = sub_text.get("opciok", {})
            subs.append({
                "kod": sub["kod"],
                "kerdes": sub_text.get("kerdes", sub["kerdes"]),
                "opciok": [(code, labels.get(code, label)) for code, label in sub["opciok"]],
            })
        copy["subs"] = subs
        out.append(copy)
    return out


ITEM_CODES = [item["kod"] for item in ITEMS]
ITEMS_BY_CODE = {item["kod"]: item for item in ITEMS}
UPPER_CODES = [item["kod"] for item in ITEMS if item["jaw"] == "Felső állcsont"]
LOWER_CODES = [item["kod"] for item in ITEMS if item["jaw"] == "Alsó állcsont"]
RANK_SLOTS = 3   # állcsontonként a három legfontosabb adottság
# A folytonos tételek pólusaihoz rendelt mérési értékek (a kérdés szövegében is
# szerepelnek). Az elemzés ezekkel váltja a pólusok közötti sikerkülönbséget
# egységnyi (mm-, fok-) hatássá; a predict_szakertoi_prior_logit.R regiszterével
# azonosnak kell lennie. A kohorsz mért eloszlásából (kb. alsó és felső ötöd)
# kerekített, a vizsgálatvezető által módosítható javaslatok.
POLE_VALUES = {
    "F1": {"A": 10.0, "B": 5.0, "unit": "mm"},
    "F3": {"A": 25.0, "B": 17.0, "unit": "mm"},
    "F4": {"A": 140.0, "B": 125.0, "unit": "°"},
    "F6": {"A": 2.5, "M": 10.0, "B": 20.0, "unit": "° eltérés 90°-tól"},
}

IRANY_VALUES = {"A_kedvezotlenebb", "B_kedvezotlenebb", "nem_monoton", "nincs_kulonbseg", "nem_tudom"}
IRANY_LABELS = IRANY_LABELS_BY_LANG["hu"]
DIRECTIONAL = {"A_kedvezotlenebb", "B_kedvezotlenebb"}
P_IRANY_VALUES = [50, 60, 70, 80, 90, 95, 99]
# A teljes lemezes fogpótlás helybentartó tényezőinek hagyományos felosztása,
# a klinikai-anatómiai és a fizikai tényezők átfedését feloldva: az anatómiai
# adottság a helybentartás fizikai útján (szívóhatás, nyálfilm-tapadás,
# mechanikai megkapaszkodás, felületnagyság), az állékonyságon vagy az
# alátámasztáson és a tűrésen keresztül hat. Járulékos tényezők (paszták)
# itt nem értelmezettek.
MECHANISM_GROUPS = [
    ("Helybentartás (retenció)", [
        ("szivohatas", "szívóhatás, szélzárás"),
        ("nyalfilm_tapadas", "tapadás a nyálfilmen át (adhézió, kapillárishatás)"),
        ("alamenos_megkapaszkodas", "megkapaszkodás alámenős képleten"),
        ("illeszkedo_felulet", "az illeszkedő felület nagysága"),
    ]),
    ("Állékonyság (stabilitás)", [
        ("izomegyensuly", "az izmok és a nyelv erőhatásai"),
        ("gerinc_vezetes", "a gerinc alakja, magassága (oldalirányú megvezetés)"),
        ("ragoero_irany", "az antagonista fogazat, a rágóerők iránya"),
    ]),
    ("Alátámasztás és tűrés", [
        ("nyomaseloszlas", "teherviselés, nyomáseloszlás"),
        ("fajdalom", "fájdalom, nyomásérzékenység, felfekvés"),
        ("tureskepesseg", "a beteg tűrőképessége (idegentest-érzés, öklendezés)"),
    ]),
]
MECHANISMS = [(code, label) for _, group in MECHANISM_GROUPS for code, label in group] + [("egyeb", "egyéb (írja a megjegyzésbe)")]
MECHANISM_LABELS = dict(MECHANISMS)
# Korábbi (2026-09-06 előtti) kódok, hogy a régi sorok is olvashatók maradjanak.
LEGACY_MECHANISM_CODES = {"retencio", "stabilitas", "alatamasztas", "technikai"}
MECHANISM_CODES = {code for code, _ in MECHANISMS} | LEGACY_MECHANISM_CODES
ITEM_TEXT_FIELDS = {"kuszob", "megjegyzes", "mechanizmus_egyeb"}
# Pólusonként pontbecslés és tartomány (húsz hasonló becslésből tizenkilenc ezen
# belül); az optimum alakú tételeknél a közepes (M) forgatókönyv is. A
# kulonbseg_min/max a v1 űrlap öröksége: olvasható marad, de már nem kérdezzük.
ITEM_INT_FIELDS = {
    "siker_A": (0, 100), "siker_A_min": (0, 100), "siker_A_max": (0, 100),
    "siker_B": (0, 100), "siker_B_min": (0, 100), "siker_B_max": (0, 100),
    "siker_M": (0, 100), "siker_M_min": (0, 100), "siker_M_max": (0, 100),
    # „nincs érdemi különbség”: egyetlen közös szám (K) mindkét változatra
    "siker_K": (0, 100), "siker_K_min": (0, 100), "siker_K_max": (0, 100),
    "kulonbseg_min": (0, 100), "kulonbseg_max": (0, 100),
}
ITEM_BOOL_FIELDS = {"nagysag_nem_tudom"}
MAGNITUDE_ANSWERS = {"A_kedvezotlenebb", "B_kedvezotlenebb", "nem_monoton"}
NODIFF_TOLERANCE = 10   # „nincs érdemi különbség” mellett ennél nagyobb pontkülönbség ellentmondás

BACKGROUND_FIELDS = {
    "diploma_ev": ("int", 1950, 2100),
    "evek_gyakorlat": ("int", 0, 70),
    "fogsorok_szama_kat": ("choice", {"<100", "100-500", "500-1000", "1000-3000", ">3000"}),
    "oktat": ("choice", {"igen", "nem", "korabban"}),
    "tevekenyseg": ("choice", {"egyetemi", "maganpraxis", "mindketto", "egyeb"}),
    "szakvizsga": ("text", 200),
    "evi_fogsorok": ("int", 0, 2000),
    "nyelv": ("choice", set(LANGS)),
    # a kitöltő szerepe és a fogtechnikusi háttérkérdések (a fogorvosi diploma/szakvizsga helyett)
    "szerep": ("choice", set(ROLES)),
    "kepesites_ev": ("int", 1950, 2100),
    "mester": ("choice", {"igen", "nem"}),
    # fogtechnikus: miből tudja meg, hogyan vált be a fogsor (a becslés forrása)
    "visszajelzes": ("choice", {"rendszeres_fogorvosi", "visszakerulo_munkak", "kozvetlen_beteg", "egyeb"}),
}


def role_of(background):
    """A kitöltő szerepe a háttéradatokból; a régi (szerep nélküli) sorok fogorvosiak."""
    role = (background or {}).get("szerep")
    return role if role in ROLES else "fogorvos"
CALIBRATION_FIELDS = {
    # felkészítő gyakorlókérdések (SHELF-mintájú kalibrációs gyakorlat; nem kötelező)
    **{f"gyak_{i}_{part}": ("int", 0, 100000) for i in (1, 2, 3) for part in ("min", "pont", "max")},
    "gyak_irany": ("choice", {"duna", "rajna"}),
    "gyak_irany_p": ("choice", {str(v) for v in (50, 60, 70, 80, 90, 95, 99)}),
    "felkeszites_kesz": ("choice", {"1"}),
    "alap_siker_100": ("int", 0, 100),
    "anatomia_sulya_pct": ("int", 0, 100),
    "felso_vagy_also": ("choice", {"felso", "also", "egyforman"}),
    "felso_also_arany": ("text", 40),
    "kizaro_kepletek": ("text", 2000),
    "regi_fogsor": ("choice", {"alig", "kozepesen", "erosen"}),
    "regi_fogsor_megjegyzes": ("text", 1000),
}
CLOSING_FIELDS = {
    # v2.2: állcsontonként három rangsorhely; a v2.0-s összevont rang_1..5 olvasható marad
    **{f"rang_felso_{i}": ("choice", set(UPPER_CODES)) for i in range(1, RANK_SLOTS + 1)},
    **{f"rang_also_{i}": ("choice", set(LOWER_CODES)) for i in range(1, RANK_SLOTS + 1)},
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
    "kulonbseg_min", "kulonbseg_max", "alak", "mechanizmus", "kuszob", "megjegyzes", "szerep",
    "siker_A_min", "siker_A_max", "siker_B_min", "siker_B_max", "siker_M", "siker_M_min", "siker_M_max",
    "nagysag_nem_tudom", "polus_A_ertek", "polus_M_ertek", "polus_B_ertek", "polus_egyseg",
    "siker_K", "siker_K_min", "siker_K_max",
]
BACKGROUND_CSV_COLUMNS = [
    "szakerto_id", "datum", "evek_gyakorlat", "fogsorok_szama_kat", "oktat", "alap_siker_100",
    "anatomia_sulya_pct", "rang_1", "rang_2", "rang_3", "rang_4", "rang_5",
    "rang_felso_1", "rang_felso_2", "rang_felso_3", "rang_also_1", "rang_also_2", "rang_also_3", "hianyzo_kepletek", "megjegyzes",
    "nev", "intezmeny", "szerep", "visszajelzes", "ajanlo_kod",
]


class MailError(RuntimeError):
    """E-mail-küldési hiba (hiányzó beállítás vagy SendGrid-hiba)."""


MAIL_TEXTS = {
    "hu": {
        "subject": "Kérés a tapasztalatáról a teljes fogsor sikeréről (PREDICT-vizsgálat, kb. 30 perc)",
        "body": (
            "Tisztelt {name}!\n\n{referrer_sentence}"
            "A Semmelweis Egyetem Fogpótlástani Klinikáján a PREDICT-vizsgálatban ({acronym}) azt kutatjuk, mely anatómiai "
            "adottságok segítik, és melyek nehezítik a teljes lemezes fogsor sikerét. Elődeink és tanáraink ezt a tapasztalatukból "
            "tanították; mi most ezt a tapasztalati tudást szeretnénk összegyűjteni néhány, a teljes fogsor készítésében nagy "
            "gyakorlattal rendelkező kollégától, és összevetni a mért betegadatainkkal.\n\n"
            "A válaszokat Bayes-i statisztikai módszerrel dolgozzuk fel: a tapasztalt kollégák véleményét előzetes tudásként "
            "(priorként) építjük be a modellbe, és ezt frissítjük a mért betegadatokkal. Így kis betegszám mellett is értelmezhető "
            "eredményt kapunk, és láthatóvá válik, hol erősíti meg a mérés a klinikai tapasztalatot, és hol mond ellent neki.\n\n"
            "Ezért kérem Önt, hogy töltsön ki egy kérdőívet. Tizenhat anatómiai adottságról kérdezzük ugyanazt a néhány dolgot: melyik "
            "változat a rosszabb a fogsor sikere szempontjából, mennyire biztos ebben, és száz beteg közül hánynak lesz sikeres a fogsora "
            "az egyik és a másik esetben. A kérdőív egy rövid, ötperces felkészítővel kezdődik, amely a bizonytalanság megadásában segít. A kitöltés körülbelül 35–45 perc.\n\n"
            "Nincs jó vagy rossz válasz. Nem a tankönyvre vagyunk kíváncsiak, hanem arra, mit tanított Önnek a saját praxisa, akkor is, "
            "ha az eltér a tanultaktól. Kérem, egyedül töltse ki, és ne beszélje meg közben kollégákkal, mert éppen az egymástól független "
            "vélemények érdekelnek minket. Ha valamiben bizonytalan, jelölje azt; a bizonytalanság is fontos információ.\n\n"
            "A kérdőívet ezen a személyes linken éri el, belépési kód nélkül:\n{link}\n\n"
            "A válaszok maguktól mentődnek, a kitöltést bármikor megszakíthatja, és ugyanazon a számítógépen később folytathatja. "
            "A nevét csak én látom, hogy tudjam, kit kérdeztem meg; a válaszokat kóddal azonosítjuk, és kizárólag a többi kolléga "
            "válaszával együtt, összesítve használjuk fel. Egyéni válasz névvel nem kerül nyilvánosságra.\n\n"
            "{deadline_sentence}"
            "Ha bármi kérdése van, keressen bizalommal.\n\n"
            "Köszönöm az idejét és a tapasztalatát.\n\n"
            "Tisztelettel,\n{signature}"
        ),
        "deadline": "Hálás lennék, ha a kérdőívet két héten belül, {deadline}-ig ki tudná tölteni.\n\n",
        "deadline_reminder": "Hálás lennék, ha a kérdőívet {deadline}-ig ki tudná tölteni.\n\n",
        "referrer": "Erre a felmérésre {referrer} kolléga ajánlotta Önt.\n\n",
        "reminder_subject": "Emlékeztető: PREDICT szakértői kérdőív",
        "reminder_body": (
            "Tisztelt {name}!\n\n{referrer_sentence}"
            "Nemrég küldtem a PREDICT-vizsgálat szakértői kérdőívét a teljes fogsor sikerét befolyásoló anatómiai adottságokról. "
            "Ha már kitöltötte, köszönöm, és kérem, tekintse tárgytalannak ezt a levelet. Ha még nem jutott rá ideje, a személyes link "
            "továbbra is él, és a megkezdett kitöltés onnan folytatható, ahol abbahagyta:\n{link}\n\n"
            "{deadline_sentence}"
            "Köszönöm a segítségét.\n\n"
            "Tisztelettel,\n{signature}"
        ),
    },
    "en": {
        "subject": "A request for your experience on complete denture success (PREDICT study, about 30 minutes)",
        "body": (
            "Dear {name},\n\n{referrer_sentence}"
            "In the PREDICT study ({acronym}) at the Department of Prosthodontics, Semmelweis University, we are investigating "
            "which anatomical features help, and which hinder, the success of complete dentures. Our predecessors and teachers "
            "taught this from experience; we now want to collect this experiential knowledge from a small number of colleagues "
            "with extensive experience in complete denture treatment and compare it with our measured patient data.\n\n"
            "We analyse the answers with Bayesian statistical methods: the judgement of experienced colleagues enters the model as "
            "prior knowledge, which is then updated with the measured patient data. This yields interpretable results even with a "
            "small number of patients, and shows where the measurements confirm clinical experience and where they contradict it.\n\n"
            "I would therefore like to ask you to complete a questionnaire. For sixteen anatomical features we ask the same few things: "
            "which variant is worse for the success of the denture, how sure you are, and how many out of a hundred patients would have "
            "a successful denture in one case and in the other. The questionnaire opens with a short, five-minute preparation that helps with expressing uncertainty. Completing it takes about 35–45 minutes.\n\n"
            "There are no right or wrong answers. We are not asking about the textbook but about what your own practice has taught you, "
            "even where it differs from what you were taught. Please fill it in on your own, without discussing it with colleagues, "
            "because it is the independent opinions that we need. If you are unsure about something, mark that; uncertainty is valuable "
            "information too.\n\n"
            "You can reach the questionnaire through this personal link, no access code needed:\n{link}\n\n"
            "Your answers are saved automatically; you can stop at any time and continue later on the same computer. Only I see your "
            "name, so that I know whom I have asked; the answers are identified by a code and used solely pooled with the answers of "
            "the other colleagues. No individual answer is published with a name.\n\n"
            "{deadline_sentence}"
            "If you have any questions, please do not hesitate to contact me.\n\n"
            "Thank you for your time and your experience.\n\n"
            "Yours sincerely,\n{signature}"
        ),
        "deadline": "I would be grateful if you could complete the questionnaire within two weeks, by {deadline}.\n\n",
        "deadline_reminder": "I would be grateful if you could complete the questionnaire by {deadline}.\n\n",
        "referrer": "You were recommended for this survey by {referrer}.\n\n",
        "reminder_subject": "Reminder: PREDICT expert questionnaire",
        "reminder_body": (
            "Dear {name},\n\n{referrer_sentence}"
            "I recently sent you the PREDICT study's expert questionnaire on the anatomical features that influence the success of "
            "complete dentures. If you have already completed it, thank you, and please disregard this message. If you have not yet had "
            "time, your personal link is still active and a questionnaire in progress continues where you left off:\n{link}\n\n"
            "{deadline_sentence}"
            "Thank you for your help.\n\n"
            "Yours sincerely,\n{signature}"
        ),
    },
}


# A fogtechnikusi felkérés: ugyanaz a levél, a praxisra és a betegekre utaló
# mondatok a laborból látható tapasztalatra igazítva.
MAIL_ROLE_TEXTS = {
    "fogtechnikus": {
        "hu": {
            "subject": "Kérés a tapasztalatáról a teljes fogsor sikeréről (PREDICT-vizsgálat, fogtechnikus kollégáknak, kb. 30 perc)",
            "body": (
                "Tisztelt {name}!\n\n{referrer_sentence}"
                "A Semmelweis Egyetem Fogpótlástani Klinikáján a PREDICT-vizsgálatban ({acronym}) azt kutatjuk, mely anatómiai "
                "adottságok segítik, és melyek nehezítik a teljes lemezes fogsor sikerét. Elődeink és tanáraink ezt a tapasztalatukból "
                "tanították; mi most ezt a tapasztalati tudást szeretnénk összegyűjteni néhány, a teljes fogsor készítésében nagy "
                "gyakorlattal rendelkező fogorvos és fogtechnikus kollégától, és összevetni a mért betegadatainkkal. A fogtechnikus "
                "kollégák évente sokkal több teljes fogsort látnak a mintákon és a visszakerülő munkákban, mint egy-egy fogorvos, ezért "
                "az Ön tapasztalata külön értékes.\n\n"
                "A válaszokat Bayes-i statisztikai módszerrel dolgozzuk fel: a tapasztalt kollégák véleményét előzetes tudásként "
                "(priorként) építjük be a modellbe, és ezt frissítjük a mért betegadatokkal. Így kis betegszám mellett is értelmezhető "
                "eredményt kapunk, és láthatóvá válik, hol erősíti meg a mérés a gyakorlati tapasztalatot, és hol mond ellent neki.\n\n"
                "Ezért kérem Önt, hogy töltsön ki egy kérdőívet. Tizenhat anatómiai adottságról kérdezzük ugyanazt a néhány dolgot: melyik "
                "változat a rosszabb a fogsor sikere szempontjából, mennyire biztos ebben, és száz beteg közül hánynak lesz sikeres a fogsora "
                "az egyik és a másik esetben. Ha egy adottságot a mintáról, a laborból nem lehet megítélni, azt is jelölheti. A kérdőív egy rövid, ötperces felkészítővel kezdődik, amely a bizonytalanság megadásában segít. A kitöltés "
                "körülbelül 35–45 perc.\n\n"
                "Nincs jó vagy rossz válasz. Nem a tankönyvre vagyunk kíváncsiak, hanem arra, mit tanított Önnek a saját munkája, akkor is, "
                "ha az eltér a tanultaktól. Kérem, egyedül töltse ki, és ne beszélje meg közben kollégákkal, mert éppen az egymástól független "
                "vélemények érdekelnek minket. Ha valamiben bizonytalan, jelölje azt; a bizonytalanság is fontos információ.\n\n"
                "A kérdőívet ezen a személyes linken éri el, belépési kód nélkül:\n{link}\n\n"
                "A válaszok maguktól mentődnek, a kitöltést bármikor megszakíthatja, és ugyanazon a számítógépen később folytathatja. "
                "A nevét csak én látom, hogy tudjam, kit kérdeztem meg; a válaszokat kóddal azonosítjuk, és kizárólag a többi kolléga "
                "válaszával együtt, összesítve használjuk fel. Egyéni válasz névvel nem kerül nyilvánosságra.\n\n"
                "{deadline_sentence}"
                "Ha bármi kérdése van, keressen bizalommal.\n\n"
                "Köszönöm az idejét és a tapasztalatát.\n\n"
                "Tisztelettel,\n{signature}"
            ),
        },
        "en": {
            "subject": "A request for your experience on complete denture success (PREDICT study, for dental technicians, about 30 minutes)",
            "body": (
                "Dear {name},\n\n{referrer_sentence}"
                "In the PREDICT study ({acronym}) at the Department of Prosthodontics, Semmelweis University, we are investigating "
                "which anatomical features help, and which hinder, the success of complete dentures. Our predecessors and teachers "
                "taught this from experience; we now want to collect this experiential knowledge from a small number of dentists and "
                "dental technicians with extensive experience in complete denture work and compare it with our measured patient data. "
                "Dental technicians see far more complete dentures each year, on the casts and in the work that comes back, than any "
                "single dentist, which makes your experience particularly valuable.\n\n"
                "We analyse the answers with Bayesian statistical methods: the judgement of experienced colleagues enters the model as "
                "prior knowledge, which is then updated with the measured patient data. This yields interpretable results even with a "
                "small number of patients, and shows where the measurements confirm practical experience and where they contradict it.\n\n"
                "I would therefore like to ask you to complete a questionnaire. For sixteen anatomical features we ask the same few things: "
                "which variant is worse for the success of the denture, how sure you are, and how many out of a hundred patients would have "
                "a successful denture in one case and in the other. If a feature cannot be judged from the cast, in the laboratory, you can "
                "mark that too. The questionnaire opens with a short, five-minute preparation that helps with expressing uncertainty. Completing it takes about 35–45 minutes.\n\n"
                "There are no right or wrong answers. We are not asking about the textbook but about what your own work has taught you, "
                "even where it differs from what you were taught. Please fill it in on your own, without discussing it with colleagues, "
                "because it is the independent opinions that we need. If you are unsure about something, mark that; uncertainty is valuable "
                "information too.\n\n"
                "You can reach the questionnaire through this personal link, no access code needed:\n{link}\n\n"
                "Your answers are saved automatically; you can stop at any time and continue later on the same computer. Only I see your "
                "name, so that I know whom I have asked; the answers are identified by a code and used solely pooled with the answers of "
                "the other colleagues. No individual answer is published with a name.\n\n"
                "{deadline_sentence}"
                "If you have any questions, please do not hesitate to contact me.\n\n"
                "Thank you for your time and your experience.\n\n"
                "Yours sincerely,\n{signature}"
            ),
        },
    },
}


def _configured_sender_name():
    return (os.getenv("EMAIL_FROM_NAME") or os.getenv("SMTP_FROM_NAME") or os.getenv("EXPERT_MAIL_FROM_NAME")
            or "PREDICT-vizsgálat").strip()


def mail_from_name():
    """A feladó megjelenő neve; a PREDICT akkor is szerepel benne, ha a beállított név nem tartalmazza."""
    name = _configured_sender_name()
    return name if "predict" in name.lower() else f"{name} (PREDICT)"


def mail_signature():
    """A levél aláírása. A Render környezeti változóiban a sortörést a két karakteres
    '\\n' jelöli (a felület nem értelmezi), ezért azt itt valódi sortöréssé alakítjuk."""
    signature = os.getenv("EXPERT_MAIL_SIGNATURE") or _configured_sender_name()
    return signature.replace("\\n", "\n").replace("\r\n", "\n").strip()


def public_base_url():
    """A levelekbe és a meghívó-linkekbe kerülő állandó cím (APP_BASE_URL, pl.
    https://predict-study.hu). Ha nincs beállítva, a kérés hosztja marad, ami a
    Render-címről megnyitott admin oldalon onrender.com-os linket adna."""
    return (os.getenv("APP_BASE_URL") or os.getenv("PUBLIC_BASE_URL") or "").strip().rstrip("/")


def absolute_url(endpoint, **values):
    base = public_base_url()
    if base:
        return base + url_for(endpoint, **values)
    return url_for(endpoint, _external=True, **values)


def email_header_html(logo_url=None, partner_logo_url=None):
    """A levél fejléce: balra a PREDICT-logó, jobbra a Semmelweis Egyetem logója,
    táblázatban, hogy a levelezőprogramok egyformán rajzolják."""
    if not logo_url and not partner_logo_url:
        return ""
    left = f'<img src="{logo_url}" alt="PREDICT" width="112" style="width:112px;height:auto;display:block">' if logo_url else ""
    right = (f'<img src="{partner_logo_url}" alt="Semmelweis Egyetem" width="176" style="width:176px;height:auto;display:block">'
             if partner_logo_url else "")
    return ('<table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" '
            'style="max-width:560px;margin:0 0 22px;border-bottom:1px solid #dce7e9">'
            f'<tr><td align="left" valign="bottom" style="padding:0 0 14px">{left}</td>'
            f'<td align="right" valign="bottom" style="padding:0 0 14px">{right}</td></tr></table>')


def build_email(kind, lang, name, link, deadline=None, logo_url=None, partner_logo_url=None, role="fogorvos", referrer=None):
    """(subject, text, html) a meghívóhoz ('invite') vagy az emlékeztetőhöz ('reminder'); a szerep
    (fogorvos / fogtechnikus) a felkérés szövegét váltja; a határidő ISO-dátum, a nyelv szerint
    formázva; referrer: az ajánló kolléga neve (hólabda-meghívó)."""
    key = "en" if lang == "en" else "hu"
    texts = dict(MAIL_TEXTS[key])
    texts.update(MAIL_ROLE_TEXTS.get(role, {}).get(key, {}))
    deadline_key = "deadline_reminder" if kind == "reminder" else "deadline"
    deadline_sentence = texts[deadline_key].format(deadline=format_deadline(deadline, key)) if deadline else ""
    referrer_sentence = texts["referrer"].format(referrer=referrer) if (referrer and kind != "reminder") else ""
    body_key, subject_key = ("reminder_body", "reminder_subject") if kind == "reminder" else ("body", "subject")
    text = texts[body_key].format(name=name, link=link, deadline_sentence=deadline_sentence, signature=mail_signature(),
                                  referrer_sentence=referrer_sentence, acronym=STUDY_ACRONYM["en" if lang == "en" else "hu"])
    paragraphs = text.split("\n\n")
    html_parts = []
    for paragraph in paragraphs:
        escaped = (paragraph.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        escaped = escaped.replace(link, f'<a href="{link}">{link}</a>').replace("\n", "<br>")
        html_parts.append(f"<p>{escaped}</p>")
    logo = email_header_html(logo_url, partner_logo_url)
    html = '<div style="font-family: Georgia, serif; font-size: 15px; line-height: 1.5; color: #1f2933">' + logo + "".join(html_parts) + "</div>"
    return texts[subject_key], text, html


# SMTP-beállítások a ShadeMatch- és a maxillofaciális alkalmazással azonos
# környezeti változókból (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD vagy
# SMTP_PASS, SMTP_SENDER_EMAIL vagy SMTP_FROM, SMTP_USE_TLS, SMTP_USE_SSL,
# EMAIL_FROM_NAME vagy SMTP_FROM_NAME, EXPERT_MAIL_REPLY_TO vagy SMTP_REPLY_TO);
# SendGrid esetén SMTP_HOST=smtp.sendgrid.net, SMTP_USER=apikey, jelszó = API-kulcs.
# Ha a változók hiányoznak, a testvérprojekt .env-jét próbálja (helyi fejlesztés).
SIBLING_ENV_CANDIDATES = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "maxillofacialisrehabilitacio", ".env")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shadematch_python", ".env")),
]


def resolve_mail_settings():
    settings = {
        "host": os.getenv("SMTP_HOST", "").strip(),
        "port": int(os.getenv("SMTP_PORT", "587") or "587"),
        "user": os.getenv("SMTP_USER", "").strip(),
        "password": os.getenv("SMTP_PASSWORD", "").strip() or os.getenv("SMTP_PASS", "").strip(),
        "sender": os.getenv("SMTP_SENDER_EMAIL", "").strip() or os.getenv("SMTP_FROM", "").strip(),
        "use_tls": os.getenv("SMTP_USE_TLS", "true").strip().lower() != "false",
        "use_ssl": os.getenv("SMTP_USE_SSL", "false").strip().lower() == "true",
    }
    # A konigfogaszat-mintának megfelelően a SendGrid API-kulcs önmagában is elég
    # (SENDGRID_API_KEY vagy SendGridAPI_Key): ilyenkor az SMTP-relay adatai
    # adottak, csak a feladó címe kell. Az SMTP_PASS-ban a ${SendGridAPI_Key}
    # hivatkozást is feloldjuk.
    api_key = os.getenv("SENDGRID_API_KEY", "").strip() or os.getenv("SendGridAPI_Key", "").strip()
    if settings["password"] and api_key:
        for placeholder in ("${SendGridAPI_Key}", "${SENDGRID_API_KEY}"):
            settings["password"] = settings["password"].replace(placeholder, api_key)
    if not settings["password"] and api_key:
        settings["password"] = api_key
        settings["host"] = settings["host"] or "smtp.sendgrid.net"
        settings["user"] = settings["user"] or "apikey"
    if all([settings["host"], settings["user"], settings["password"], settings["sender"]]):
        return settings
    for candidate in SIBLING_ENV_CANDIDATES:
        if not os.path.exists(candidate):
            continue
        try:
            from dotenv import dotenv_values
        except ImportError:  # pragma: no cover
            break
        values = dotenv_values(candidate)
        settings["host"] = settings["host"] or (values.get("SMTP_HOST") or "").strip()
        settings["port"] = settings["port"] or int(values.get("SMTP_PORT") or 587)
        settings["user"] = settings["user"] or (values.get("SMTP_USER") or "").strip()
        settings["password"] = settings["password"] or ((values.get("SMTP_PASSWORD") or "").strip() or (values.get("SMTP_PASS") or "").strip())
        settings["sender"] = settings["sender"] or ((values.get("SMTP_SENDER_EMAIL") or "").strip() or (values.get("SMTP_FROM") or "").strip() or (values.get("SMTP_USER") or "").strip())
        if values.get("SMTP_USE_TLS") is not None:
            settings["use_tls"] = str(values.get("SMTP_USE_TLS")).strip().lower() != "false"
        if values.get("SMTP_USE_SSL") is not None:
            settings["use_ssl"] = str(values.get("SMTP_USE_SSL")).strip().lower() == "true"
        break
    return settings


def mail_configured():
    settings = resolve_mail_settings()
    return all([settings["host"], settings["user"], settings["password"], settings["sender"]])


def smtp_send(to_email, to_name, subject, text, html):
    """Egy levél elküldése SMTP-n (a ShadeMatch küldőjével azonos fejlécekkel); hiba esetén MailError."""
    settings = resolve_mail_settings()
    if not all([settings["host"], settings["user"], settings["password"], settings["sender"]]):
        raise MailError("Az e-mail-küldés nincs beállítva (SMTP_HOST, SMTP_USER, SMTP_PASSWORD és SMTP_SENDER_EMAIL szükséges).")
    sender = settings["sender"]
    domain = sender.split("@")[-1] if "@" in sender else "predict.local"
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = formataddr((mail_from_name(), sender))
    message["To"] = formataddr((to_name, to_email)) if to_name else to_email
    message["Reply-To"] = os.getenv("EXPERT_MAIL_REPLY_TO") or os.getenv("SMTP_REPLY_TO") or sender
    message["Date"] = formatdate(localtime=True)
    message["Message-ID"] = make_msgid("predict", domain=domain)
    message["X-Entity-Ref-ID"] = uuid.uuid4().hex
    message["Auto-Submitted"] = "auto-generated"
    message["X-Auto-Response-Suppress"] = "OOF, AutoReply"
    message.set_content(text)
    message.add_alternative(html, subtype="html")
    smtp_class = smtplib.SMTP_SSL if settings["use_ssl"] else smtplib.SMTP
    try:
        with smtp_class(settings["host"], settings["port"], timeout=20) as server:
            if settings["use_tls"] and not settings["use_ssl"]:
                server.starttls()
            server.login(settings["user"], settings["password"])
            server.send_message(message)
    except (smtplib.SMTPException, OSError) as err:
        raise MailError(f"Az SMTP-küldés nem sikerült: {err}") from err


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
    if field in ITEM_BOOL_FIELDS:
        if raw == "":
            return field, None
        if raw not in {"1", "0", "on", "off", "true", "false"}:
            raise FieldError("Nem megengedett érték.")
        return field, raw in {"1", "on", "true"}
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


def completeness_errors(data, lang="hu"):
    """A beküldés feltételei (hiánylista, emberi olvasásra, a kért nyelven)."""
    t = UI["en" if lang == "en" else "hu"]
    problems = []
    background = data.get("background") or {}
    calibration = data.get("calibration") or {}
    closing = data.get("closing") or {}
    items = data.get("items") or {}
    if background.get("evek_gyakorlat") is None:
        problems.append(t["err_evek"])
    if not background.get("fogsorok_szama_kat"):
        problems.append(t["err_fogsorok"])
    if calibration.get("alap_siker_100") is None:
        problems.append(t["err_b1"])
    if calibration.get("anatomia_sulya_pct") is None:
        problems.append(t["err_b2"])
    if role_of(background) == "fogtechnikus" and not background.get("visszajelzes"):
        problems.append(t["err_visszajelzes"])
    for item in localized_items(lang):
        answers = items.get(item["kod"]) or {}
        label = f"{item['kod']} · {item['nev']}"
        if not answers.get("irany"):
            problems.append(f"{label}: {t['err_direction']}")
        elif answers.get("irany") in DIRECTIONAL and answers.get("p_irany") is None:
            problems.append(f"{label}: {t['err_certainty']}")
        problems.extend(f"{label}: {text}" for text in consistency_errors(item, answers, t))
    if not closing.get("onertekeles"):
        problems.append(t["err_d5"])
    return problems


def consistency_errors(item, answers, t):
    """Egy tétel számszerű válaszainak belső ellentmondásai és hiányai (beküldést
    gátló szabályok; a szöveg a kért nyelven). A számok köre az iránytól függ:
    irányos válasz → A és B pólus; „a közepes a legjobb” → A, közepes, B;
    „nincs érdemi különbség” → egyetlen közös szám (K); „nem tudom megítélni” →
    számot nem kérünk, a beírt számokat figyelmen kívül hagyjuk."""
    problems = []
    irany = answers.get("irany")
    if not irany or irany == "nem_tudom":
        return problems
    unknown = bool(answers.get("nagysag_nem_tudom"))
    if irany == "nincs_kulonbseg":
        needed = ["K"]
    elif irany == "nem_monoton":
        needed = ["A", "M", "B"] if item.get("M") else ["A", "B"]
    else:
        needed = ["A", "B"]
    point = {pole: answers.get(f"siker_{pole}") for pole in ("A", "M", "B", "K")}
    lo = {pole: answers.get(f"siker_{pole}_min") for pole in point}
    hi = {pole: answers.get(f"siker_{pole}_max") for pole in point}
    for pole in needed:
        if lo[pole] is not None and hi[pole] is not None and lo[pole] > hi[pole]:
            problems.append(t["err_range_order"].format(pole=pole))
        if point[pole] is not None and lo[pole] is not None and point[pole] < lo[pole]:
            problems.append(t["err_range_contains"].format(pole=pole))
        if point[pole] is not None and hi[pole] is not None and point[pole] > hi[pole]:
            problems.append(t["err_range_contains"].format(pole=pole))
    if not unknown:
        if any(point[pole] is None for pole in needed):
            problems.append(t["err_magnitude_required"])
        elif any(lo[pole] is None or hi[pole] is None for pole in needed):
            problems.append(t["err_range_required"])
    if unknown:
        return problems
    a, b, m = point["A"], point["B"], point["M"]
    if a is not None and b is not None:
        if irany == "A_kedvezotlenebb" and a > b:
            problems.append(t["err_numbers_direction"].format(worse="B"))
        if irany == "B_kedvezotlenebb" and b > a:
            problems.append(t["err_numbers_direction"].format(worse="A"))
        if irany == "nincs_kulonbseg" and point["K"] is None and abs(a - b) >= NODIFF_TOLERANCE:
            problems.append(t["err_nodiff_numbers"].format(diff=abs(a - b)))   # v2.0-s (K nélküli) válaszok
        if irany == "nem_monoton" and m is not None and m < max(a, b):
            problems.append(t["err_optimum_numbers"])
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
            pole = POLE_VALUES.get(item["kod"], {})
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
                "szerep": role_of(response.get("background")),
                **{key: (answers.get(key) if answers.get(key) is not None else "")
                   for key in ("siker_A_min", "siker_A_max", "siker_B_min", "siker_B_max", "siker_M", "siker_M_min", "siker_M_max",
                               "siker_K", "siker_K_min", "siker_K_max")},
                "nagysag_nem_tudom": "1" if answers.get("nagysag_nem_tudom") else "",
                "polus_A_ertek": pole.get("A", ""), "polus_M_ertek": pole.get("M", ""), "polus_B_ertek": pole.get("B", ""),
                "polus_egyseg": pole.get("unit", ""),
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
        for key, label in (("diploma_ev", "diploma"), ("kepesites_ev", "képesítés"), ("mester", "mesterfogtechnikus"),
                           ("tevekenyseg", "tevékenység"), ("szakvizsga", "szakvizsga"), ("evi_fogsorok", "fogsor/év")):
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
            **{f"rang_{jaw}_{i}": closing.get(f"rang_{jaw}_{i}", "") or "" for jaw in ("felso", "also") for i in range(1, RANK_SLOTS + 1)},
            "hianyzo_kepletek": closing.get("hianyzo_kepletek", "") or "",
            "megjegyzes": " | ".join(notes),
            "nev": response.get("expert_name") or "",
            "intezmeny": response.get("expert_affiliation") or "",
            "szerep": role_of(background),
            "visszajelzes": background.get("visszajelzes", "") or "",
            "ajanlo_kod": background.get("ajanlo_kod", "") or "",
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


def create_expert_blueprint(connection_factory, mail_sender=None):
    bp = Blueprint("expert", __name__, url_prefix="/expert")
    send_mail = mail_sender or smtp_send

    def mail_available():
        """Van-e küldő: beadott küldő (teszt) vagy beállított SMTP."""
        return mail_sender is not None or mail_configured()

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

    def current_lang():
        requested = request.args.get("lang")
        if requested in LANGS:
            session["expert_lang"] = requested
        return session.get("expert_lang", "hu")

    @bp.context_processor
    def inject_helpers():
        lang = current_lang()
        items = localized_items(lang)
        return {
            "expert_csrf_token": csrf_token,
            "expert_code_required": access_code_required,
            "lang": lang,
            "t": ui_texts(lang, None),
            "expert_items": items,
            "irany_labels": IRANY_LABELS_BY_LANG[lang],
            "p_irany_values": P_IRANY_VALUES,
            "mechanism_labels": MECHANISM_LABELS,
            "item_codes": ITEM_CODES,
            "upper_codes": UPPER_CODES,
            "lower_codes": LOWER_CODES,
            "rank_slots": RANK_SLOTS,
            "item_names": {item["kod"]: item["nev"] for item in items},
            "role_labels": ROLE_LABELS[lang],
            "roles": ROLES,
            "pole_values": POLE_VALUES,
            "referrals_enabled": referrals_enabled,
            "referral_limit": REFERRAL_LIMIT,
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
                      AND column_name = 'invite_email'
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
        "background, calibration, items, closing, form_version, submitted_at, created_at, updated_at, "
        "invited_at, opened_at, invite_note, invite_email, invite_sent_at, reminder_sent_at, invite_deadline"
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

    def referral_count(expert_code):
        found = rows("SELECT COUNT(*) AS n FROM expert_prior_responses WHERE background->>'ajanlo_kod' = %s", [expert_code])
        return int(found[0]["n"]) if found else 0

    def email_invited(email):
        return bool(rows("SELECT id FROM expert_prior_responses WHERE lower(invite_email) = %s", [email.lower()]))

    def referral_context(response):
        """Az ajánló doboz adatai a záró és a válasz-oldalhoz (csak a saját, elfogadott kitöltésnél)."""
        if not referrals_enabled() or response["state"] == "invited" or session.get("expert_token") != response["token"]:
            return None
        return {"left": max(REFERRAL_LIMIT - referral_count(response["expert_code"]), 0), "limit": REFERRAL_LIMIT}

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
        response["role"] = role_of(response["background"])
        response["role_label"] = ROLE_LABELS["hu"][response["role"]]
        response["simulated"] = response.get("form_version") == SIM_VERSION
        response["invited"] = bool(response.get("invited_at"))
        if response.get("status") == "submitted":
            response["state"] = "submitted"
        elif response["invited"] and not response.get("consent_confirmed"):
            response["state"] = "invited"
        else:
            response["state"] = "draft"
        for key in ("created_at", "updated_at", "submitted_at", "invite_sent_at", "reminder_sent_at"):
            response[f"{key}_label"] = format_stamp(response.get(key)) if response.get(key) else ""
        return response

    def create_response(cursor, expert_name, expert_affiliation, lang, consent, invited, invite_note=None,
                        invite_email=None, invite_deadline=None, role="fogorvos", referrer=None):
        """Új kitöltés sora folytonos SZnn kóddal; a token a kitöltés kulcsa. referrer: {kod, nev} az ajánlóról."""
        token = secrets.token_urlsafe(24)
        background = {"nyelv": lang, "szerep": role}
        if referrer:
            background.update(ajanlo_kod=referrer.get("kod"), ajanlo_nev=referrer.get("nev"))
        cursor.execute(
            """
            INSERT INTO expert_prior_responses
                (expert_code, expert_name, expert_affiliation, token, consent_confirmed, form_version, background,
                 invited_at, invite_note, invite_email, invite_deadline)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, CASE WHEN %s THEN CURRENT_TIMESTAMP ELSE NULL END, %s, %s, %s)
            RETURNING id
            """,
            ["SZ-új", expert_name, expert_affiliation, token, consent, FORM_VERSION, Json(background), invited,
             invite_note, invite_email, invite_deadline],
        )
        new_id = cursor.fetchone()[0]
        cursor.execute(
            """
            SELECT COALESCE(MAX(CAST(SUBSTRING(expert_code FROM '^SZ([0-9]+)$') AS INTEGER)), 0) + 1
            FROM expert_prior_responses
            WHERE expert_code ~ '^SZ[0-9]+$'
            """
        )
        next_number = cursor.fetchone()[0]
        code = f"SZ{int(next_number):02d}"
        cursor.execute("UPDATE expert_prior_responses SET expert_code = %s WHERE id = %s", [code, new_id])
        return new_id, code, token

    # -- szakértői oldalak ---------------------------------------------------------
    @bp.get("/meghivo/<token>")
    def invite(token):
        """Személyes meghívó-link: kód nélkül, csak az adott kitöltésbe enged be."""
        setup_response = require_schema()
        if setup_response:
            return setup_response
        response = get_response_by_token(token)
        if response is None:
            lang = current_lang()
            return render_template("expert_invalid.html", message=UI[lang]["invite_invalid"]), 404
        response = decorate(response)
        session["expert_authenticated"] = True
        session["expert_token"] = token
        lang = (response["background"] or {}).get("nyelv")
        if lang in LANGS:
            session["expert_lang"] = lang
        if response["state"] == "submitted":
            return redirect(url_for("expert.form", token=token))
        if response["state"] == "draft":
            return redirect(url_for("expert.form", token=token))
        return redirect(url_for("expert.start"))

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
            flash(UI[current_lang()]["login_wrong"], "error")
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
        lang = current_lang()
        if current and current["state"] == "invited":
            return render_template("expert_start.html", current=current, t=ui_texts(lang, current["role"]), role=current["role"])
        return render_template("expert_start.html", current=current, role=None)

    @bp.post("/start")
    @require_expert
    def start_response():
        validate_csrf()
        setup_response = require_schema()
        if setup_response:
            return setup_response
        expert_name = _clean_text(request.form.get("expert_name"), 200)
        expert_affiliation = _clean_text(request.form.get("expert_affiliation"), 200) or None
        lang = current_lang()
        if request.form.get("consent") != "on":
            flash(UI[lang]["consent_missing"], "error")
            return redirect(url_for("expert.start"))
        if not expert_name:
            flash(UI[lang]["name_missing"], "error")
            return redirect(url_for("expert.start"))
        role = request.form.get("szerep") or "fogorvos"   # a régi (szerep nélküli) űrlap fogorvosi
        invite_token = request.form.get("invite_token", "")
        if invite_token and invite_token == session.get("expert_token"):
            # Meghívott kitöltés elfogadása: a sor már létezik, a hozzájárulást
            # és a (javítható) nevet rögzítjük, a kitöltés innentől piszkozat.
            # A szerep a meghívóból jön; az űrlapon javítható.
            role_patch = Json({"szerep": role}) if role in ROLES else Json({})
            execute_transaction([(
                """
                UPDATE expert_prior_responses
                SET consent_confirmed = TRUE, expert_name = %s, expert_affiliation = %s,
                    background = COALESCE(background, '{}'::jsonb) || %s::jsonb,
                    opened_at = COALESCE(opened_at, CURRENT_TIMESTAMP), updated_at = CURRENT_TIMESTAMP
                WHERE token = %s AND status = 'draft'
                """,
                [expert_name, expert_affiliation, role_patch, invite_token],
            )])
            return redirect(url_for("expert.form", token=invite_token))
        if role not in ROLES:
            flash(UI[lang]["role_q"], "error")
            return redirect(url_for("expert.start"))
        conn = connection_factory()
        try:
            with conn.cursor() as cursor:
                _, _, token = create_response(cursor, expert_name, expert_affiliation, lang, True, False, role=role)
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
        texts = ui_texts(current_lang(), response["role"])
        if response["status"] == "submitted":
            return render_template("expert_view.html", response=response, admin=False, t=texts, role=response["role"],
                                   referral=referral_context(response))
        if response["state"] == "invited":
            return redirect(url_for("expert.start"))
        if (response["calibration"] or {}).get("felkeszites_kesz") != "1":
            return redirect(url_for("expert.prep", token=token))
        return render_template("expert_form.html", response=response, errors=[], t=texts, role=response["role"],
                               nodiff_tolerance=NODIFF_TOLERANCE)

    @bp.get("/urlap/<token>/felkeszites")
    @require_expert
    def prep(token):
        """Felkészítő a valószínűségi ítéletekre (SHELF-mintájú): magyarázat, kidolgozott
        példa, három gyakorlókérdés azonnali visszajelzéssel. A kérdőív csak ezután nyílik."""
        response = get_response_by_token(token)
        if response is None:
            abort(404)
        response = decorate(response)
        session["expert_token"] = token
        if response["status"] == "submitted":
            return redirect(url_for("expert.form", token=token))
        if response["state"] == "invited":
            return redirect(url_for("expert.start"))
        return render_template("expert_prep.html", response=response, t=ui_texts(current_lang(), response["role"]),
                               role=response["role"], p_irany_values=P_IRANY_VALUES)

    @bp.post("/urlap/<token>/felkeszites/kesz")
    @require_expert
    def prep_done(token):
        validate_csrf()
        response = get_response_by_token(token)
        if response is None:
            abort(404)
        if response["status"] == "draft":
            execute_transaction([(
                """
                UPDATE expert_prior_responses
                SET calibration = COALESCE(calibration, '{}'::jsonb) || %s::jsonb, updated_at = CURRENT_TIMESTAMP
                WHERE token = %s AND status = 'draft'
                """,
                [Json({"felkeszites_kesz": "1"}), token],
            )])
        return redirect(url_for("expert.form", token=token))

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
        lang = current_lang()
        if "nyelv" not in data["background"]:
            data["background"]["nyelv"] = lang
        if data["background"].get("szerep") not in ROLES:
            data["background"]["szerep"] = role_of(decorate(response)["background"])
        problems = completeness_errors(data, lang)
        if errors or problems:
            merged = decorate({**response, **{k: data[k] for k in ("background", "calibration", "closing", "items")}})
            flash(UI[lang]["submit_incomplete"], "error")
            return render_template("expert_form.html", response=merged, errors=errors + problems,
                                   t=ui_texts(lang, merged["role"]), role=merged["role"], nodiff_tolerance=NODIFF_TOLERANCE), 400
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
        # a token a munkamenetben marad: a záró oldal és a saját válasz nézete így tudja, hogy
        # ugyanaz a kitöltő van jelen (ajánlás), és a kezdőoldal a beküldött állapotot mutatja
        return redirect(url_for("expert.done", token=token))

    @bp.get("/kesz/<token>")
    @require_expert
    def done(token):
        response = get_response_by_token(token)
        if response is None:
            abort(404)
        response = decorate(response)
        return render_template("expert_done.html", response=response, t=ui_texts(current_lang(), response["role"]),
                               role=response["role"], referral=referral_context(response))

    @bp.post("/urlap/<token>/ajanlas")
    @require_expert
    def refer(token):
        """Egy meghívott kolléga ajánl egy másikat: új meghívó-sor a saját kódjával az
        ajánló háttérben, személyes link, felkérő levél (ha a levélküldés be van állítva).
        Korlát: REFERRAL_LIMIT ajánlás / kitöltő, egy e-mail-cím csak egyszer."""
        validate_csrf()
        response = get_response_by_token(token)
        if response is None:
            abort(404)
        response = decorate(response)
        lang = current_lang()
        t = ui_texts(lang, response["role"])
        back = redirect(url_for("expert.done", token=token) if response["status"] == "submitted" else url_for("expert.form", token=token))
        if referral_context(response) is None:
            abort(403)
        name = _clean_text(request.form.get("ref_name"), 200)
        email = _clean_text(request.form.get("ref_email"), 200).lower()
        role = request.form.get("ref_role", "fogorvos")
        role = role if role in ROLES else "fogorvos"
        ref_lang = request.form.get("ref_lang", lang)
        ref_lang = ref_lang if ref_lang in LANGS else lang
        include_name = request.form.get("ref_include_name") == "on"
        if not name or "@" not in email or "." not in email.rsplit("@", 1)[-1]:
            flash(t["referral_invalid"], "error")
            return back
        if referral_count(response["expert_code"]) >= REFERRAL_LIMIT:
            flash(t["referral_limit"].format(n=REFERRAL_LIMIT), "error")
            return back
        if email_invited(email):
            flash(t["referral_exists"], "error")
            return back
        conn = connection_factory()
        try:
            with conn.cursor() as cursor:
                _, _, new_token = create_response(cursor, name, None, ref_lang, False, True, f"ajánlotta: {response['expert_code']}", email, None,
                                                  role=role, referrer={"kod": response["expert_code"], "nev": response.get("expert_name")})
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        link = absolute_url("expert.invite", token=new_token)
        if not mail_available():
            flash(t["referral_created_no_mail"].format(name=name), "success")
            return back
        sent = deliver(new_token, "invite", email, name, ref_lang, link, None, role,
                       referrer=(response.get("expert_name") if include_name else None), quiet=True)
        flash(t["referral_sent"].format(name=name) if sent else t["referral_mail_failed"], "success" if sent else "error")
        return back

    # -- vizsgálatvezetői (admin) oldalak ---------------------------------------------
    @bp.get("/admin")
    @require_admin
    def admin():
        setup_response = require_schema()
        if setup_response:
            return setup_response
        include_simulated = request.args.get("szimulacio") == "1"
        role_filter = request.args.get("szerep")
        if role_filter not in ROLES:
            role_filter = None
        responses = [decorate(row) for row in list_responses()]
        for row in responses:
            row["invite_link"] = absolute_url("expert.invite", token=row["token"])
        simulated_count = sum(1 for row in responses if row["simulated"])
        submitted_all = [row for row in responses if row["status"] == "submitted" and (include_simulated or not row["simulated"])]
        role_counts = {role: sum(1 for row in submitted_all if row["role"] == role) for role in ROLES}
        submitted = [row for row in submitted_all if role_filter is None or row["role"] == role_filter]
        return render_template(
            "expert_admin.html",
            responses=responses,
            submitted_count=len(submitted),
            invited_count=sum(1 for row in responses if row["state"] == "invited"),
            draft_count=sum(1 for row in responses if row["state"] == "draft"),
            new_code=request.args.get("uj"),
            mail_configured=mail_configured(),
            start_url=absolute_url("expert.start"),
            role_filter=role_filter,
            role_counts=role_counts,
            simulated_count=simulated_count,
            include_simulated=include_simulated,
            tally=item_tally(submitted),
        )

    @bp.post("/admin/meghivo")
    @require_admin
    def admin_invite():
        validate_csrf()
        setup_response = require_schema()
        if setup_response:
            return setup_response
        expert_name = _clean_text(request.form.get("expert_name"), 200)
        expert_affiliation = _clean_text(request.form.get("expert_affiliation"), 200) or None
        lang = request.form.get("lang", "hu")
        if lang not in LANGS:
            lang = "hu"
        role = request.form.get("szerep", "fogorvos")
        if role not in ROLES:
            role = "fogorvos"
        note = _clean_text(request.form.get("invite_note"), 500) or None
        email = _clean_text(request.form.get("invite_email"), 200) or None
        deadline = None   # a határidő a levél küldésekor áll be: küldés + DEADLINE_DAYS nap
        send_now = request.form.get("send_now") == "on"
        if not expert_name:
            flash("A meghívóhoz add meg a szakértő nevét.", "error")
            return redirect(url_for("expert.admin"))
        if email and "@" not in email:
            flash("Az e-mail-cím nem tűnik érvényesnek.", "error")
            return redirect(url_for("expert.admin"))
        conn = connection_factory()
        try:
            with conn.cursor() as cursor:
                _, code, token = create_response(cursor, expert_name, expert_affiliation, lang, False, True, note, email, deadline, role=role)
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        link = absolute_url("expert.invite", token=token)
        flash(f"Meghívó elkészült: {code} · {expert_name}. Link: {link}", "success")
        if send_now and email:
            deliver(token, "invite", email, expert_name, lang, link, deadline, role)
        return redirect(url_for("expert.admin", uj=code))

    @bp.post("/admin/meghivo/tomeges")
    @require_admin
    def admin_bulk_invite():
        """Több meghívó egyszerre: soronként név, munkahely, e-mail, szerep, nyelv. A hibás
        vagy már meghívott sorok kimaradnak (jelentéssel), a többi egy tranzakcióban jön
        létre, majd a levelek egyenként mennek ki; az eredmény soronként flash-üzenetben."""
        validate_csrf()
        setup_response = require_schema()
        if setup_response:
            return setup_response
        names = request.form.getlist("nev[]")
        affiliations = request.form.getlist("munkahely[]")
        emails = request.form.getlist("email[]")
        roles = request.form.getlist("szerep[]")
        langs = request.form.getlist("nyelv[]")
        send_now = request.form.get("send_now") == "on"
        pad = lambda values, n: list(values) + [""] * (n - len(values))
        n = max(len(names), len(emails))
        names, affiliations, emails, roles, langs = (pad(v, n) for v in (names, affiliations, emails, roles, langs))
        valid, report, seen = [], [], set()
        for index in range(n):
            name = _clean_text(names[index], 200)
            affiliation = _clean_text(affiliations[index], 200) or None
            email = _clean_text(emails[index], 200).lower()
            role = roles[index] if roles[index] in ROLES else "fogorvos"
            lang = langs[index] if langs[index] in LANGS else "hu"
            if not name and not email and not affiliation:
                continue
            if not name or "@" not in email or "." not in email.rsplit("@", 1)[-1]:
                report.append(f"{index + 1}. sor: kimaradt, hiányzik a név vagy hibás az e-mail-cím ({name or '–'}, {email or '–'}).")
                continue
            if email in seen or email_invited(email):
                report.append(f"{index + 1}. sor: {name} ({email}) már meghívva, kimaradt.")
                continue
            seen.add(email)
            valid.append((name, affiliation, email, role, lang))
        created = []
        if valid:
            conn = connection_factory()
            try:
                with conn.cursor() as cursor:
                    for name, affiliation, email, role, lang in valid:
                        _, code, token = create_response(cursor, name, affiliation, lang, False, True, None, email, None, role=role)
                        created.append((code, token, name, email, role, lang))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
        sent = 0
        for code, token, name, email, role, lang in created:
            if send_now and mail_available():
                link = absolute_url("expert.invite", token=token)
                if deliver(token, "invite", email, name, lang, link, None, role, quiet=True):
                    sent += 1
                    report.append(f"{code} · {name}: meghívó elkészült, levél elment ({email}).")
                else:
                    report.append(f"{code} · {name}: meghívó elkészült, de a levél nem ment el ({email}); a listából újra küldhető.")
            else:
                report.append(f"{code} · {name}: meghívó elkészült, levél nem ment (a listából küldhető).")
        summary = f"Tömeges meghívó: {len(created)} meghívó elkészült, {sent} levél elment, {n - len(created)} sor kimaradt vagy üres."
        flash(summary, "success" if created else "error")
        for line in report:
            flash(line, "success" if "elment" in line and "nem ment" not in line else "error")
        return redirect(url_for("expert.admin"))

    def deliver(token, kind, email, name, lang, link, deadline, role="fogorvos", referrer=None, quiet=False):
        """Meghívó vagy emlékeztető küldése. A meghívó határideje a küldés napja +
        DEADLINE_DAYS (ISO-dátumként az adatbázisba is kerül); az emlékeztető a tárolt
        határidőt ismétli. Az eredmény flash-üzenetben (quiet=True: csak visszatérési érték)."""
        if kind == "invite" and not deadline:
            deadline = deadline_iso()
        subject, text, html = build_email(kind, lang, name, link, deadline,
                                          logo_url=absolute_url("static", filename="predict-logo.png"),
                                          partner_logo_url=absolute_url("static", filename="semmelweis-logo.png"), role=role, referrer=referrer)
        try:
            send_mail(email, name, subject, text, html)
        except MailError as err:
            if not quiet:
                flash(f"A levél nem ment el ({email}): {err}", "error")
            return False
        if kind == "reminder":
            statement = ("UPDATE expert_prior_responses SET reminder_sent_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP WHERE token = %s", [token])
        else:
            statement = ("UPDATE expert_prior_responses SET invite_sent_at = CURRENT_TIMESTAMP, invite_deadline = COALESCE(invite_deadline, %s), "
                         "updated_at = CURRENT_TIMESTAMP WHERE token = %s", [deadline, token])
        execute_transaction([statement])
        if not quiet:
            flash(("Emlékeztető elküldve: " if kind == "reminder" else "Meghívó elküldve: ") + f"{name} ({email}).", "success")
        return True

    @bp.post("/admin/<int:response_id>/email")
    @require_admin
    def admin_email(response_id):
        """Meghívó (első alkalommal) vagy emlékeztető (később) küldése egy sorhoz."""
        validate_csrf()
        response = get_response_by_id(response_id)
        if response is None:
            abort(404)
        response = decorate(response)
        if response["status"] == "submitted":
            flash("Ez a kitöltés már beküldve, nem kell levél.", "error")
            return redirect(url_for("expert.admin"))
        email = _clean_text(request.form.get("invite_email"), 200) or response.get("invite_email")
        if not email or "@" not in email:
            flash("Adj meg egy e-mail-címet a küldéshez.", "error")
            return redirect(url_for("expert.admin"))
        if email != response.get("invite_email"):
            execute_transaction([("UPDATE expert_prior_responses SET invite_email = %s WHERE id = %s", [email, response_id])])
        lang = (response["background"] or {}).get("nyelv", "hu")
        link = absolute_url("expert.invite", token=response["token"])
        kind = "reminder" if response.get("invite_sent_at") else "invite"
        deliver(response["token"], kind, email, response["expert_name"] or "", lang, link, response.get("invite_deadline"), response["role"])
        return redirect(url_for("expert.admin"))

    @bp.get("/admin/<int:response_id>")
    @require_admin
    def admin_view(response_id):
        response = get_response_by_id(response_id)
        if response is None:
            abort(404)
        response = decorate(response)
        return render_template("expert_view.html", response=response, admin=True, role=response["role"])

    @bp.post("/admin/<int:response_id>/visszanyitas")
    @require_admin
    def admin_reopen(response_id):
        """Beküldött válasz visszanyitása szerkesztésre: piszkozat lesz, minden válasz
        megmarad, a kitöltő a saját linkjén módosít és újra beküld; a visszanyitás
        ténye a megjegyzésben rögzül (közléshez: melyik válasz módosult beküldés után)."""
        validate_csrf()
        response = get_response_by_id(response_id)
        if response is None:
            abort(404)
        if response["status"] != "submitted":
            flash("Csak beküldött válasz nyitható vissza.", "error")
            return redirect(url_for("expert.admin_view", response_id=response_id))
        stamp = format_stamp(datetime.now(timezone.utc), "%Y-%m-%d")
        execute_transaction([(
            """
            UPDATE expert_prior_responses
            SET status = 'draft', submitted_at = NULL, updated_at = CURRENT_TIMESTAMP,
                invite_note = CASE WHEN invite_note IS NULL OR invite_note = '' THEN %s ELSE invite_note || ' · ' || %s END
            WHERE id = %s AND status = 'submitted'
            """,
            [f"visszanyitva {stamp}", f"visszanyitva {stamp}", response_id],
        )])
        flash(f"{response['expert_code']} visszanyitva szerkesztésre; a válaszok megmaradtak, a kitöltő a saját linkjén módosíthat és újra beküldhet.", "success")
        return redirect(url_for("expert.admin_view", response_id=response_id))

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

    def export_selection():
        """Alapból csak a beküldött, nem szimulált válaszok; ?status=all a
        piszkozatokat, ?szimulacio=1 a szimulált sorokat is beveszi."""
        include_drafts = request.args.get("status") == "all"
        include_simulated = request.args.get("szimulacio") == "1"
        responses = [decorate(row) for row in list_responses(None if include_drafts else "submitted")]
        return [row for row in responses if include_simulated or not row["simulated"]]

    @bp.get("/admin/export/priorok.csv")
    @require_admin
    def export_priors():
        return csv_response(prior_rows(export_selection()), PRIOR_CSV_COLUMNS, "predict_expert_priorok.csv")

    @bp.get("/admin/export/hatter.csv")
    @require_admin
    def export_background():
        return csv_response(background_rows(export_selection()), BACKGROUND_CSV_COLUMNS, "predict_expert_hatter.csv")

    return bp
