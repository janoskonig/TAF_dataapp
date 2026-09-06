#!/usr/bin/env python3
"""Szimulált szakértői válaszok a PREDICT prior-felméréshez.

Célja: bemutatni, hogyan dolgozza fel a lánc (applikáció → CSV-export →
predict_bayes_feltaro.R) a majdani éles válaszokat. A válaszok tételenként
eltérő egyetértésű, kitalált profilokból származnak, és SEMMILYEN valós
szakértői véleményt nem tükröznek.

Használat:
    python3 simulate_expert_responses.py            # CSV-k írása (SZIMULACIO utótaggal)
    python3 simulate_expert_responses.py --db-insert  # + beszúrás az adatbázisba
    python3 simulate_expert_responses.py --db-cleanup # a szimulált sorok törlése

A szimulált adatbázis-sorok form_version = 'v1.0-SZIMULACIO' jelölést kapnak,
a kódjuk SZIMnn, a nevük „Szimulált szakértő nn”; a törlés erre a jelölésre
szűr, valós válaszhoz nem nyúl.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime, timedelta

from expert_priors import (
    BACKGROUND_CSV_COLUMNS,
    ITEMS,
    P_IRANY_VALUES,
    PRIOR_CSV_COLUMNS,
    SIM_VERSION,
    background_rows,
    prior_rows,
    write_csv,
)
CERT_LEVELS = P_IRANY_VALUES  # 50, 60, 70, 80, 90, 95, 99

# Tételenkénti "konszenzus-profil": irányválaszok valószínűsége, bizonyosság-
# súlyok (a 7 skálafokra), a 100 betegre vetített különbség (átlag, szórás),
# tipikus mechanizmusok. A számok kitaláltak, csak a heterogenitást illusztrálják.
PROFILES = {
    "F1": dict(p=(0.87, 0.03, 0.00, 0.07, 0.03), cert=(0, 0, .05, .15, .35, .30, .15), diff=(22, 7), mech=["illeszkedo_felulet", "nyomaseloszlas"], kuszob=("{:.0f} mm", 4, 8)),
    "F2": dict(p=(0.30, 0.05, 0.55, 0.07, 0.03), cert=(0, .05, .20, .35, .30, .10, 0), diff=(12, 6), mech=["alamenos_megkapaszkodas", "fajdalom"], kuszob=("enyhe, egyenletes alámenősség", None, None)),
    "F3": dict(p=(0.70, 0.05, 0.00, 0.20, 0.05), cert=(0, .05, .20, .35, .30, .10, 0), diff=(15, 6), mech=["szivohatas", "nyalfilm_tapadas"], kuszob=("{:.0f} mm", 15, 20)),
    "F4": dict(p=(0.65, 0.05, 0.00, 0.20, 0.10), cert=(.05, .10, .25, .30, .20, .10, 0), diff=(13, 6), mech=["izomegyensuly", "gerinc_vezetes"], kuszob=("{:.0f}°", 128, 138)),
    "F5": dict(p=(0.90, 0.00, 0.00, 0.07, 0.03), cert=(0, 0, .05, .15, .30, .30, .20), diff=(28, 8), mech=["nyomaseloszlas", "fajdalom"], kuszob=None),
    "F6": dict(p=(0.60, 0.05, 0.10, 0.15, 0.10), cert=(.05, .10, .25, .30, .20, .10, 0), diff=(12, 6), mech=["izomegyensuly", "gerinc_vezetes"], kuszob=("{:.0f}°", 5, 15)),
    "F7": dict(p=(0.50, 0.17, 0.00, 0.30, 0.03), cert=(.10, .15, .30, .25, .15, .05, 0), diff=(10, 6), mech=["alamenos_megkapaszkodas", "fajdalom"], kuszob=None),
    "F8": dict(p=(0.25, 0.10, 0.00, 0.35, 0.30), cert=(.15, .20, .30, .25, .10, 0, 0), diff=(8, 5), mech=["izomegyensuly", "gerinc_vezetes"], kuszob=None),
    "A1": dict(p=(0.95, 0.00, 0.00, 0.03, 0.02), cert=(0, 0, 0, .10, .30, .35, .25), diff=(32, 9), mech=["gerinc_vezetes", "nyomaseloszlas"], kuszob=None),
    "A3": dict(p=(0.30, 0.20, 0.00, 0.40, 0.10), cert=(.20, .25, .30, .20, .05, 0, 0), diff=(7, 5), mech=["izomegyensuly", "tureskepesseg"], kuszob=None),
    "A4": dict(p=(0.70, 0.07, 0.00, 0.20, 0.03), cert=(.05, .10, .25, .30, .20, .10, 0), diff=(12, 6), mech=["fajdalom", "szivohatas"], kuszob=None),
    "A5": dict(p=(0.75, 0.03, 0.00, 0.15, 0.07), cert=(0, .05, .20, .30, .30, .10, .05), diff=(16, 7), mech=["izomegyensuly", "gerinc_vezetes"], kuszob=None),
    "TUB": dict(p=(0.85, 0.03, 0.00, 0.07, 0.05), cert=(0, 0, .10, .20, .35, .25, .10), diff=(22, 8), mech=["gerinc_vezetes", "nyomaseloszlas"], kuszob=None),
    "A10": dict(p=(0.35, 0.05, 0.10, 0.25, 0.25), cert=(.15, .20, .30, .25, .10, 0, 0), diff=(9, 5), mech=["izomegyensuly", "tureskepesseg"], kuszob=None),
    "A11": dict(p=(0.93, 0.00, 0.00, 0.04, 0.03), cert=(0, 0, .05, .10, .30, .35, .20), diff=(30, 9), mech=["szivohatas", "izomegyensuly"], kuszob=None),
    "A12": dict(p=(0.70, 0.03, 0.00, 0.20, 0.07), cert=(.05, .10, .25, .30, .20, .10, 0), diff=(12, 6), mech=["fajdalom", "nyomaseloszlas"], kuszob=None),
}
DIRECTIONS = ["B_kedvezotlenebb", "A_kedvezotlenebb", "nem_monoton", "nincs_kulonbseg", "nem_tudom"]
RANK_WEIGHTS = {"A1": .9, "A11": .8, "TUB": .6, "F5": .5, "A5": .3, "F3": .3, "F4": .3, "F7": .2, "A12": .2, "A4": .15,
                "F1": .15, "F6": .1, "F2": .1, "A3": .05, "F8": .05, "A10": .05}
COMMENT_POOL = [
    "", "", "", "", "", "",
    "Csak jó lenyomattal érvényesül.",
    "Fiatalabb betegnél kevésbé számít.",
    "Sok múlik a beteg alkalmazkodásán.",
    "Implantátum-elhorgonyzásnál más a kép.",
    "Egyoldali forma jobban tolerálható.",
]
MISSING_POOL = [
    "", "", "",
    "nyálmennyiség és nyálminőség",
    "a beteg motivációja és korábbi fogsorviselése",
    "a vestibulum mélysége és a frenulumok tapadása",
    "a nyelv mérete és helyzete",
    "a mimikai izmok tónusa",
]
WORST_COMBO_POOL = ["A1 + A11", "A1 + TUB", "F5 + F4", "A1 + A5", "F1 + F3", "TUB + A11"]


def clip(value, lo, hi):
    return max(lo, min(hi, value))


def weighted_choice(rng, options, weights):
    return rng.choices(list(options), weights=list(weights), k=1)[0]


def shift_certainty(rng, weights, shift):
    """Az egyéni meggyőződés eltolja a bizonyosság-eloszlást egy-egy fokkal."""
    weights = list(weights)
    if shift > 0:
        weights = [0.0] + weights[:-1]
        weights[-1] += 0.0
    elif shift < 0:
        weights = weights[1:] + [0.0]
    total = sum(weights)
    if total <= 0:
        return weighted_choice(rng, CERT_LEVELS, (0, 0, .1, .3, .3, .2, .1))
    return weighted_choice(rng, CERT_LEVELS, [w / total for w in weights])


def simulate_expert(rng, index):
    years = int(clip(rng.lognormvariate(2.9, 0.45), 4, 42))
    per_year = int(clip(rng.gauss(22, 12), 4, 70))
    total = years * per_year
    category = "<100" if total < 100 else "100-500" if total < 500 else "500-1000" if total < 1000 else "1000-3000" if total < 3000 else ">3000"
    teaches = weighted_choice(rng, ["igen", "nem", "korabban"], (0.45, 0.4, 0.15))
    activity = weighted_choice(rng, ["egyetemi", "maganpraxis", "mindketto", "egyeb"], (0.35, 0.3, 0.3, 0.05))
    optimism = clip(rng.gauss(70, 8), 50, 90)            # alap-sikerarány
    conviction = rng.choice([-1, 0, 0, 0, 1])              # bizonyosság-eltolás
    scale = clip(rng.gauss(1.0, 0.2), 0.6, 1.5)            # hatásnagyság-hajlam
    background = {
        "diploma_ev": 2026 - years - rng.randint(0, 3),
        "evek_gyakorlat": years,
        "fogsorok_szama_kat": category,
        "oktat": teaches,
        "tevekenyseg": activity,
        "szakvizsga": rng.choice(["fogpótlástan", "fogpótlástan", "konzerváló fogászat és fogpótlástan", "szájsebészet"]),
        "evi_fogsorok": per_year,
    }
    calibration = {
        "alap_siker_100": int(round(optimism)),
        "anatomia_sulya_pct": int(clip(rng.gauss(45, 15), 10, 85)),
        "felso_vagy_also": weighted_choice(rng, ["felso", "also", "egyforman"], (0.1, 0.65, 0.25)),
        "felso_also_arany": rng.choice([None, None, "2", "3", "1,5"]),
        "kizaro_kepletek": rng.choice([None, None, "mélyült negatív gerinc tömött szájfenékkel", "abszolút mozgékony tuberculum mindkét oldalon", "frontális lötyögő gerinc nagy alámenősséggel"]),
        "regi_fogsor": weighted_choice(rng, ["alig", "kozepesen", "erosen"], (0.2, 0.5, 0.3)),
        "regi_fogsor_megjegyzes": None,
    }
    items = {}
    for item in ITEMS:
        profile = PROFILES[item["kod"]]
        direction = weighted_choice(rng, DIRECTIONS, profile["p"])
        if direction == "nem_monoton" and not item["optimum"]:
            direction = "B_kedvezotlenebb"
        answer = {"irany": direction, "kuszob": None, "megjegyzes": rng.choice(COMMENT_POOL) or None}
        if direction in ("A_kedvezotlenebb", "B_kedvezotlenebb"):
            answer["p_irany"] = shift_certainty(rng, profile["cert"], conviction)
            diff = clip(rng.gauss(profile["diff"][0] * scale, profile["diff"][1]), 2, 60)
            favourable = clip(optimism + diff / 2, 5, 98)
            unfavourable = clip(optimism - diff / 2, 2, 95)
            if direction == "B_kedvezotlenebb":
                answer["siker_A"], answer["siker_B"] = int(round(favourable)), int(round(unfavourable))
            else:
                answer["siker_A"], answer["siker_B"] = int(round(unfavourable)), int(round(favourable))
            if rng.random() < 0.85:
                answer["kulonbseg_min"] = int(clip(diff - rng.uniform(6, 15), 0, 100))
                answer["kulonbseg_max"] = int(clip(diff + rng.uniform(8, 22), 0, 100))
        else:
            answer["p_irany"] = None
            for key in ("siker_A", "siker_B", "kulonbseg_min", "kulonbseg_max"):
                answer[key] = None
        if profile["kuszob"] and rng.random() < 0.7:
            template, lo, hi = profile["kuszob"]
            answer["kuszob"] = template.format(rng.uniform(lo, hi)) if lo is not None else template
        for sub in item["subs"]:
            if rng.random() < 0.85:
                codes = [code for code, _ in sub["opciok"]]
                weights = [3 if i == 0 else (2 if i == 1 else 1) for i in range(len(codes))]
                answer[f"sub_{sub['kod']}"] = weighted_choice(rng, codes, weights)
        items[item["kod"]] = answer
    ranks = []
    pool = dict(RANK_WEIGHTS)
    while len(ranks) < 5:
        pick = weighted_choice(rng, list(pool.keys()), list(pool.values()))
        ranks.append(pick)
        pool.pop(pick)
    closing = {
        **{f"rang_{i + 1}": ranks[i] for i in range(5)},
        "legrosszabb_kombinacio": rng.choice(WORST_COMBO_POOL),
        "ellensulyozo": rng.choice([None, None, "puha szájfenék a lapos gerinc mellett", "jó tuberculum a sorvadt gerincnél"]),
        "hianyzo_kepletek": rng.choice(MISSING_POOL) or None,
        "onertekeles": weighted_choice(rng, ["1", "2", "3", "4", "5"], (0.05, 0.15, 0.4, 0.3, 0.1)),
        "megjegyzes": None,
    }
    submitted = datetime(2026, 9, 8, 9, 0) + timedelta(days=rng.randint(0, 40), hours=rng.randint(0, 9), minutes=rng.randint(0, 59))
    return {
        "id": index,
        "expert_code": f"SZIM{index:02d}",
        "expert_name": f"Szimulált szakértő {index:02d}",
        "expert_affiliation": rng.choice(["Szimulált klinika A", "Szimulált klinika B", "Szimulált praxis", None]),
        "token": f"szimulacio-{index:02d}",
        "status": "submitted",
        "consent_confirmed": True,
        "background": background,
        "calibration": calibration,
        "items": items,
        "closing": closing,
        "form_version": SIM_VERSION,
        "submitted_at": submitted,
        "created_at": submitted - timedelta(minutes=45),
        "updated_at": submitted,
    }


def simulate(n=30, seed=2026):
    rng = random.Random(seed)
    return [simulate_expert(rng, i + 1) for i in range(n)]


def write_csvs(responses, prefix="predict_expert"):
    prior_path = f"{prefix}_priorok_SZIMULACIO.csv"
    background_path = f"{prefix}_hatter_SZIMULACIO.csv"
    with open(prior_path, "w", encoding="utf-8", newline="") as handle:
        handle.write(write_csv(prior_rows(responses), PRIOR_CSV_COLUMNS))
    with open(background_path, "w", encoding="utf-8", newline="") as handle:
        handle.write(write_csv(background_rows(responses), BACKGROUND_CSV_COLUMNS))
    return prior_path, background_path


def connect():
    import psycopg2
    from dotenv import load_dotenv

    load_dotenv(".env")
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("Hiányzik a DATABASE_URL.")
    return psycopg2.connect(url)


def db_insert(responses):
    from psycopg2.extras import Json

    conn = connect()
    try:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM expert_prior_responses WHERE form_version = %s", [SIM_VERSION])
            for response in responses:
                cursor.execute(
                    """
                    INSERT INTO expert_prior_responses
                        (expert_code, expert_name, expert_affiliation, token, status, consent_confirmed,
                         background, calibration, items, closing, form_version, submitted_at, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, 'submitted', TRUE, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        response["expert_code"], response["expert_name"], response["expert_affiliation"], response["token"],
                        Json(response["background"]), Json(response["calibration"]), Json(response["items"]), Json(response["closing"]),
                        SIM_VERSION, response["submitted_at"], response["created_at"], response["updated_at"],
                    ],
                )
            cursor.execute("SELECT COUNT(*) FROM expert_prior_responses WHERE form_version = %s", [SIM_VERSION])
            count = cursor.fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    return count


def db_cleanup():
    conn = connect()
    try:
        with conn.cursor() as cursor:
            cursor.execute("DELETE FROM expert_prior_responses WHERE form_version = %s", [SIM_VERSION])
            deleted = cursor.rowcount
            cursor.execute("SELECT COUNT(*) FROM expert_prior_responses")
            remaining = cursor.fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    return deleted, remaining


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--db-insert", action="store_true", help="szimulált sorok beszúrása az adatbázisba (jelölve)")
    parser.add_argument("--db-cleanup", action="store_true", help="a jelölt szimulált sorok törlése")
    args = parser.parse_args()
    if args.db_cleanup:
        deleted, remaining = db_cleanup()
        print(f"Törölve: {deleted} szimulált sor; a táblában maradt: {remaining}")
        return
    responses = simulate(args.n, args.seed)
    prior_path, background_path = write_csvs(responses)
    print(f"{len(responses)} szimulált szakértő → {prior_path}, {background_path}")
    if args.db_insert:
        count = db_insert(responses)
        print(f"Adatbázisba beszúrva (form_version={SIM_VERSION}): {count} sor")


if __name__ == "__main__":
    main()
