#!/usr/bin/env python3
"""TDK pilot: does mandibular ridge form track burden or performance?

One anatomical predictor is compared against three conceptually different
outcomes in the same complete-case sample. Outputs are aggregate only.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DATA_FILE = Path("patients.csv")
OUTPUT_DIR = Path("stat_output")
REPORT_FILE = Path("TDK_A1_kimeneti_profil_report.md")
SEED = 20260819
N_BOOTSTRAP = 50_000

OHIP_ITEMS = [f"OHIP_{number}" for number in range(1, 6)]
A1_RISK_MAP = {1: 0.0, 2: 1.0 / 3.0, 3: 2.0 / 3.0, 4: 1.0, 5: 1.0}
CHEWING_WORSE_MAP = {
    "Kiváló": 1.0,
    "Jó": 2.0,
    "Átlagos": 3.0,
    "Rossz": 4.0,
    "Nagyon rossz": 5.0,
}

OUTCOMES = {
    "OHIP5_worse": "OHIP-5 – beteg által megélt teher",
    "chewing_worse": "Önértékelt rágóképesség",
    "MAI_worse": "MAI – objektív rágásteljesítmény",
}


def strict_sum(frame: pd.DataFrame) -> pd.Series:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    return numeric.sum(axis=1, min_count=numeric.shape[1])


def fmt_hu(value: float, digits: int = 2, signed: bool = False) -> str:
    spec = f"{'+' if signed else ''}.{digits}f"
    return format(value, spec).replace(".", ",")


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    required = {"A1_Kaan", "denture_type", "init_mai_huedegree", "chewing_today_situation", *OHIP_ITEMS}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = raw.copy()
    frame["A1_category"] = pd.to_numeric(frame["A1_Kaan"], errors="coerce")
    invalid_a1 = frame["A1_category"].notna() & ~frame["A1_category"].isin(A1_RISK_MAP)
    if invalid_a1.any():
        raise ValueError(f"Invalid A1 categories: {int(invalid_a1.sum())}")
    frame["A1_risk"] = frame["A1_category"].map(A1_RISK_MAP)
    frame["OHIP5_worse"] = strict_sum(frame[OHIP_ITEMS])
    frame["MAI_worse"] = pd.to_numeric(frame["init_mai_huedegree"], errors="coerce")
    frame["chewing_worse"] = frame["chewing_today_situation"].astype(str).str.strip().map(CHEWING_WORSE_MAP)

    target = frame["denture_type"].isin(["both", "lower"])
    complete_columns = ["A1_risk", *OUTCOMES.keys()]
    complete = target & frame[complete_columns].notna().all(axis=1)
    analysis = frame.loc[complete, ["A1_risk", "A1_category", *OUTCOMES.keys(), *OHIP_ITEMS]].copy()

    if analysis.shape[0] < 20:
        raise ValueError("Too few common complete cases")
    if not analysis["OHIP5_worse"].between(0, 20).all():
        raise ValueError("OHIP-5 outside expected 0-20 range")
    if not analysis["chewing_worse"].between(1, 5).all():
        raise ValueError("Self-rated chewing outside expected 1-5 range")

    audit = {
        "source_rows": int(frame.shape[0]),
        "mandibular_target_rows": int(target.sum()),
        "common_complete_rows": int(analysis.shape[0]),
        "excluded_from_common_sample": int(target.sum() - analysis.shape[0]),
        "target_missing_A1": int((target & frame["A1_risk"].isna()).sum()),
        "target_missing_MAI": int((target & frame["MAI_worse"].isna()).sum()),
        "target_missing_OHIP5": int((target & frame["OHIP5_worse"].isna()).sum()),
        "target_missing_self_rated_chewing": int((target & frame["chewing_worse"].isna()).sum()),
    }
    return analysis, audit


def spearman_profile(values: np.ndarray) -> np.ndarray:
    """A1 correlations, then the two predeclared correlation contrasts."""

    matrix = stats.spearmanr(values, axis=0).statistic
    correlations = np.asarray(matrix[0, 1:4], dtype=float)
    # Positive contrasts mean A1 is more strongly related to OHIP-5.
    contrasts = np.array(
        [
            correlations[0] - correlations[2],  # OHIP minus objective MAI
            correlations[0] - correlations[1],  # OHIP minus self-rated chewing
        ]
    )
    return np.concatenate([correlations, contrasts])


def estimate_profile(frame: pd.DataFrame) -> dict[str, object]:
    ordered = frame[["A1_risk", "OHIP5_worse", "chewing_worse", "MAI_worse"]].to_numpy(dtype=float)
    observed = spearman_profile(ordered)
    rng = np.random.default_rng(SEED)
    bootstrap = np.empty((N_BOOTSTRAP, observed.size), dtype=float)
    valid = 0
    for _ in range(N_BOOTSTRAP):
        sample = ordered[rng.integers(0, ordered.shape[0], size=ordered.shape[0])]
        estimate = spearman_profile(sample)
        if np.all(np.isfinite(estimate)):
            bootstrap[valid] = estimate
            valid += 1
    if valid < 0.99 * N_BOOTSTRAP:
        raise RuntimeError("Too many invalid bootstrap replicates")
    intervals = np.quantile(bootstrap[:valid], [0.025, 0.975], axis=0).T

    leave_one_out = np.array(
        [spearman_profile(np.delete(ordered, index, axis=0)) for index in range(ordered.shape[0])]
    )

    outcome_keys = list(OUTCOMES)
    associations = {}
    for index, key in enumerate(outcome_keys):
        associations[key] = {
            "label": OUTCOMES[key],
            "n": int(ordered.shape[0]),
            "spearman_rho": float(observed[index]),
            "bootstrap_95_ci": [float(intervals[index, 0]), float(intervals[index, 1])],
            "leave_one_out_range": [
                float(leave_one_out[:, index].min()),
                float(leave_one_out[:, index].max()),
            ],
        }

    contrast_names = ["OHIP5_minus_MAI", "OHIP5_minus_self_rated_chewing"]
    contrasts = {}
    for offset, name in enumerate(contrast_names, start=3):
        contrasts[name] = {
            "delta_rho": float(observed[offset]),
            "bootstrap_95_ci": [float(intervals[offset, 0]), float(intervals[offset, 1])],
            "leave_one_out_range": [
                float(leave_one_out[:, offset].min()),
                float(leave_one_out[:, offset].max()),
            ],
        }

    return {
        "bootstrap_replicates": int(valid),
        "associations": associations,
        "contrasts": contrasts,
    }


def plot_profile(profile: dict[str, object]) -> None:
    associations = profile["associations"]
    order = ["OHIP5_worse", "chewing_worse", "MAI_worse"]
    estimates = [associations[key]["spearman_rho"] for key in order]
    lower = [associations[key]["bootstrap_95_ci"][0] for key in order]
    upper = [associations[key]["bootstrap_95_ci"][1] for key in order]
    labels = [associations[key]["label"] for key in order]
    colors = ["#D97706", "#2563EB", "#2563EB"]
    positions = np.array([2, 1, 0])

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(10, 5.8), dpi=160)
    for position, estimate, low, high, color in zip(positions, estimates, lower, upper, colors):
        ax.errorbar(
            estimate,
            position,
            xerr=[[estimate - low], [high - estimate]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=2.2,
            capsize=5,
            markersize=8,
            zorder=3,
        )
        ax.text(
            0.83,
            position,
            f"ρ={fmt_hu(estimate, 2, True)}  [{fmt_hu(low, 2)}; {fmt_hu(high, 2)}]",
            va="center",
            ha="right",
            fontsize=10.5,
            color="#111827",
        )

    ax.axvline(0, color="#6B7280", linewidth=1.2, linestyle="--")
    ax.set_yticks(positions, labels)
    ax.set_xlim(-0.5, 0.9)
    ax.set_xticks(np.arange(-0.4, 0.9, 0.2))
    ax.set_xlabel("Spearman-korreláció az A1 gerincforma súlyosságával")
    ax.set_title(
        "Ugyanaz az anatómiai jel eltérő mintázatot ad a három kimeneten",
        loc="left",
        fontweight="bold",
        pad=24,
    )
    ax.text(
        0,
        1.025,
        f"Közös teljeseset-minta: n={associations['OHIP5_worse']['n']}; pont = ρ, vonal = bootstrap 95%-os CI",
        transform=ax.transAxes,
        color="#4B5563",
        va="bottom",
    )
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tdk_a1_outcome_profile.png", bbox_inches="tight")
    plt.close(fig)


def association_table(profile: dict[str, object]) -> str:
    rows = [
        "| Kimenet | n | Spearman ρ | Bootstrap 95%-os CI | LOO-tartomány |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ["OHIP5_worse", "chewing_worse", "MAI_worse"]:
        item = profile["associations"][key]
        ci = item["bootstrap_95_ci"]
        loo = item["leave_one_out_range"]
        rows.append(
            f"| {item['label']} | {item['n']} | {fmt_hu(item['spearman_rho'], 2, True)} "
            f"| {fmt_hu(ci[0], 2)}; {fmt_hu(ci[1], 2)} "
            f"| {fmt_hu(loo[0], 2)}; {fmt_hu(loo[1], 2)} |"
        )
    return "\n".join(rows)


def write_report(audit: dict[str, int], profile: dict[str, object]) -> None:
    associations = profile["associations"]
    ohip = associations["OHIP5_worse"]
    chew = associations["chewing_worse"]
    mai = associations["MAI_worse"]
    delta_mai = profile["contrasts"]["OHIP5_minus_MAI"]
    delta_chew = profile["contrasts"]["OHIP5_minus_self_rated_chewing"]

    report = f"""# A mandibularis gerincforma kimeneti profilja

**Kisebb scope-ú TDK keresztmetszeti pilot elemzés — 2026-08-19**

## Miért jobb kérdés ez?

A klinikailag kedvezőtlen anatómia és a rosszabb betegállapot egyszerű
összekapcsolása közel tautologikus. A jelen elemzés ezért nem azt kérdezi,
hogy a „rossz anatómia rossz-e”, hanem azt, hogy **ugyanaz az anatómiai
súlyosság hol jelenik meg**:

1. a beteg által megélt összetett teherben (OHIP-5),
2. a közvetlen önértékelt rágóképességben, vagy
3. az objektív színkeveréses rágásteljesítményben (MAI).

## Javasolt TDK-cím

> **A mandibularis gerincforma kimeneti profilja: objektív
> rágásteljesítmény vagy beteg által megélt teher?**

## Fő eredmény

Ugyanazon {ohip['n']} betegben a súlyosabb A1 gerincforma az OHIP-5-tel
mérsékelt kapcsolatot mutatott (`rho={fmt_hu(ohip['spearman_rho'], 2, True)}`),
miközben az objektív MAI-jal (`rho={fmt_hu(mai['spearman_rho'], 2, True)}`) és
a közvetlen önértékelt rágóképességgel
(`rho={fmt_hu(chew['spearman_rho'], 2, True)}`) nulla körüli pontbecslést,
de széles bizonytalansági intervallumot adott. Ez **eltérő kimeneti
pilotmintázat**, nem bizonyított mechanizmus vagy kapcsolatnélküliség.

## Anyag és módszer

- **Dizájn:** másodlagos, keresztmetszeti, feltáró pilot elemzés.
- **Forrás:** {audit['source_rows']} betegrekord; mandibularisan releváns
  (`both` vagy `lower`) beteg {audit['mandibular_target_rows']}.
- **Közös teljeseset-minta:** A1, OHIP-5, önértékelt rágóképesség és MAI
  együttes megléte mellett `n={audit['common_complete_rows']}`. A közös minta
  biztosítja, hogy a korrelációk különbsége ne eltérő betegösszetételből
  adódjon.
- **Prediktor:** A1 Kaán szerinti gerincforma, előzetesen rögzített telítődő
  monoton kódolással: 1→0; 2→1/3; 3→2/3; 4–5→1.
- **Kimenetek:** mindháromnál a magasabb érték rosszabbat jelent: OHIP-5
  0–20, önértékelt rágás 1–5, MAI huedegree.
- **Becslés:** Spearman-korreláció, {profile['bootstrap_replicates']:,}
  betegszintű bootstrap ismétlésből származó percentilis 95%-os
  intervallummal. A két korrelációkülönbséget ugyanabban a bootstrap
  mintában számítottuk.
- **Stabilitás:** minden beteget egyenként kihagyó leave-one-out ellenőrzés.

## Eredmények

{association_table(profile)}

![Az A1 gerincforma három kimenettel mutatott kapcsolata](stat_output/tdk_a1_outcome_profile.png)

**Ábra alternatív leírása:** Az A1–OHIP-5 korreláció pozitív és a bootstrap
intervalluma nem éri el a nullát. Az A1–önértékelt rágás és A1–MAI becslése
nulla körüli, széles intervallummal.

### A korrelációk közvetlen összevetése

- OHIP-5 mínusz MAI: `Delta rho={fmt_hu(delta_mai['delta_rho'], 2, True)}`;
  bootstrap 95%-os CI `{fmt_hu(delta_mai['bootstrap_95_ci'][0], 2)};
  {fmt_hu(delta_mai['bootstrap_95_ci'][1], 2)}`.
- OHIP-5 mínusz önértékelt rágás:
  `Delta rho={fmt_hu(delta_chew['delta_rho'], 2, True)}`; bootstrap 95%-os CI
  `{fmt_hu(delta_chew['bootstrap_95_ci'][0], 2)};
  {fmt_hu(delta_chew['bootstrap_95_ci'][1], 2)}`.

Az OHIP–MAI különbség iránya érdekes, de az intervallum a nullát is
tartalmazza; ebből nem állítható biztos kimenetspecificitás. Az OHIP és az
egytételes önértékelés különbsége pontosabb, de részben a két mérőeszköz
eltérő információtartalmából is adódhat.

## TDK-n védhető következtetés

> A vizsgált pilotmintában a klinikai mandibularis gerincforma súlyossága
> a többdimenziós betegteherrel mutatott kapcsolatot, miközben az objektív
> színkeveréses teljesítmény és az egytételes globális rágásértékelés
> becslése nulla körüli és bizonytalan volt. A mintázat felveti, hogy az
> anatómiai, objektív funkcionális és betegjelzett kimenetek nem kezelhetők
> automatikusan egymás helyettesítőiként. Ennek megerősítéséhez nagyobb,
> prospektíven tervezett minta szükséges.

Ez érdemben különbözik attól az állítástól, hogy „a rossz anatómia rossz”:
a klinikai döntési kérdés az, hogy **melyik kimenetet kell mérni**, ha a
betegterhet vagy a mechanikai teljesítményt akarjuk megérteni.

## Szakirodalmi kontextus

Az objektív és szubjektív rágásmérés egyezése a teljesfogsor-viselőkben nem
lezárt kérdés. Egy 2024-es vizsgálatban az objektív mérés a harapási erővel
és az okkluzális kontaktussal, a szubjektív mérés pedig az életminőséggel és
a tápláltsággal járt együtt; a két mérési mód között nem tudtak egyezést
kimutatni. Más vizsgálatok ugyanakkor mérsékelt kapcsolatot találtak, tehát
a jelen eredmény nem egyszerűen ismert tény reprodukciója.

- [Wongcharee és mtsai., 2024 – szubjektív és objektív rágás teljesfogsor-viselőkben](https://he02.tci-thaijo.org/index.php/mdentjournal/article/view/267362)
- [Yamamoto és Shiga, 2018 – rágásteljesítmény és OHRQoL](https://www.sciencedirect.com/science/article/pii/S1883195818300069)
- [Campos és mtsai., 2018 – technikai minőség, rágási hatékonyság és életminőség](https://pubmed.ncbi.nlm.nih.gov/29120095/)

## Kötelező korlátok

1. A kérdés a korábbi adatfeltárás után lett kijelölve, ezért az eredmény
   hipotézisgeneráló; nincs prospektíven preregisztrált megerősítő teszt.
2. Az OHIP-5, a globális rágásértékelés és a MAI nem azonos reliabilitású és
   skálájú mérőeszközök. A korrelációkülönbség részben mérési pontossági
   különbség lehet.
3. A közös minta csak {audit['common_complete_rows']} beteg; a nulla körüli
   MAI-korreláció intervalluma széles, ezért nincs bizonyíték ekvivalenciára
   vagy valódi kapcsolatnélküliségre.
4. A keresztmetszeti elemzés nem állapít meg okságot, prognózist vagy
   kezelésre adott választ.
5. A jelen MAI egy adott teszt és feldolgozási módszer eredménye; a
   következtetés nem általánosítható automatikusan minden objektív
   rágásvizsgálatra.
6. A kezelésre jelentkező betegminta szelektált.

## Reprodukálhatóság és adatvédelem

- Elemzőszkript: `tdk_a1_outcome_profile_analysis.py`
- Aggregált eredmény: `stat_output/tdk_a1_outcome_profile_summary.json`
- Aggregált korrelációs tábla: `stat_output/tdk_a1_outcome_profile.csv`
- Ábra: `stat_output/tdk_a1_outcome_profile.png`
- A kimenetek nem tartalmaznak nevet, TAJ-számot vagy betegszintű adatot.
""".replace("50,000", "50 000")
    REPORT_FILE.write_text(report, encoding="utf-8")


def write_aggregate_csv(profile: dict[str, object]) -> None:
    rows = []
    for key, item in profile["associations"].items():
        rows.append(
            {
                "outcome": key,
                "label": item["label"],
                "n": item["n"],
                "spearman_rho": item["spearman_rho"],
                "ci_low": item["bootstrap_95_ci"][0],
                "ci_high": item["bootstrap_95_ci"][1],
                "loo_low": item["leave_one_out_range"][0],
                "loo_high": item["leave_one_out_range"][1],
            }
        )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "tdk_a1_outcome_profile.csv", index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    raw = pd.read_csv(DATA_FILE)
    analysis, audit = prepare_data(raw)
    profile = estimate_profile(analysis)
    payload = {
        "analysis": "A1 mandibular ridge form outcome profile",
        "status": "exploratory cross-sectional pilot",
        "data_audit": audit,
        **profile,
    }
    (OUTPUT_DIR / "tdk_a1_outcome_profile_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_aggregate_csv(profile)
    plot_profile(profile)
    write_report(audit, profile)
    print(
        "Analysis complete: "
        f"n={audit['common_complete_rows']}, "
        f"A1-OHIP rho={profile['associations']['OHIP5_worse']['spearman_rho']:.3f}, "
        f"A1-MAI rho={profile['associations']['MAI_worse']['spearman_rho']:.3f}"
    )
    print(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
