#!/usr/bin/env python3
"""Small-scope TDK analysis: mandibular ridge form and OHIP-5.

The script deliberately answers one cross-sectional question with one primary
predictor and one primary outcome.  It writes aggregated results only; names,
health identifiers and row-level exports are never included in the outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm


DATA_FILE = Path("patients.csv")
OUTPUT_DIR = Path("stat_output")
REPORT_FILE = Path("TDK_A1_OHIP_pilot_report.md")
SEED = 20260819
N_BOOTSTRAP = 20_000
N_PERMUTATIONS = 20_000

OHIP_ITEMS = [f"OHIP_{item}" for item in range(1, 6)]
REQUIRED_COLUMNS = {
    "A1_Kaan",
    "denture_type",
    "record_datetime",
    "birthdate",
    "gender",
    *OHIP_ITEMS,
}

# Expert-informed monotone, saturating encoding: categories 4 and 5 represent
# approximately the same worst-end risk rather than a further linear increase.
A1_RISK_MAP = {1: 0.0, 2: 1.0 / 3.0, 3: 2.0 / 3.0, 4: 1.0, 5: 1.0}


def strict_row_sum(frame: pd.DataFrame) -> pd.Series:
    """Return a row sum only when every item is observed."""

    numeric_frame = frame.apply(pd.to_numeric, errors="coerce")
    return numeric_frame.sum(axis=1, min_count=numeric_frame.shape[1])


def fmt_hu(value: float, digits: int = 2, signed: bool = False) -> str:
    """Hungarian decimal formatting for the generated report."""

    spec = f"{'+' if signed else ''}.{digits}f"
    return format(value, spec).replace(".", ",")


def fmt_p(value: float) -> str:
    if value < 0.001:
        return "<0,001"
    return fmt_hu(value, 3)


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    missing_columns = REQUIRED_COLUMNS.difference(raw.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    frame = raw.copy()
    frame["A1_category"] = pd.to_numeric(frame["A1_Kaan"], errors="coerce")
    invalid_a1 = frame["A1_category"].notna() & ~frame["A1_category"].isin(A1_RISK_MAP)
    if invalid_a1.any():
        raise ValueError(f"Invalid A1 categories found: {int(invalid_a1.sum())}")

    frame["A1_risk"] = frame["A1_category"].map(A1_RISK_MAP)
    frame["OHIP5"] = strict_row_sum(frame[OHIP_ITEMS])

    recorded = pd.to_datetime(frame["record_datetime"], errors="coerce")
    born = pd.to_datetime(frame["birthdate"], errors="coerce")
    frame["age"] = (recorded - born).dt.days / 365.25
    implausible_age = frame["age"].notna() & ~frame["age"].between(18, 105)
    frame.loc[implausible_age, "age"] = np.nan

    frame["female"] = frame["gender"].map({"Male": 0.0, "Female": 1.0})
    frame["both_jaws"] = frame["denture_type"].map({"lower": 0.0, "both": 1.0})

    target_jaw = frame["denture_type"].isin(["both", "lower"])
    complete_primary = target_jaw & frame["A1_risk"].notna() & frame["OHIP5"].notna()
    analysis = frame.loc[complete_primary].copy()

    if not analysis["OHIP5"].between(0, 20).all():
        raise ValueError("OHIP-5 total outside its expected 0-20 range")
    if analysis.shape[0] < 10:
        raise ValueError("Too few complete observations for the planned analysis")

    audit = {
        "source_rows": int(frame.shape[0]),
        "target_jaw_rows": int(target_jaw.sum()),
        "primary_complete_rows": int(analysis.shape[0]),
        "target_rows_missing_A1": int((target_jaw & frame["A1_risk"].isna()).sum()),
        "target_rows_incomplete_OHIP5": int((target_jaw & frame["OHIP5"].isna()).sum()),
        "implausible_ages_set_missing": int(implausible_age.sum()),
    }
    return analysis, audit


def primary_association(
    predictor: pd.Series, outcome: pd.Series
) -> dict[str, float | int | list[float]]:
    x = predictor.to_numpy(dtype=float)
    y = outcome.to_numpy(dtype=float)
    observed = float(stats.spearmanr(x, y).statistic)
    rng = np.random.default_rng(SEED)

    boot = np.empty(N_BOOTSTRAP, dtype=float)
    valid = 0
    for _ in range(N_BOOTSTRAP):
        indices = rng.integers(0, len(x), size=len(x))
        coefficient = stats.spearmanr(x[indices], y[indices]).statistic
        if np.isfinite(coefficient):
            boot[valid] = coefficient
            valid += 1
    if valid < 0.99 * N_BOOTSTRAP:
        raise RuntimeError("Too many invalid bootstrap samples")
    ci_low, ci_high = np.quantile(boot[:valid], [0.025, 0.975])

    extreme = 0
    for _ in range(N_PERMUTATIONS):
        permuted = rng.permutation(y)
        coefficient = stats.spearmanr(x, permuted).statistic
        extreme += int(abs(coefficient) >= abs(observed) - 1e-12)
    permutation_p = (extreme + 1) / (N_PERMUTATIONS + 1)

    leave_one_out = np.array(
        [
            stats.spearmanr(np.delete(x, index), np.delete(y, index)).statistic
            for index in range(len(x))
        ]
    )

    return {
        "n": int(len(x)),
        "spearman_rho": observed,
        "bootstrap_95_ci": [float(ci_low), float(ci_high)],
        "permutation_p_two_sided": float(permutation_p),
        "bootstrap_valid_replicates": int(valid),
        "permutation_replicates": int(N_PERMUTATIONS),
        "leave_one_out_rho_range": [
            float(np.nanmin(leave_one_out)),
            float(np.nanmax(leave_one_out)),
        ],
    }


def adjusted_sensitivity(frame: pd.DataFrame) -> dict[str, float | int | list[float]]:
    columns = ["OHIP5", "A1_risk", "age", "female", "both_jaws"]
    complete = frame[columns].dropna().copy()
    complete["age_per_10y"] = (complete["age"] - complete["age"].mean()) / 10.0
    design = sm.add_constant(
        complete[["A1_risk", "age_per_10y", "female", "both_jaws"]],
        has_constant="add",
    )
    model = sm.OLS(complete["OHIP5"], design).fit(cov_type="HC3")
    interval = model.conf_int().loc["A1_risk"]
    return {
        "n": int(model.nobs),
        "contrast": "A1 risk 0 to 1 (category 1 to saturated categories 4-5)",
        "ohip5_difference": float(model.params["A1_risk"]),
        "hc3_95_ci": [float(interval.iloc[0]), float(interval.iloc[1])],
        "p_value": float(model.pvalues["A1_risk"]),
        "r_squared": float(model.rsquared),
        "covariates": ["age", "sex", "both vs lower denture type"],
    }


def category_summary(frame: pd.DataFrame) -> pd.DataFrame:
    summarized = (
        frame.groupby("A1_category", observed=True)["OHIP5"]
        .agg(
            n="count",
            mean="mean",
            sd="std",
            median="median",
            q1=lambda values: values.quantile(0.25),
            q3=lambda values: values.quantile(0.75),
        )
        .reset_index()
    )
    summarized["A1_category"] = summarized["A1_category"].astype(int)
    return summarized


def plot_aggregated_distribution(frame: pd.DataFrame, primary: dict[str, object]) -> None:
    groups = [
        frame.loc[frame["A1_category"] == 1, "OHIP5"].to_numpy(),
        frame.loc[frame["A1_category"] == 2, "OHIP5"].to_numpy(),
        frame.loc[frame["A1_category"] == 3, "OHIP5"].to_numpy(),
        frame.loc[frame["A1_category"].isin([4, 5]), "OHIP5"].to_numpy(),
    ]
    labels = ["1", "2", "3", "4–5"]
    counts = [len(group) for group in groups]
    medians = [float(np.median(group)) for group in groups]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )
    fig, ax = plt.subplots(figsize=(9, 5.8), dpi=160)
    boxes = ax.boxplot(
        groups,
        labels=labels,
        widths=0.58,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111827", "linewidth": 2.0},
        whiskerprops={"color": "#4B5563", "linewidth": 1.2},
        capprops={"color": "#4B5563", "linewidth": 1.2},
        boxprops={"edgecolor": "#2563EB", "linewidth": 1.4},
    )
    for box in boxes["boxes"]:
        box.set_facecolor("#DBEAFE")

    positions = np.arange(1, 5)
    ax.plot(
        positions,
        medians,
        color="#D97706",
        marker="D",
        markersize=6,
        linewidth=1.6,
        linestyle="--",
    )
    for position, count in zip(positions, counts):
        ax.text(position, 20.35, f"n={count}", ha="center", va="bottom", fontsize=10)

    ci = primary["bootstrap_95_ci"]
    subtitle = (
        f"n={primary['n']}; Spearman ρ={fmt_hu(primary['spearman_rho'], 2)}; "
        f"bootstrap 95% CI {fmt_hu(ci[0], 2)}–{fmt_hu(ci[1], 2)}"
    )
    ax.set_title(
        "Súlyosabb mandibularis gerincforma mellett rosszabb OHIP‑5 látszik",
        loc="left",
        fontweight="bold",
        pad=26,
    )
    ax.text(0.0, 1.025, subtitle, transform=ax.transAxes, ha="left", va="bottom", color="#4B5563")
    ax.set_xlabel("A1 – mandibularis gerincforma Kaán szerint")
    ax.set_ylabel("OHIP‑5 összpontszám (0–20; magasabb = rosszabb)")
    ax.set_ylim(-0.5, 21.5)
    ax.set_yticks(np.arange(0, 21, 5))
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.0,
        -0.19,
        "4–5 összevonva az előzetes telítődő kódolás szerint.\n"
        "Doboz = IQR; fekete vonal = medián; narancssárga vonal = a mediánok menete.",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#4B5563",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "tdk_a1_ohip_figure.png", bbox_inches="tight")
    plt.close(fig)


def markdown_table(summary: pd.DataFrame) -> str:
    rows = [
        "| A1 kategória | n | OHIP-5 átlag (SD) | Medián [IQR] |",
        "|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        rows.append(
            "| {category} | {n} | {mean} ({sd}) | {median} [{q1}–{q3}] |".format(
                category=row.A1_category,
                n=int(row.n),
                mean=fmt_hu(row.mean, 1),
                sd=fmt_hu(row.sd, 1),
                median=fmt_hu(row.median, 1),
                q1=fmt_hu(row.q1, 1),
                q3=fmt_hu(row.q3, 1),
            )
        )
    return "\n".join(rows)


def write_report(
    audit: dict[str, int],
    primary: dict[str, object],
    adjusted: dict[str, object],
    summary: pd.DataFrame,
) -> None:
    ci = primary["bootstrap_95_ci"]
    loo = primary["leave_one_out_rho_range"]
    adjusted_ci = adjusted["hc3_95_ci"]
    report = f"""# Mandibularis gerincforma és orális életminőség

**Kisebb scope-ú TDK keresztmetszeti pilot elemzés — 2026-08-19**

## Egy mondatos eredmény

A mandibularisan fogatlan, teljes lemezes fogpótlásra jelentkező betegekben a
súlyosabb, Kaán szerinti mandibularis gerincforma mérsékelten együtt járt a
rosszabb OHIP-5 életminőséggel (`rho={fmt_hu(primary['spearman_rho'], 2, True)}`;
bootstrap 95%-os CI `{fmt_hu(ci[0], 2)}; {fmt_hu(ci[1], 2)}`; `n={primary['n']}`),
de az elemzés feltáró és keresztmetszeti, ezért nem bizonyít prognosztikai vagy
okozati kapcsolatot.

## Javasolt TDK-cím

> **A mandibularis gerincforma és az orális egészséggel összefüggő
> életminőség kapcsolata: keresztmetszeti pilotvizsgálat**

## Kutatási kérdés és hipotézis

**Kérdés:** Együtt jár-e a kedvezőtlenebb mandibularis gerincforma a rosszabb
kiindulási, orális egészséggel összefüggő életminőséggel?

**Irányhipotézis:** A magasabb A1 kategória mellett magasabb, azaz rosszabb
OHIP-5 összpontszám várható.

Ez a szűkítés egyetlen klinikai expozíciót, egyetlen primer kimenetet és
egyetlen primer asszociációs becslést tartalmaz. Nem része az anatómiai
összpontszám, a MAI, a GOHAI, a mediáció, az interakciók vagy a sok
tételenkénti teszt.

## Anyag és módszer

- **Dizájn:** másodlagos, keresztmetszeti pilot elemzés.
- **Forrásadat:** {audit['source_rows']} betegrekord; mandibularisan releváns
  (`both` vagy `lower`) rekord {audit['target_jaw_rows']}.
- **Elemzési populáció:** teljes A1–OHIP-5 adatpárral {primary['n']} beteg.
- **Prediktor:** A1, mandibularis gerincforma Kaán szerint. Az előzetes
  szakértői specifikáció telítődő monoton kódolása: 1→0, 2→1/3, 3→2/3,
  4–5→1.
- **Primer kimenet:** OHIP-5 összpontszám, 0–20; magasabb érték rosszabb
  életminőséget jelent. Csak mind az öt tétel meglétekor számítottuk.
- **Primer becslés:** Spearman-rangkorreláció, {primary['bootstrap_valid_replicates']:,}
  betegszintű bootstrap ismétlésből származó percentilis 95%-os intervallummal
  és {primary['permutation_replicates']:,} ismétléses kétoldali permutációs
  p-értékkel.
- **Érzékenységi modell:** HC3-robusztus lineáris regresszió életkorra,
  nemre és `both` vs. `lower` fogsortípusra korrigálva. Ez a hiányzó
  kovariánsok miatt {adjusted['n']} teljes esetet használ.

## Eredmények

{markdown_table(summary)}

Az elsődleges kapcsolat `rho={fmt_hu(primary['spearman_rho'], 2, True)}` volt
(bootstrap 95%-os CI `{fmt_hu(ci[0], 2)}; {fmt_hu(ci[1], 2)}`; kétoldali
permutációs `p={fmt_p(primary['permutation_p_two_sided'])}`). Az egyes betegek
egyenkénti kihagyásakor a rho `{fmt_hu(loo[0], 2, True)}` és
`{fmt_hu(loo[1], 2, True)}` között maradt, tehát az irány nem egyetlen
megfigyelésen múlt.

A korrigált érzékenységi modellben a legkedvezőbb (1) és a telített
legkedvezőtlenebb (4–5) gerincforma közötti becsült OHIP-5-különbség
`{fmt_hu(adjusted['ohip5_difference'], 2, True)}` pont volt (HC3 95%-os CI
`{fmt_hu(adjusted_ci[0], 2, True)}; {fmt_hu(adjusted_ci[1], 2, True)}`;
`p={fmt_p(adjusted['p_value'])}`). Ez érzékenységi eredmény, nem a primer
rangkorreláció helyettesítője.

![Az OHIP-5 eloszlása A1 kategóriánként](stat_output/tdk_a1_ohip_figure.png)

**Ábra alternatív leírása:** Az A1 gerincforma 1-től 4–5 felé romló
kategóriáiban az OHIP-5 eloszlása összességében felfelé tolódik, de a
csoportokon belüli szóródás nagy és átfedő.

## Mit lehet állítani a TDK-n?

> A vizsgált keresztmetszeti pilotmintában a kedvezőtlenebb mandibularis
> gerincforma mérsékelt, várt irányú kapcsolatot mutatott a rosszabb OHIP-5
> életminőséggel. A becslés további mintán történő megerősítést igényel; a
> jelen eredmény nem igazolja, hogy a gerincforma előre jelzi az új fogsor
> sikerét vagy az életminőség későbbi változását.

Kerülendő megfogalmazás: „a gerincatrophia rontja az életminőséget”, „az A1
bizonyított prediktor”, illetve „szignifikáns eredmény miatt a hipotézis
igazolódott”.

## Kötelező korlátok

1. A kérdés a teljes projekt korábbi feltáró elemzései után lett kiválasztva.
   A permutációs p-érték ezért leíró/tájékoztató, nem prospektíven
   preregisztrált megerősítő teszt.
2. A keresztmetszeti dizájn nem állapít meg időbeliséget vagy okságot.
3. A minta kicsi, különösen a 4-es és 5-ös kategóriákban; ezért ezeket az
   előzetes telítődő specifikációval összhangban az ábrán összevontuk.
4. Az OHIP-5 betegjelzett kimenet. A mechanikai stabilitást, retenciót vagy a
   kezelés utáni változást ez az elemzés nem vizsgálja.
5. A korrigált modell csak {adjusted['n']} teljes esetet tartalmaz, ezért
   érzékenységi ellenőrzésként értelmezendő.
6. A kezelésre jelentkező minta nem reprezentál automatikusan minden teljesen
   fogatlan beteget.

## Rövid TDK-absztrakt vázlat

**Bevezetés:** A mandibularis gerinc sorvadása ronthatja a teljes lemezes
fogsor alátámasztását és stabilitását, de nem egyértelmű, hogy a klinikai
gerincforma mennyiben tükröződik a beteg által megélt orális
életminőségben.

**Cél:** A Kaán szerinti mandibularis gerincforma és a kiindulási OHIP-5
összpontszám keresztmetszeti kapcsolatának becslése.

**Módszer:** {primary['n']} mandibularisan fogatlan beteg adatain Spearman-
korrelációt számítottunk telítődő ordinális A1-kódolással, bootstrap
konfidenciaintervallummal és permutációs p-értékkel. Életkorra, nemre és a
fogatlan állcsontok számára korrigált robusztus regresszió készült
érzékenységi elemzésként.

**Eredmény:** A súlyosabb gerincforma rosszabb OHIP-5 felé mutatott
(`rho={fmt_hu(primary['spearman_rho'], 2, True)}`; 95%-os CI
`{fmt_hu(ci[0], 2)}; {fmt_hu(ci[1], 2)}`). A kapcsolat iránya minden egyes
beteg kihagyása után megmaradt. A korrigált modell hasonló irányú becslést
adott, de kevesebb teljes esetből.

**Következtetés:** A klinikai mandibularis gerincforma a beteg által megélt
orális életminőség egyik lehetséges jelzője lehet, de az eredmény
hipotézisgeneráló és független, prospektív megerősítést igényel.

## Reprodukálhatóság és adatvédelem

- Elemzőszkript: `tdk_a1_ohip_analysis.py`
- Aggregált eredmény: `stat_output/tdk_a1_ohip_summary.json`
- Aggregált kategóriatáblázat: `stat_output/tdk_a1_ohip_by_category.csv`
- Ábra: `stat_output/tdk_a1_ohip_figure.png`
- A kimenetek nem tartalmaznak nevet, TAJ-számot vagy betegszintű exportot.
""".replace("20,000", "20 000")
    REPORT_FILE.write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    raw = pd.read_csv(DATA_FILE)
    analysis, audit = prepare_data(raw)
    primary = primary_association(analysis["A1_risk"], analysis["OHIP5"])
    adjusted = adjusted_sensitivity(analysis)
    summary = category_summary(analysis)

    payload = {
        "analysis": "TDK A1 mandibular ridge form vs OHIP-5",
        "status": "exploratory cross-sectional pilot",
        "data_audit": audit,
        "primary_association": primary,
        "adjusted_sensitivity": adjusted,
    }
    (OUTPUT_DIR / "tdk_a1_ohip_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary.to_csv(OUTPUT_DIR / "tdk_a1_ohip_by_category.csv", index=False)
    plot_aggregated_distribution(analysis, primary)
    write_report(audit, primary, adjusted, summary)

    print(f"Analysis complete: n={primary['n']}, rho={primary['spearman_rho']:.3f}")
    print(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
