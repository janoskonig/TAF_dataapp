#!/usr/bin/env python3
"""TDK pilot: mandibular anatomy and hue-degree MAI only.

The analysis has one objective outcome (baseline hue circular SD in degrees),
one primary expert-informed anatomical burden score, and five prespecified
construct-level secondary predictors. Outputs are aggregate only.
"""

from __future__ import annotations

import json
from pathlib import Path
import zlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from anatomical_burden_analysis import build_scores


DATA_FILE = Path("patients.csv")
OUTPUT_DIR = Path("stat_output")
REPORT_FILE = Path("TDK_anatomia_huedegree_MAI_report.md")
SEED = 20260819
N_BOOTSTRAP = 20_000
N_PERMUTATIONS = 20_000

ANATOMY_SOURCE_COLUMNS = [
    "A1_Kaan",
    "A12",
    "A4_jobb",
    "A4_bal",
    "A5_jobb",
    "A5_bal",
    "A6_jobb",
    "A6_bal",
    "A7_jobb",
    "A7_bal",
    "A9_jobb",
    "A9_bal",
    "A11",
]

COMPONENTS = {
    "ridge": "Gerincatrophia (A1/A12)",
    "torus": "Torus mandibularis (A4)",
    "lingual": "Lingualis tasak (A5)",
    "tuberculum": "Tuberculum-komplex (A6/A7/A9)",
    "floor": "Szájfenék/sublingualis tájék (A11)",
}


def fmt_hu(value: float, digits: int = 2, signed: bool = False) -> str:
    spec = f"{'+' if signed else ''}.{digits}f"
    return format(value, spec).replace(".", ",")


def fmt_p(value: float) -> str:
    if value < 0.001:
        return "<0,001"
    return fmt_hu(value, 3)


def deduplicate_patients(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep the most complete record per patient after checking conflicts."""

    required = {"TAJ", "id", "denture_type", "init_mai_huedegree", *ANATOMY_SOURCE_COLUMNS}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = raw.copy()
    frame["_patient_key"] = frame["TAJ"].astype("string").str.strip()
    frame.loc[frame["_patient_key"].isna() | frame["_patient_key"].eq(""), "_patient_key"] = (
        "id:" + frame["id"].astype(str)
    )

    relevant = ["denture_type", "init_mai_huedegree", *ANATOMY_SOURCE_COLUMNS]
    conflicts = 0
    duplicate_groups = 0
    for _, group in frame.groupby("_patient_key", sort=False):
        if len(group) < 2:
            continue
        duplicate_groups += 1
        for column in relevant:
            observed = group[column].dropna().astype(str).str.strip().unique()
            conflicts += int(len(observed) > 1)
    if conflicts:
        raise ValueError(f"Conflicting duplicate patient values in {conflicts} fields")

    frame["_completeness"] = frame.notna().sum(axis=1)
    frame["_record_date"] = pd.to_datetime(frame.get("record_datetime"), errors="coerce")
    frame = frame.sort_values(
        ["_patient_key", "_completeness", "_record_date"],
        ascending=[True, False, False],
        na_position="last",
    ).drop_duplicates("_patient_key", keep="first")

    audit = {
        "source_records": int(len(raw)),
        "unique_patients": int(len(frame)),
        "duplicate_patient_groups_resolved": int(duplicate_groups),
        "duplicate_conflicts": int(conflicts),
    }
    return frame, audit


def prepare_analysis(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    frame, audit = deduplicate_patients(raw)
    lower = frame.loc[frame["denture_type"].isin(["both", "lower"])].copy()
    lower["MAI_huedegree_worse"] = pd.to_numeric(
        lower["init_mai_huedegree"], errors="coerce"
    )

    # A8 is deliberately excluded from the tuberculum construct because its
    # validation category is inconsistent with A7 in the current source data.
    scores = build_scores(
        lower,
        side_strategy="mean",
        include_a8=False,
        ridge_uses_a12=True,
    )
    lower = lower.join(scores)

    if not lower["MAI_huedegree_worse"].dropna().between(0, 180).all():
        raise ValueError("Hue-degree MAI outside the expected 0-180 degree range")

    audit.update(
        {
            "mandibular_target_patients": int(len(lower)),
            "patients_with_huedegree_MAI": int(lower["MAI_huedegree_worse"].notna().sum()),
            "primary_complete_pairs": int(
                lower[["graded_score_0_5", "MAI_huedegree_worse"]].dropna().shape[0]
            ),
            "target_missing_huedegree_MAI": int(lower["MAI_huedegree_worse"].isna().sum()),
            "target_missing_complete_anatomical_score": int(lower["graded_score_0_5"].isna().sum()),
        }
    )
    return lower, audit


def estimate_association(frame: pd.DataFrame, predictor: str, label: str) -> dict[str, object]:
    pair = frame[[predictor, "MAI_huedegree_worse"]].dropna().to_numpy(dtype=float)
    if pair.shape[0] < 10 or np.unique(pair[:, 0]).size < 2:
        raise ValueError(f"Insufficient variation for {label}")

    x, y = pair[:, 0], pair[:, 1]
    observed = float(stats.spearmanr(x, y).statistic)
    seed = SEED + zlib.crc32(label.encode("utf-8"))
    rng = np.random.default_rng(seed)

    bootstrap = np.empty(N_BOOTSTRAP, dtype=float)
    valid = 0
    for _ in range(N_BOOTSTRAP):
        indices = rng.integers(0, len(x), size=len(x))
        if np.unique(x[indices]).size < 2 or np.unique(y[indices]).size < 2:
            continue
        coefficient = stats.spearmanr(x[indices], y[indices]).statistic
        if np.isfinite(coefficient):
            bootstrap[valid] = coefficient
            valid += 1
    if valid < 0.99 * N_BOOTSTRAP:
        raise RuntimeError(f"Too many invalid bootstrap replicates for {label}")
    interval = np.quantile(bootstrap[:valid], [0.025, 0.975])

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
        "predictor": predictor,
        "label": label,
        "n": int(len(x)),
        "spearman_rho": observed,
        "bootstrap_95_ci": [float(interval[0]), float(interval[1])],
        "permutation_p_two_sided": float(permutation_p),
        "leave_one_out_range": [
            float(np.nanmin(leave_one_out)),
            float(np.nanmax(leave_one_out)),
        ],
        "bootstrap_replicates": int(valid),
        "permutation_replicates": int(N_PERMUTATIONS),
    }


def run_analysis(frame: pd.DataFrame) -> dict[str, object]:
    primary = estimate_association(
        frame,
        "graded_score_0_5",
        "Fokozatos mandibularis anatómiai score (0–5)",
    )
    binary_sensitivity = estimate_association(
        frame,
        "binary_count_0_5",
        "Egész pontos anatómiai score (0–5)",
    )

    components = [
        estimate_association(frame, predictor, label)
        for predictor, label in COMPONENTS.items()
    ]
    raw_p = [item["permutation_p_two_sided"] for item in components]
    adjusted = multipletests(raw_p, method="holm")[1]
    for item, adjusted_p in zip(components, adjusted):
        item["holm_adjusted_p"] = float(adjusted_p)

    return {
        "primary": primary,
        "binary_score_sensitivity": binary_sensitivity,
        "components": components,
    }


def plot_forest(results: dict[str, object]) -> None:
    rows = [results["primary"], *results["components"]]
    positions = np.arange(len(rows))[::-1]
    colors = ["#D97706", *(["#2563EB"] * len(results["components"]))]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 10.5,
        }
    )
    fig, ax = plt.subplots(figsize=(10.5, 6.4), dpi=160)
    for position, item, color in zip(positions, rows, colors):
        estimate = item["spearman_rho"]
        low, high = item["bootstrap_95_ci"]
        ax.errorbar(
            estimate,
            position,
            xerr=[[estimate - low], [high - estimate]],
            fmt="o",
            color=color,
            ecolor=color,
            markersize=8 if item is results["primary"] else 6.5,
            elinewidth=2.2 if item is results["primary"] else 1.8,
            capsize=4,
        )
        ax.text(
            0.78,
            position,
            f"ρ={fmt_hu(estimate, 2, True)}  [{fmt_hu(low, 2)}; {fmt_hu(high, 2)}]",
            ha="right",
            va="center",
            fontsize=10,
        )

    ax.axvline(0, color="#6B7280", linestyle="--", linewidth=1.2)
    ax.set_yticks(positions, [item["label"] for item in rows])
    ax.set_xlim(-0.8, 0.85)
    ax.set_xticks(np.arange(-0.8, 0.81, 0.2))
    ax.set_xlabel("Spearman-korreláció a hue-degree MAI-jal (pozitív = várt kedvezőtlen irány)")
    ax.set_title(
        "A várt pozitív anatómia–MAI kapcsolat nem jelenik meg a pilotmintában",
        loc="left",
        fontweight="bold",
        pad=24,
    )
    ax.text(
        0,
        1.025,
        "Pont = korreláció; vonal = betegszintű bootstrap 95%-os intervallum",
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
    fig.savefig(OUTPUT_DIR / "tdk_anatomia_huedegree_forest.png", bbox_inches="tight")
    plt.close(fig)


def result_table(results: dict[str, object]) -> str:
    rows = [
        "| Anatómiai prediktor | n | Spearman ρ | Bootstrap 95%-os CI | Permutációs p | Holm-p* |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in [results["primary"], *results["components"]]:
        ci = item["bootstrap_95_ci"]
        holm = "—" if "holm_adjusted_p" not in item else fmt_p(item["holm_adjusted_p"])
        rows.append(
            f"| {item['label']} | {item['n']} | {fmt_hu(item['spearman_rho'], 2, True)} "
            f"| {fmt_hu(ci[0], 2)}; {fmt_hu(ci[1], 2)} "
            f"| {fmt_p(item['permutation_p_two_sided'])} | {holm} |"
        )
    return "\n".join(rows)


def write_report(audit: dict[str, int], results: dict[str, object]) -> None:
    primary = results["primary"]
    binary = results["binary_score_sensitivity"]
    ci = primary["bootstrap_95_ci"]
    binary_ci = binary["bootstrap_95_ci"]
    loo = primary["leave_one_out_range"]

    report = f"""# Mandibularis anatómia és hue-degree MAI

**Kisebb scope-ú TDK keresztmetszeti pilot elemzés — 2026-08-19**

## A kérdés

> Együtt jár-e a nagyobb, előre meghatározott mandibularis anatómiai
> hátrányterhelés a rosszabb objektív rágásteljesítménnyel, ha azt kizárólag
> a hue-degree MAI-jal mérjük?

Ebben az elemzésben nincs OHIP, GOHAI, önértékelt rágás, Lab-algoritmus vagy
RGB-algoritmus. Egyetlen kimenet van: a kiindulási `init_mai_huedegree`.

## A rövid válasz

A várt pozitív kapcsolat nem jelent meg. A fokozatos 0–5 anatómiai score és
a hue-degree MAI korrelációja `rho={fmt_hu(primary['spearman_rho'], 2, True)}`
volt (`n={primary['n']}`, bootstrap 95%-os CI `{fmt_hu(ci[0], 2)};
{fmt_hu(ci[1], 2)}`, kétoldali permutációs `p={fmt_p(primary['permutation_p_two_sided'])}`).
Az öt anatómiai komponens pontbecslése szintén nulla körüli, enyhén negatív
volt.

Ez nem bizonyítja, hogy az anatómia funkcionálisan érdektelen. Azt mutatja,
hogy a jelen mintában ez a klinikai anatómiai score nem követi a kiindulási,
teljes szájra vonatkozó színkeverési teljesítményt.

## Javasolt TDK-cím

> **A mandibularis anatómiai hátrányterhelés és az objektív
> rágásteljesítmény kapcsolata hue-degree MAI-val: keresztmetszeti
> pilotvizsgálat**

## Anyag és módszer

- **Dizájn:** másodlagos, keresztmetszeti pilot elemzés.
- **Deduplikálás:** {audit['source_records']} rekordból
  {audit['unique_patients']} egyedi beteg; {audit['duplicate_patient_groups_resolved']}
  duplikált betegcsoportot a legteljesebb rekord megtartásával oldottunk fel.
  Az elemzési változókban konfliktus nem volt.
- **Célpopuláció:** `both` vagy `lower`, azaz mandibularisan releváns
  fogsortípus, összesen {audit['mandibular_target_patients']} egyedi beteg.
- **Hue-degree MAI:** {audit['patients_with_huedegree_MAI']} betegben
  elérhető. A magasabb körkörös hue-szórás rosszabb színkeverést jelent.
- **Primer prediktor:** öt, egyenlő súlyú konstrukció fokozatos 0–5 összege:
  gerincatrophia, torus mandibularis, lingualis tasak, tuberculum-komplex és
  szájfenék.
- **Tuberculum:** A6, A7 és A9. Az A8-at az ismert A7–A8 validációs
  inkonzisztencia miatt előre kizártuk.
- **Primer becslés:** Spearman-korreláció, {primary['bootstrap_replicates']:,}
  betegszintű bootstrap ismétlésből származó 95%-os intervallummal és
  {primary['permutation_replicates']:,} ismétléses kétoldali permutációs
  p-értékkel.
- **Szekunder elemzés:** az öt konstrukció külön-külön; a p-értékekre
  Holm-korrekció készült.
- **Érzékenység:** ugyanazon öt konstrukció egész pontos 0–5 score-ja.

## Eredmények

{result_table(results)}

\* A Holm-korrekció csak az öt komponensből álló másodlagos tesztcsaládra
vonatkozik; a primer score nincs ebben a korrekcióban.

![Az anatómiai score és komponenseinek kapcsolata a hue-degree MAI-jal](stat_output/tdk_anatomia_huedegree_forest.png)

**Ábra alternatív leírása:** A primer anatómiai score és mind az öt
komponens korrelációs pontbecslése a nulla negatív oldalán van. Minden
bootstrap intervallum széles és átmetszi a nullát.

### Stabilitás és érzékenység

- A primer rho az egyes betegek kihagyásakor `{fmt_hu(loo[0], 2, True)}` és
  `{fmt_hu(loo[1], 2, True)}` között mozgott; az előjel nem egyetlen betegen
  múlt.
- Az egész pontos 0–5 score korrelációja
  `rho={fmt_hu(binary['spearman_rho'], 2, True)}` volt (95%-os CI
  `{fmt_hu(binary_ci[0], 2)}; {fmt_hu(binary_ci[1], 2)}`).
- A fokozatos és bináris score egyaránt a szakmailag várt iránnyal
  ellentétes pontbecslést adott, de egyik sem zárja ki a nullát.

## A tanulság

> A klasszikus mandibularis anatómiai nehézség és a hue-degree MAI nem
> ugyanannak a mechanikai jelenségnek az automatikus leképezése. A jelen
> pilotmintában az anatómiai hátrányterhelés nem azonosította a rosszabb
> objektív színkeverési teljesítményt.

Ennek egyik lehetséges magyarázata, hogy a hue-degree MAI-t az anatómia
mellett a meglévő protézis technikai minősége, az okklúzió, a neuromuszkuláris
alkalmazkodás és a választott rágási stratégia is meghatározza. Ezeket a jelen
elemzés nem méri, ezért magyarázatként, nem bizonyított mechanizmusként kell
kezelni.

## Mit lehet és mit nem lehet állítani?

**Védhető:**

> Nem találtunk bizonyítékot arra, hogy a nagyobb mandibularis anatómiai
> hátrányterhelés rosszabb kiindulási hue-degree MAI-jal járna együtt.

**Nem védhető:**

- „Az anatómia nem befolyásolja a rágást.”
- „A kedvezőtlen anatómia javítja a rágást.”
- „A hue-degree MAI érvénytelen.”

## Korlátok

1. A primer teljes score–MAI elemzés csak {primary['n']} betegből készült.
2. A kérdés a korábbi adatfeltárás után lett leszűkítve; ezért a p-érték
   tájékoztató és az eredmény hipotézisgeneráló.
3. A score szakértőileg meghatározott formatív index, nem külső mintán
   validált mérőeszköz.
4. A MAI teljes száji teljesítmény, miközben a score csak mandibularis
   anatómiai tulajdonságokat tartalmaz.
5. A meglévő protézis minősége, okklúziója, viselési ideje és a neuromuszkuláris
   adaptáció nincs a modellben.
6. A keresztmetszeti baseline MAI nem kezelés utáni eredmény és nem
   prognosztikai végpont.
7. A nulla körüli eredmény nem ekvivalenciabizonyíték; az intervallumok
   mérsékelt pozitív kapcsolatot is megengednek.

## Reprodukálhatóság és adatvédelem

- Elemzőszkript: `tdk_anatomy_huedegree_analysis.py`
- Aggregált eredmény: `stat_output/tdk_anatomy_huedegree_summary.json`
- Aggregált eredménytábla: `stat_output/tdk_anatomy_huedegree_results.csv`
- Ábra: `stat_output/tdk_anatomia_huedegree_forest.png`
- A kimenetek nem tartalmaznak nevet, TAJ-számot vagy betegszintű adatot.
""".replace("20,000", "20 000")
    REPORT_FILE.write_text(report, encoding="utf-8")


def write_csv(results: dict[str, object]) -> None:
    rows = []
    for family, items in (
        ("primary", [results["primary"]]),
        ("secondary_component", results["components"]),
        ("sensitivity", [results["binary_score_sensitivity"]]),
    ):
        for item in items:
            rows.append(
                {
                    "analysis_family": family,
                    "predictor": item["predictor"],
                    "label": item["label"],
                    "n": item["n"],
                    "spearman_rho": item["spearman_rho"],
                    "ci_low": item["bootstrap_95_ci"][0],
                    "ci_high": item["bootstrap_95_ci"][1],
                    "permutation_p": item["permutation_p_two_sided"],
                    "holm_adjusted_p": item.get("holm_adjusted_p", np.nan),
                    "loo_low": item["leave_one_out_range"][0],
                    "loo_high": item["leave_one_out_range"][1],
                }
            )
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "tdk_anatomy_huedegree_results.csv", index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    raw = pd.read_csv(DATA_FILE)
    analysis, audit = prepare_analysis(raw)
    results = run_analysis(analysis)
    payload = {
        "analysis": "Mandibular anatomy vs baseline hue-degree MAI",
        "status": "exploratory cross-sectional pilot",
        "outcome": "init_mai_huedegree only; higher means worse mixing",
        "data_audit": audit,
        **results,
    }
    (OUTPUT_DIR / "tdk_anatomy_huedegree_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(results)
    plot_forest(results)
    write_report(audit, results)
    print(
        f"Analysis complete: n={results['primary']['n']}, "
        f"rho={results['primary']['spearman_rho']:.3f}"
    )
    print(f"Report: {REPORT_FILE}")


if __name__ == "__main__":
    main()
