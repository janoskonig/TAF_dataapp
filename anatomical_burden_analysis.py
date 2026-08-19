"""Exploratory mandibular anatomical adversity-burden analysis.

Purpose
-------
Construct a transparent, expert-informed formative index from five distinct
mandibular anatomical mechanisms and examine its cross-sectional association
with baseline oral-health-related quality of life and chewing outcomes.

This analysis is explicitly exploratory.  It does not validate a prognostic
score and it does not use outcome-derived weights or cut-points.

Privacy
-------
Patient identifiers are used only in memory to recognise the digital-model
subgroup.  No patient-level or identifying data are written to output files.
"""

from __future__ import annotations

import hashlib
import json
import re
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import norm, spearmanr


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "patients.csv"
OUT = ROOT / "stat_output"
DIGITAL_ROOT = Path(
    "/Volumes/T7/TAF KUTATÁS 2024/éles/modellanalízis backup mentés"
)
SEED = 20260819
N_BOOT = 10_000
N_PERM = 20_000


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def parse_decimal(value: object) -> float:
    if pd.isna(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    try:
        parsed = json.loads(str(value))
        return float(parsed.get("$decimal", np.nan))
    except (TypeError, ValueError, json.JSONDecodeError, AttributeError):
        return np.nan


def strict_row_sum(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    return values.sum(axis=1, min_count=len(columns))


def row_max(frame: pd.DataFrame) -> pd.Series:
    result = frame.max(axis=1, skipna=True)
    result[frame.notna().sum(axis=1) == 0] = np.nan
    return result


def bilateral_values(
    frame: pd.DataFrame, base: str, mapping: dict[int, float]
) -> pd.DataFrame:
    out = {}
    for side in ("jobb", "bal"):
        out[side] = numeric(frame[f"{base}_{side}"]).map(mapping)
    return pd.DataFrame(out, index=frame.index)


def bilateral_aggregate(
    frame: pd.DataFrame,
    base: str,
    mapping: dict[int, float],
    strategy: str,
) -> pd.Series:
    values = bilateral_values(frame, base, mapping)
    if strategy == "mean":
        result = values.mean(axis=1, skipna=True)
    elif strategy == "worst":
        result = values.max(axis=1, skipna=True)
    else:
        raise ValueError(f"Unknown bilateral strategy: {strategy}")
    result[values.notna().sum(axis=1) == 0] = np.nan
    return result


def bilateral_any(
    frame: pd.DataFrame, base: str, adverse_codes: set[int]
) -> pd.Series:
    values = pd.DataFrame(
        {
            side: numeric(frame[f"{base}_{side}"])
            for side in ("jobb", "bal")
        },
        index=frame.index,
    )
    result = values.isin(adverse_codes).any(axis=1).astype(float)
    result[values.notna().sum(axis=1) == 0] = np.nan
    return result


def build_scores(
    frame: pd.DataFrame,
    side_strategy: str = "mean",
    include_a8: bool = True,
    ridge_uses_a12: bool = True,
) -> pd.DataFrame:
    """Build binary and graded construct-level scores.

    Each construct contributes at most one point.  The binary tuberculum
    construct is adverse when the pre-specified graded block severity is at
    least 0.5; this cut-point is not estimated from an outcome.
    """

    score = pd.DataFrame(index=frame.index)

    a1 = numeric(frame["A1_Kaan"])
    a12 = numeric(frame["A12"])
    ridge_a1 = a1.map({1: 0.0, 2: 1 / 3, 3: 2 / 3, 4: 1.0, 5: 1.0})
    ridge_a12 = a12.map({1: 0.0, 2: 0.5, 3: 1.0})
    if ridge_uses_a12:
        score["ridge"] = row_max(pd.DataFrame({"A1": ridge_a1, "A12": ridge_a12}))
        ridge_binary = ((a1 >= 3) | (a12 >= 2)).astype(float)
        ridge_binary[a1.isna() & a12.isna()] = np.nan
    else:
        score["ridge"] = ridge_a1
        ridge_binary = (a1 >= 3).astype(float)
        ridge_binary[a1.isna()] = np.nan

    score["torus"] = bilateral_any(frame, "A4", {2, 3})
    score["lingual"] = bilateral_aggregate(
        frame, "A5", {2: 0.0, 1: 0.5, 3: 1.0}, side_strategy
    )

    tuberculum_maps = {
        "A6": {1: 0.0, 2: 0.5, 3: 1.0},
        "A7": {1: 0.0, 2: 0.5, 3: 1.0},
        "A8": {1: 0.0, 2: 1.0, 3: 1.0},
        "A9": {1: 0.0, 2: 0.5, 3: 1.0},
    }
    tuberculum_items = ["A6", "A7", "A9"]
    if include_a8:
        tuberculum_items.insert(2, "A8")
    tuberculum_parts = pd.DataFrame(
        {
            item: bilateral_aggregate(
                frame, item, tuberculum_maps[item], side_strategy
            )
            for item in tuberculum_items
        },
        index=frame.index,
    )
    score["tuberculum"] = tuberculum_parts.mean(axis=1, skipna=False)
    score["floor"] = numeric(frame["A11"]).map({2: 0.0, 1: 0.5, 3: 1.0})

    score["ridge_binary"] = ridge_binary
    score["torus_binary"] = score["torus"]
    score["lingual_binary"] = bilateral_any(frame, "A5", {3})
    score["tuberculum_binary"] = (score["tuberculum"] >= 0.5).astype(float)
    score.loc[score["tuberculum"].isna(), "tuberculum_binary"] = np.nan
    floor_raw = numeric(frame["A11"])
    score["floor_binary"] = (floor_raw == 3).astype(float)
    score.loc[floor_raw.isna(), "floor_binary"] = np.nan

    graded = ["ridge", "torus", "lingual", "tuberculum", "floor"]
    binary = [f"{name}_binary" for name in graded]
    score["graded_score_0_5"] = score[graded].sum(axis=1, min_count=5)
    score["binary_count_0_5"] = score[binary].sum(axis=1, min_count=5)
    return score


def digital_taj_set(root: Path) -> set[str]:
    ids: set[str] = set()
    if not root.exists():
        return ids
    for path in root.iterdir():
        digits = "".join(re.findall(r"\d", path.name))
        if len(digits) >= 9:
            ids.add(digits[-9:])
    return ids


def normalise_taj(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(r"\D", "", regex=True).str.zfill(9)


def pairwise_xy(x: pd.Series, y: pd.Series) -> np.ndarray:
    return pd.DataFrame({"x": x, "y": y}).dropna().to_numpy(float)


def bootstrap_spearman_ci(
    values: np.ndarray, label: str, n_boot: int = N_BOOT
) -> tuple[float, float]:
    seed = SEED + zlib.crc32(label.encode("utf-8"))
    rng = np.random.default_rng(seed)
    n = len(values)
    draws: list[float] = []
    for _ in range(n_boot):
        sample = values[rng.integers(0, n, n)]
        if np.unique(sample[:, 0]).size < 2 or np.unique(sample[:, 1]).size < 2:
            continue
        draws.append(float(spearmanr(sample[:, 0], sample[:, 1]).statistic))
    if not draws:
        return np.nan, np.nan
    return tuple(np.quantile(draws, [0.025, 0.975]).astype(float))


def permutation_spearman_p(
    values: np.ndarray, observed: float, label: str, n_perm: int = N_PERM
) -> float:
    seed = SEED + 1_000_000 + zlib.crc32(label.encode("utf-8"))
    rng = np.random.default_rng(seed)
    x = values[:, 0]
    y = values[:, 1]
    extreme = 0
    for _ in range(n_perm):
        rho = float(spearmanr(x, rng.permutation(y)).statistic)
        extreme += abs(rho) >= abs(observed) - 1e-15
    return (1 + extreme) / (n_perm + 1)


def association(
    x: pd.Series, y: pd.Series, label: str, run_resampling: bool = True
) -> dict[str, object]:
    values = pairwise_xy(x, y)
    n = len(values)
    if n < 8 or np.unique(values[:, 0]).size < 2 or np.unique(values[:, 1]).size < 2:
        return {"label": label, "n": n, "status": "insufficient_data"}
    rho = float(spearmanr(values[:, 0], values[:, 1]).statistic)
    if run_resampling:
        lo, hi = bootstrap_spearman_ci(values, label)
        p_perm = permutation_spearman_p(values, rho, label)
    else:
        lo, hi, p_perm = np.nan, np.nan, np.nan
    return {
        "label": label,
        "n": n,
        "status": "ok",
        "rho": rho,
        "ci_low": lo,
        "ci_high": hi,
        "permutation_p_two_sided": p_perm,
    }


def adjusted_ols(
    frame: pd.DataFrame, predictor: str, outcome: str
) -> dict[str, object]:
    work = frame[[predictor, outcome, "age", "male", "single_arch"]].dropna().copy()
    base = {"predictor": predictor, "outcome": outcome, "n": len(work)}
    if len(work) < 15 or work[predictor].nunique() < 2:
        return base | {"status": "insufficient_data"}
    work["age_z"] = (work["age"] - work["age"].mean()) / work["age"].std(ddof=0)
    design = sm.add_constant(
        work[[predictor, "age_z", "male", "single_arch"]], has_constant="add"
    )
    fit = sm.OLS(work[outcome], design).fit(cov_type="HC3")
    beta = float(fit.params[predictor])
    se = float(fit.bse[predictor])
    lo, hi = beta + norm.ppf([0.025, 0.975]) * se
    return base | {
        "status": "ok",
        "beta_per_point": beta,
        "robust_se": se,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_two_sided": float(fit.pvalues[predictor]),
        "r_squared": float(fit.rsquared),
    }


def round_value(value: object, digits: int = 3) -> object:
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return round(float(value), digits)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def format_hu(value: float, digits: int = 2, signed: bool = False) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    pattern = f"{{:{'+' if signed else ''}.{digits}f}}"
    return pattern.format(value).replace(".", ",")


def make_figures(frame: pd.DataFrame, results: dict[str, object]) -> list[Path]:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    blue = "#2C7FB8"
    orange = "#F28E2B"
    grey = "#5B6573"

    dist = (
        frame["binary_count_0_5"]
        .dropna()
        .astype(int)
        .value_counts()
        .reindex(range(6), fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.bar(dist.index, dist.values, color=blue, edgecolor="white", width=0.78)
    for bar, count in zip(bars, dist.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + 0.25,
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=12,
            color=grey,
        )
    ax.set_title("Az anatómiai hátrányterhelés a teljes 0–5 tartományt lefedi")
    ax.set_xlabel("Kedvezőtlen mandibularis konstrukciók száma (0–5)")
    ax.set_ylabel("Betegek száma")
    ax.set_xticks(range(6))
    ax.set_ylim(0, max(dist.values) + 3)
    ax.grid(axis="x", visible=False)
    fig.text(0.01, 0.01, "Forrás: PREDICT baseline adatpillanat; n=40 teljes score", fontsize=9, color=grey)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path_distribution = OUT / "anatomiai_hatranyterheles_score_eloszlas.png"
    fig.savefig(path_distribution, dpi=300, bbox_inches="tight")
    plt.close(fig)

    plot = frame[["binary_count_0_5", "OHIP_sum"]].dropna().copy()
    rng = np.random.default_rng(SEED)
    plot["x_jitter"] = plot["binary_count_0_5"] + rng.uniform(-0.12, 0.12, len(plot))
    summary = plot.groupby("binary_count_0_5")["OHIP_sum"].agg(
        median="median", q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75)
    )
    primary = results["associations"]["primary_binary_vs_OHIP"]
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ax.scatter(
        plot["x_jitter"],
        plot["OHIP_sum"],
        s=55,
        alpha=0.65,
        color=blue,
        edgecolor="white",
        linewidth=0.5,
        label="Beteg",
    )
    ax.errorbar(
        summary.index,
        summary["median"],
        yerr=[summary["median"] - summary["q1"], summary["q3"] - summary["median"]],
        fmt="D",
        markersize=7,
        color=orange,
        ecolor=orange,
        elinewidth=2,
        capsize=5,
        label="Medián és IQR",
        zorder=4,
    )
    ax.set_title("Gyenge, bizonytalan OHIP-trend a nagyobb anatómiai teher mellett")
    ax.set_xlabel("Kedvezőtlen mandibularis konstrukciók száma (0–5)", fontsize=13)
    ax.set_ylabel("OHIP-5 összpontszám\n(0–20; magasabb = rosszabb)", fontsize=13, labelpad=10)
    ax.set_xticks(range(6))
    ax.set_ylim(-0.5, 20.5)
    ax.legend(frameon=False, loc="upper left")
    annotation = (
        f"Spearman ρ = {format_hu(primary['rho'])}\n"
        f"bootstrap 95% CI: {format_hu(primary['ci_low'])} – "
        f"{format_hu(primary['ci_high'])}\n"
        f"n = {primary['n']}"
    )
    ax.text(
        0.98,
        0.96,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#D0D5DD"},
    )
    fig.text(0.17, 0.025, "Keresztmetszeti, exploratív kapcsolat; nem prognosztikai validáció", fontsize=9, color=grey)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.18, top=0.88)
    path_ohip = OUT / "anatomiai_hatranyterheles_ohip.png"
    fig.savefig(path_ohip, dpi=300, bbox_inches="tight")
    plt.close(fig)

    rows = []
    labels = {
        "primary_binary_vs_OHIP": "Egész pontos score → OHIP",
        "graded_vs_OHIP": "Fokozatos score → OHIP",
        "primary_binary_vs_GOHAI_worse": "Egész pontos score → GOHAI (rosszabb)",
        "primary_binary_vs_MAI": "Egész pontos score → MAI",
        "primary_binary_vs_chewing_worse": "Egész pontos score → önbevallott rágás (rosszabb)",
    }
    for key, label in labels.items():
        result = results["associations"][key]
        rows.append(
            {
                "label": label,
                "rho": result["rho"],
                "lo": result["ci_low"],
                "hi": result["ci_high"],
                "n": result["n"],
            }
        )
    forest = pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 6.2))
    y = np.arange(len(forest))
    ax.axvline(0, color="#98A2B3", linewidth=1.2, linestyle="--")
    ax.errorbar(
        forest["rho"],
        y,
        xerr=[forest["rho"] - forest["lo"], forest["hi"] - forest["rho"]],
        fmt="o",
        color=blue,
        ecolor=blue,
        markersize=8,
        elinewidth=2,
        capsize=4,
    )
    ax.set_yticks(y, [f"{label}  (n={n})" for label, n in zip(forest["label"], forest["n"])])
    ax.set_xlim(-0.75, 0.75)
    ax.set_xlabel("Spearman-korreláció (ρ), bootstrap 95%-os intervallummal")
    ax.set_title("A score nem mutat egységes kapcsolatot minden kimenettel")
    ax.grid(axis="y", visible=False)
    fig.text(0.01, 0.01, "Pozitív érték: nagyobb anatómiai teher mellett rosszabb kimenet", fontsize=9, color=grey)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path_forest = OUT / "anatomiai_hatranyterheles_kimenetek.png"
    fig.savefig(path_forest, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [path_distribution, path_ohip, path_forest]


def make_report(results: dict[str, object], report_path: Path) -> None:
    associations = results["associations"]
    p = associations["primary_binary_vs_OHIP"]
    g = associations["graded_vs_OHIP"]
    gohai = associations["primary_binary_vs_GOHAI_worse"]
    mai = associations["primary_binary_vs_MAI"]
    chew = associations["primary_binary_vs_chewing_worse"]
    adjusted = results["adjusted_models"]["binary_count_OHIP"]
    digital = associations["digital_binary_vs_OHIP"]
    distribution = results["score_distribution"]
    prevalence = results["binary_component_prevalence"]
    sensitivity = results["score_sensitivity"]
    leave_one_out = results["leave_one_construct_out"]

    dist_text = ", ".join(f"{k} pont: {v}" for k, v in distribution.items())
    component_rows = "\n".join(
        f"| {name} | {value['events']}/{value['n']} | {format_hu(value['proportion'] * 100, 1)}% |"
        for name, value in prevalence.items()
    )
    sensitivity_rows = "\n".join(
        f"| {row['specification']} | {row['n']} | {format_hu(row['rho'], 2, True)} | "
        f"{format_hu(row['ci_low'])}; {format_hu(row['ci_high'])} |"
        for row in sensitivity
    )
    leave_out_values = [row["rho"] for row in leave_one_out.values()]
    leave_out_min = min(leave_out_values)
    leave_out_max = max(leave_out_values)
    no_torus = leave_one_out["torus_binary"]

    text = f"""# Mandibularis anatómiai hátrányterhelési index

**PREDICT keresztmetszeti pilot elemzés — 2026-08-19**

## Rövid eredmény

Az előre rögzített, ötkonstrukciós egész pontos anatómiai score a mandibularis
mintában technikailag jól számítható és megfelelő szórást mutatott. A nagyobb
anatómiai teher a rosszabb OHIP-5 felé mutatott, de a bizonytalansági
intervallum a nullát is tartalmazta: `rho={format_hu(p['rho'], 2, True)}`
(`n={p['n']}`, bootstrap 95%-os CI `{format_hu(p['ci_low'])};
{format_hu(p['ci_high'])}`, kétoldali permutációs `p={format_hu(p['permutation_p_two_sided'], 3)}`).

A fokozatos 0–5 score valamivel erősebb OHIP-jelet adott
(`rho={format_hu(g['rho'], 2, True)}`, 95%-os CI `{format_hu(g['ci_low'])};
{format_hu(g['ci_high'])}`), de ez is exploratív. A GOHAI, a MAI és az
önbevallott rágóképesség nem mutatott koherens, azonos irányú összefüggést.

## Kutatási kérdés

> Együtt jár-e a nagyobb, szakértőileg előre meghatározott mandibularis
> anatómiai hátrányterhelés a rosszabb kiindulási orális életminőséggel?

Ez keresztmetszeti asszociációs kérdés. A jelen elemzés nem igazolja, hogy a
score előre jelzi az elkészülő új fogsor eredményét.

## Populáció és kimenetek

- Forrás: `patients.csv`, 48 egyedi betegrekord.
- Mandibularis elemzési populáció: `both` vagy `lower`, összesen
  **{results['cohort']['lower_or_both_n']} beteg**.
- Teljes ötkonstrukciós score: **{results['cohort']['complete_score_n']} beteg**.
- Elsődleges kimenet: OHIP-5 összeg, 0–20, magasabb = rosszabb.
- Másodlagos kimenetek: GOHAI fordított irányban, MAI-huedegree és
  önbevallott rágóképesség fordított irányban, így minden elemzett kimenetnél
  a magasabb érték rosszabbat jelent.
- A digitális mintával rendelkező mandibularis alcsoport külön pilotként
  szerepel; az A2 modellanalízis eredménye még nem része a score-nak.

## Score-definíció

Az index formatív: különböző anatómiai hátrányokat összegez, ezért az
összetevőknek nem kell magas belső konzisztenciát mutatniuk. Minden konstrukció
legfeljebb egy pontot ér.

| Konstrukció | Egész pontos kedvezőtlen definíció |
|---|---|
| Gerincatrophia | A1 >= 3 vagy A12 >= 2; A1/A2/A12 együtt is legfeljebb egy pont |
| Torus mandibularis | A4=2 vagy 3 legalább az egyik oldalon |
| Lingualis tasak | A5=3 legalább az egyik oldalon |
| Tuberculum | A6–A9 közös, 0–1-re skálázott blokkátlaga >= 0,5 |
| Szájfenék | A11=3 |

A fokozatos érzékenységi score minden konstrukciót 0–1 között kódol, és az
öt konstrukciót azonos súllyal összeadja. A kétoldali változóknál az elsődleges
fokozatos specifikáció a két oldal átlagát használja; a rosszabbik oldalas és
az A8 nélküli változat külön érzékenységi elemzés.

## Score-eloszlás

{dist_text}.

| Konstrukció | Kedvezőtlen / mérhető | Prevalencia |
|---|---:|---:|
{component_rows}

## Elsődleges eredmény: OHIP-5

- Spearman-korreláció: `rho={format_hu(p['rho'], 2, True)}`.
- Bootstrap 95%-os intervallum: `{format_hu(p['ci_low'])}; {format_hu(p['ci_high'])}`.
- Kétoldali permutációs p-érték: `{format_hu(p['permutation_p_two_sided'], 3)}`.
- Életkorra, nemre és egy- vs kétállcsontos fogsortípusra korrigált HC3-robusztus
  lineáris modell: egy további hátránypont mellett az OHIP becsült változása
  `{format_hu(adjusted['beta_per_point'], 2, True)}` pont
  (95%-os CI `{format_hu(adjusted['ci_low'])}; {format_hu(adjusted['ci_high'])}`;
  `n={adjusted['n']}`).

Az irány mindkét modellben a szakmailag várt, de a becslés pontatlan; a nulla
és egy klinikailag értelmezhető pozitív kapcsolat egyaránt összeegyeztethető
az adatokkal.

## Másodlagos kimenetek

| Kimenet | n | Spearman rho | Bootstrap 95%-os CI |
|---|---:|---:|---:|
| GOHAI, magasabb = rosszabb | {gohai['n']} | {format_hu(gohai['rho'], 2, True)} | {format_hu(gohai['ci_low'])}; {format_hu(gohai['ci_high'])} |
| MAI, magasabb = rosszabb | {mai['n']} | {format_hu(mai['rho'], 2, True)} | {format_hu(mai['ci_low'])}; {format_hu(mai['ci_high'])} |
| Önbevallott rágás, magasabb = rosszabb | {chew['n']} | {format_hu(chew['rho'], 2, True)} | {format_hu(chew['ci_low'])}; {format_hu(chew['ci_high'])} |

A score tehát nem viselkedik univerzális kimeneti prediktorként. Ez
értelmezhető úgy, hogy az anatómiai mechanizmusok eltérően jelennek meg a
beteg által megélt életminőségben és a színkeverési tesztben, illetve hogy a
kis minta mellett a becslések instabilak.

## Score-specifikációs érzékenység OHIP-re

| Specifikáció | n | Spearman rho | Bootstrap 95%-os CI |
|---|---:|---:|---:|
{sensitivity_rows}

Az eredmény iránya nem egyetlen oldal-összevonási szabálytól függ, de a
hatásnagyság érzékeny a pontos score-definícióra. Ezért a végleges új
adatgyűjtés előtt ugyanazt a szabályt kell protokollban rögzíteni.

A konstrukciónkénti elhagyásos ellenőrzésben az OHIP-korreláció
`{format_hu(leave_out_min, 2, True)}` és `{format_hu(leave_out_max, 2, True)}`
között mozgott. A torus-komponens elhagyásakor volt a legnagyobb
(`rho={format_hu(no_torus['rho'], 2, True)}`), összhangban a korábbi
ellenirányú A4 pilotjellel. Ez **nem indokolja a torus utólagos kihagyását**;
csak megmutatja, hogy melyik komponens csökkenti leginkább a globális score
OHIP-kapcsolatát.

## Digitális mintás alcsoport

A mandibularis digitális mintás alcsoportban {results['cohort']['digital_lower_or_both_n']}
beteg volt, közülük {digital['n']} teljes score–OHIP pár. Az egész pontos score
és OHIP korrelációja `rho={format_hu(digital['rho'], 2, True)}` volt
(95%-os CI `{format_hu(digital['ci_low'])}; {format_hu(digital['ci_high'])}`).
Ez a kis, szelektált alcsoport önmagában nem alkalmas validációra. Az A2
modellmérés később a gerincatrophia-komponenst helyettesítheti vagy
finomíthatja, de nem kaphat külön hatodik pontot.

## TDK-ban védhető következtetés

> A szakértőileg előre meghatározott, ötkonstrukciós mandibularis anatómiai
> hátrányterhelési index a jelen mintában megvalósítható és megfelelő
> variabilitást mutatott. A nagyobb anatómiai teher a rosszabb OHIP-5 felé
> mutatott, de az összefüggés bizonytalan volt, és nem ismétlődött meg
> koherensen minden másodlagos kimeneten. Az eredmények hipotézisgenerálók;
> az index prognosztikai értékét új, prospektív adatokon kell vizsgálni.

## Kötelező korlátok

1. A score-definíció a meglévő adatok előzetes megtekintése után, de a jelen
   formális futtatás előtt került rögzítésre; ezért ez nem prospektíven
   preregisztrált validáció.
2. Keresztmetszeti baseline kimenetekből nem állítható, hogy az anatómia
   előre jelzi az új fogsor sikerét vagy a kezelés nehézségét.
3. A minta kicsi, a bootstrap intervallumok szélesek, és a másodlagos
   összevetések többszörös exploráció részei.
4. A tuberculum blokk formatív és az A7–A8 adatkonzisztencia ismert problémája
   miatt külön A8 nélküli érzékenységi specifikáció készült.
5. Az azonos súlyok tartalmi döntést jelentenek; a jelen adatokból becsült
   súlyok túlillesztést okoznának.
6. A digitális alcsoport kiválasztottsága és kis elemszáma miatt annak
   eredménye csak pilot.

## Reprodukálhatóság

- Elemzőszkript: `anatomical_burden_analysis.py`
- Aggregált gépi eredmény: `stat_output/anatomiai_hatranyterheles_summary.json`
- QA-audit: `stat_output/anatomiai_hatranyterheles_audit.json`
- Ábrák: `stat_output/anatomiai_hatranyterheles_*.png`
- A kimenetek nem tartalmaznak nevet, TAJ-számot vagy betegszintű adatot.
"""
    report_path.write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    raw = pd.read_csv(DATA_FILE)
    if len(raw) != raw["id"].nunique():
        raise ValueError("The source contains duplicate patient record IDs.")

    frame = raw.copy()
    frame["OHIP_sum"] = strict_row_sum(frame, [f"OHIP_{i}" for i in range(1, 6)])
    frame["GOHAI_sum"] = strict_row_sum(frame, [f"GOHAI_{i}" for i in range(1, 13)])
    frame["GOHAI_worse"] = 60 - frame["GOHAI_sum"]
    frame["MAI_worse"] = frame["init_mai_huedegree"].map(parse_decimal)
    chewing_map = {"Nagyon rossz": 5, "Rossz": 4, "Átlagos": 3, "Jó": 2, "Kiváló": 1}
    frame["chewing_worse"] = frame["chewing_today_situation"].astype(str).str.strip().map(chewing_map)
    frame["QoL_composite_worse"] = (
        (frame["OHIP_sum"] - frame["OHIP_sum"].mean()) / frame["OHIP_sum"].std(ddof=0)
        + (frame["GOHAI_worse"] - frame["GOHAI_worse"].mean())
        / frame["GOHAI_worse"].std(ddof=0)
    ) / 2

    record_date = pd.to_datetime(frame["record_datetime"], errors="coerce")
    birth_date = pd.to_datetime(frame["birthdate"], errors="coerce")
    frame["age"] = (record_date - birth_date).dt.days / 365.25
    frame.loc[(frame["age"] < 18) | (frame["age"] > 105), "age"] = np.nan
    frame["male"] = frame["gender"].map({"Female": 0.0, "Male": 1.0})
    frame["single_arch"] = (frame["denture_type"] == "lower").astype(float)

    digital_ids = digital_taj_set(DIGITAL_ROOT)
    frame["digital_model"] = normalise_taj(frame["TAJ"]).isin(digital_ids)

    lower = frame[frame["denture_type"].isin(["both", "lower"])].copy()
    primary_score = build_scores(lower, side_strategy="mean", include_a8=True, ridge_uses_a12=True)
    lower = lower.join(primary_score)

    graded_worst = build_scores(lower, side_strategy="worst", include_a8=True, ridge_uses_a12=True)
    graded_no_a8 = build_scores(lower, side_strategy="mean", include_a8=False, ridge_uses_a12=True)
    a1_only = build_scores(lower, side_strategy="mean", include_a8=True, ridge_uses_a12=False)
    binary_no_a8 = build_scores(lower, side_strategy="mean", include_a8=False, ridge_uses_a12=True)

    association_specs = {
        "primary_binary_vs_OHIP": (lower["binary_count_0_5"], lower["OHIP_sum"]),
        "graded_vs_OHIP": (lower["graded_score_0_5"], lower["OHIP_sum"]),
        "primary_binary_vs_GOHAI_worse": (lower["binary_count_0_5"], lower["GOHAI_worse"]),
        "primary_binary_vs_MAI": (lower["binary_count_0_5"], lower["MAI_worse"]),
        "primary_binary_vs_chewing_worse": (lower["binary_count_0_5"], lower["chewing_worse"]),
        "primary_binary_vs_QoL_composite": (lower["binary_count_0_5"], lower["QoL_composite_worse"]),
        "graded_worst_side_vs_OHIP": (graded_worst["graded_score_0_5"], lower["OHIP_sum"]),
        "graded_without_A8_vs_OHIP": (graded_no_a8["graded_score_0_5"], lower["OHIP_sum"]),
        "binary_A1_only_ridge_vs_OHIP": (a1_only["binary_count_0_5"], lower["OHIP_sum"]),
        "binary_without_A8_vs_OHIP": (binary_no_a8["binary_count_0_5"], lower["OHIP_sum"]),
    }
    digital_mask = lower["digital_model"]
    association_specs["digital_binary_vs_OHIP"] = (
        lower.loc[digital_mask, "binary_count_0_5"], lower.loc[digital_mask, "OHIP_sum"]
    )
    associations = {
        label: association(x, y, label) for label, (x, y) in association_specs.items()
    }

    binary_components = [
        "ridge_binary",
        "torus_binary",
        "lingual_binary",
        "tuberculum_binary",
        "floor_binary",
    ]
    leave_one_out = {}
    for component in binary_components:
        other = [c for c in binary_components if c != component]
        score = lower[other].sum(axis=1, min_count=4)
        leave_one_out[component] = association(
            score, lower["OHIP_sum"], f"leave_out_{component}"
        )

    distribution = (
        lower["binary_count_0_5"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .reindex(range(6), fill_value=0)
        .astype(int)
        .to_dict()
    )
    component_labels = {
        "Gerincatrophia": "ridge_binary",
        "Torus mandibularis": "torus_binary",
        "Lingualis tasak": "lingual_binary",
        "Tuberculum-blokk": "tuberculum_binary",
        "Szájfenék": "floor_binary",
    }
    prevalence = {}
    for label, column in component_labels.items():
        measured = int(lower[column].notna().sum())
        events = int(lower[column].sum(skipna=True))
        prevalence[label] = {
            "events": events,
            "n": measured,
            "proportion": events / measured if measured else np.nan,
        }

    score_sensitivity = []
    sensitivity_keys = [
        ("Egész pontos, elsődleges", "primary_binary_vs_OHIP"),
        ("Fokozatos, oldalátlag", "graded_vs_OHIP"),
        ("Fokozatos, rosszabbik oldal", "graded_worst_side_vs_OHIP"),
        ("Fokozatos, A8 nélkül", "graded_without_A8_vs_OHIP"),
        ("Egész pontos, gerinc csak A1", "binary_A1_only_ridge_vs_OHIP"),
        ("Egész pontos, tuberculum A8 nélkül", "binary_without_A8_vs_OHIP"),
    ]
    for specification, key in sensitivity_keys:
        item = associations[key]
        score_sensitivity.append({"specification": specification, **item})

    complete = lower[lower["binary_count_0_5"].notna()]
    by_score = {}
    for score_value, group in complete.groupby("binary_count_0_5"):
        values = group["OHIP_sum"].dropna()
        by_score[str(int(score_value))] = {
            "n": int(len(values)),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "q1": float(values.quantile(0.25)),
            "q3": float(values.quantile(0.75)),
        }

    results: dict[str, object] = {
        "analysis_date": "2026-08-19",
        "status": "exploratory_cross_sectional_pilot",
        "cohort": {
            "source_n": int(len(raw)),
            "unique_record_id_n": int(raw["id"].nunique()),
            "lower_or_both_n": int(len(lower)),
            "complete_score_n": int(lower["binary_count_0_5"].notna().sum()),
            "complete_score_OHIP_n": int(lower[["binary_count_0_5", "OHIP_sum"]].dropna().shape[0]),
            "digital_folder_n": int(len(digital_ids)),
            "digital_matched_source_n": int(frame["digital_model"].sum()),
            "digital_lower_or_both_n": int(lower["digital_model"].sum()),
        },
        "score_distribution": {str(k): int(v) for k, v in distribution.items()},
        "score_descriptives": {
            "binary_mean": float(lower["binary_count_0_5"].mean()),
            "binary_median": float(lower["binary_count_0_5"].median()),
            "binary_q1": float(lower["binary_count_0_5"].quantile(0.25)),
            "binary_q3": float(lower["binary_count_0_5"].quantile(0.75)),
            "binary_min": float(lower["binary_count_0_5"].min()),
            "binary_max": float(lower["binary_count_0_5"].max()),
            "graded_mean": float(lower["graded_score_0_5"].mean()),
            "graded_median": float(lower["graded_score_0_5"].median()),
            "graded_q1": float(lower["graded_score_0_5"].quantile(0.25)),
            "graded_q3": float(lower["graded_score_0_5"].quantile(0.75)),
            "graded_min": float(lower["graded_score_0_5"].min()),
            "graded_max": float(lower["graded_score_0_5"].max()),
        },
        "binary_component_prevalence": prevalence,
        "OHIP_by_binary_score": by_score,
        "associations": associations,
        "adjusted_models": {
            "binary_count_OHIP": adjusted_ols(lower, "binary_count_0_5", "OHIP_sum"),
            "graded_score_OHIP": adjusted_ols(lower, "graded_score_0_5", "OHIP_sum"),
        },
        "score_sensitivity": score_sensitivity,
        "leave_one_construct_out": leave_one_out,
    }

    figures = make_figures(lower, results)
    results["outputs"] = {
        "figures": [path.name for path in figures],
        "report": "PREDICT_anatomiai_hatranyterheles_report.md",
    }
    summary_path = OUT / "anatomiai_hatranyterheles_summary.json"
    summary_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=round_value),
        encoding="utf-8",
    )

    source_hash = hashlib.sha256(DATA_FILE.read_bytes()).hexdigest()
    expected_distribution_total = sum(distribution.values())
    audit = {
        "source_file": DATA_FILE.name,
        "source_sha256": source_hash,
        "source_rows": int(len(raw)),
        "unique_record_ids": int(raw["id"].nunique()),
        "duplicate_record_ids": int(raw["id"].duplicated().sum()),
        "lower_or_both_rows": int(len(lower)),
        "complete_score_rows": int(lower["binary_count_0_5"].notna().sum()),
        "score_distribution_total": int(expected_distribution_total),
        "distribution_reconciles": bool(
            expected_distribution_total == lower["binary_count_0_5"].notna().sum()
        ),
        "score_bounds_valid": bool(
            lower["binary_count_0_5"].dropna().between(0, 5).all()
            and lower["graded_score_0_5"].dropna().between(0, 5).all()
        ),
        "strict_OHIP_complete_n": int(lower["OHIP_sum"].notna().sum()),
        "strict_GOHAI_complete_n": int(lower["GOHAI_sum"].notna().sum()),
        "MAI_complete_n": int(lower["MAI_worse"].notna().sum()),
        "invalid_age_excluded_n": int(
            (((record_date - birth_date).dt.days / 365.25 < 18)
             | ((record_date - birth_date).dt.days / 365.25 > 105)).sum()
        ),
        "digital_root_available": DIGITAL_ROOT.exists(),
        "digital_folder_n": int(len(digital_ids)),
        "digital_matched_source_n": int(frame["digital_model"].sum()),
        "pii_columns_written": [],
        "patient_level_rows_written": 0,
        "resampling": {"bootstrap": N_BOOT, "permutations": N_PERM, "seed": SEED},
    }
    (OUT / "anatomiai_hatranyterheles_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    make_report(results, ROOT / "PREDICT_anatomiai_hatranyterheles_report.md")

    primary = associations["primary_binary_vs_OHIP"]
    print("Mandibularis anatomy-burden analysis completed")
    print(f"Cohort: {len(lower)}; complete score: {results['cohort']['complete_score_n']}")
    print(
        "Primary binary score vs OHIP: "
        f"rho={primary['rho']:.3f}, 95% bootstrap CI "
        f"[{primary['ci_low']:.3f}, {primary['ci_high']:.3f}], "
        f"permutation p={primary['permutation_p_two_sided']:.4f}, n={primary['n']}"
    )
    print(f"Report: {ROOT / 'PREDICT_anatomiai_hatranyterheles_report.md'}")


if __name__ == "__main__":
    main()
