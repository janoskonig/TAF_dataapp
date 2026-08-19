#!/usr/bin/env python3
"""Reproducible, explicitly synthetic follow-up simulation and OR analysis.

This script is for pipeline validation and power/method demonstrations only.
It does not impute, replace, or represent observed patient follow-up data.
No database connection is opened and no real identifier is read or written.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from sklearn.metrics import roc_auc_score, roc_curve


ANATOMY_COLUMNS = [
    "F5",
    "F7",
    "F8",
    "A1_Kaan",
    "A3_jobb",
    "A3_bal",
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
    "A13",
    "A14",
]

OUTCOMES = {
    "ohip_meaningful_improvement": {
        "label": "Anchor-alapu szamottevo OHIP-5 javulas",
        "baseline": "OHIP_baseline_total",
        "score": "OHIP_change_improvement",
        "anchor": "responsiveness_change",
        "anchor_labels": ["Kicsit javult", "Sokat javult"],
    },
    "ohip_meaningful_deterioration": {
        "label": "Anchor-alapu szamottevo OHIP-5 romlas",
        "baseline": "OHIP_baseline_total",
        "score": "OHIP_change_deterioration",
        "anchor": "responsiveness_change",
        "anchor_labels": ["Kicsit romlott", "Sokat romlott"],
    },
    "gohai_meaningful_improvement": {
        "label": "Anchor-alapu szamottevo GOHAI javulas",
        "baseline": "GOHAI_baseline_total",
        "score": "GOHAI_change_improvement",
        "anchor": "responsiveness_change",
        "anchor_labels": ["Kicsit javult", "Sokat javult"],
    },
    "gohai_meaningful_deterioration": {
        "label": "Anchor-alapu szamottevo GOHAI romlas",
        "baseline": "GOHAI_baseline_total",
        "score": "GOHAI_change_deterioration",
        "anchor": "responsiveness_change",
        "anchor_labels": ["Kicsit romlott", "Sokat romlott"],
    },
    "mai_meaningful_improvement": {
        "label": "Anchor-alapu szamottevo MAI javulas",
        "baseline": "init_mai_huedegree",
        "score": "MAI_change_improvement",
        "anchor": "chewing_change",
        "anchor_labels": ["Kicsit javult", "Sokat javult"],
    },
    "mai_meaningful_deterioration": {
        "label": "Anchor-alapu szamottevo MAI romlas",
        "baseline": "init_mai_huedegree",
        "score": "MAI_change_deterioration",
        "anchor": "chewing_change",
        "anchor_labels": ["Kicsit romlott", "Sokat romlott"],
    },
}

EXPOSURE_DEFINITIONS = {
    "F5": "F5 >= 2 vs F5 = 1",
    "F7": "F7 >= 2 vs F7 = 1",
    "F8": "F8 >= 2 vs F8 = 1",
    "A1_Kaan": "A1_Kaan >= 2 vs A1_Kaan = 1",
    "A3_jobb": "A3_jobb in {2,3} vs 1",
    "A3_bal": "A3_bal in {2,3} vs 1",
    "A4_jobb": "A4_jobb in {2,3} vs 1",
    "A4_bal": "A4_bal in {2,3} vs 1",
    "A5_jobb": "A5_jobb in {1,3} vs 2",
    "A5_bal": "A5_bal in {1,3} vs 2",
    "A6_jobb": "A6_jobb in {2,3} vs 1",
    "A6_bal": "A6_bal in {2,3} vs 1",
    "A7_jobb": "A7_jobb in {2,3} vs 1",
    "A7_bal": "A7_bal in {2,3} vs 1",
    "A8_jobb": "A8_jobb in {2,3} vs 1",
    "A8_bal": "A8_bal in {2,3} vs 1",
    "A9_jobb": "A9_jobb in {2,3} vs 1",
    "A9_bal": "A9_bal in {2,3} vs 1",
    "A11": "A11 in {1,3} vs 2",
    "A12": "A12 in {2,3} vs 1",
    "A13": "A13 in {2,3} vs 1",
    "A14": "A14 >= 2 vs A14 = 1 (protetikai szituacio)",
}


def draw_coded_feature(
    rng: np.random.Generator,
    latent: np.ndarray,
    normal_code: int,
    unfavorable_codes: list[int],
    base_logit: float,
    loading: float = 0.65,
    unfavorable_weights: list[float] | None = None,
) -> np.ndarray:
    """Draw a categorical feature with a latent correlated burden."""
    p_unfavorable = expit(base_logit + loading * latent)
    unfavorable = rng.binomial(1, p_unfavorable).astype(bool)
    values = np.full(latent.shape[0], normal_code, dtype=int)
    weights = unfavorable_weights or [1.0 / len(unfavorable_codes)] * len(unfavorable_codes)
    values[unfavorable] = rng.choice(
        unfavorable_codes,
        size=int(unfavorable.sum()),
        p=np.asarray(weights) / np.sum(weights),
    )
    return values


def anatomy_exposure(series: pd.Series, name: str) -> pd.Series:
    if name in {"F5", "F7", "F8", "A1_Kaan", "A14"}:
        return (series >= 2).astype(int)
    if name.startswith(("A3_", "A4_", "A6_", "A7_", "A8_", "A9_")):
        return series.isin([2, 3]).astype(int)
    if name.startswith("A5_") or name == "A11":
        return series.isin([1, 3]).astype(int)
    if name in {"A12", "A13"}:
        return series.isin([2, 3]).astype(int)
    raise KeyError(f"No exposure rule for {name}")


def draw_items(
    rng: np.random.Generator,
    probabilities: np.ndarray,
    n_items: int,
    lower: int,
) -> np.ndarray:
    return lower + rng.binomial(4, probabilities[:, None], size=(len(probabilities), n_items))


def status_from_ohip(total: pd.Series) -> pd.Series:
    return pd.cut(
        total,
        bins=[-np.inf, 3, 7, 11, 15, np.inf],
        labels=["Kiváló", "Jó", "Átlagos", "Rossz", "Nagyon rossz"],
    ).astype("string")


def status_from_gohai(total: pd.Series) -> pd.Series:
    return pd.cut(
        total,
        bins=[-np.inf, 35, 42, 49, 54, np.inf],
        labels=["Nagyon rossz", "Rossz", "Átlagos", "Jó", "Kiváló"],
    ).astype("string")


def change_category(score: np.ndarray) -> pd.Categorical:
    return pd.cut(
        score,
        bins=[-np.inf, -1.0, -0.25, 0.25, 1.0, np.inf],
        labels=[
            "Sokat romlott",
            "Kicsit romlott",
            "Változatlan maradt",
            "Kicsit javult",
            "Sokat javult",
        ],
    )


def simulate_followup(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "synthetic_subject_id": [f"SYN-{i:04d}" for i in range(1, n + 1)],
            "data_status": "SYNTHETIC_DO_NOT_USE_AS_OBSERVED_DATA",
            "simulation_scenario": "NULL_ANATOMY_EFFECT_PIPELINE_TEST",
            "generation_seed": seed,
            "age_years": np.clip(np.rint(rng.normal(69, 9.5, n)), 45, 90).astype(int),
            "gender": rng.choice(["Female", "Male"], size=n, p=[0.60, 0.40]),
            "denture_type": rng.choice(["lower", "both"], size=n, p=[0.58, 0.42]),
        }
    )

    burden = rng.normal(0, 1, n)
    construct_latents = {
        key: 0.75 * burden + rng.normal(0, 0.65, n)
        for key in ["F", "A1", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A11", "A12", "A13"]
    }

    for name, base in [("F5", -0.45), ("F7", -0.65), ("F8", -0.10)]:
        df[name] = draw_coded_feature(
            rng, construct_latents["F"] + rng.normal(0, 0.55, n), 1, [2, 3], base, unfavorable_weights=[0.72, 0.28]
        )

    df["A1_Kaan"] = draw_coded_feature(
        rng, construct_latents["A1"], 1, [2, 3, 4, 5], -0.20, unfavorable_weights=[0.44, 0.31, 0.17, 0.08]
    )

    for block, base in [("A3", -0.55), ("A4", -1.15), ("A6", -0.45), ("A7", -0.35), ("A8", -0.80), ("A9", -0.50)]:
        for side in ["jobb", "bal"]:
            df[f"{block}_{side}"] = draw_coded_feature(
                rng,
                construct_latents[block] + rng.normal(0, 0.55, n),
                1,
                [2, 3],
                base,
                unfavorable_weights=[0.70, 0.30],
            )

    for side in ["jobb", "bal"]:
        df[f"A5_{side}"] = draw_coded_feature(
            rng,
            construct_latents["A5"] + rng.normal(0, 0.55, n),
            2,
            [1, 3],
            -0.25,
            unfavorable_weights=[0.42, 0.58],
        )

    df["A11"] = draw_coded_feature(
        rng, construct_latents["A11"], 2, [1, 3], -0.95, unfavorable_weights=[0.35, 0.65]
    )
    df["A12"] = draw_coded_feature(
        rng, construct_latents["A12"], 1, [2, 3], -0.25, unfavorable_weights=[0.75, 0.25]
    )
    df["A13"] = draw_coded_feature(
        rng, construct_latents["A13"], 1, [2, 3], -1.05, unfavorable_weights=[0.72, 0.28]
    )
    df["A14"] = rng.choice([1, 2, 3], size=n, p=[0.30, 0.45, 0.25])

    # Outcomes deliberately do not depend on the anatomy variables. This is a
    # null-effect simulation that checks whether the analysis invents effects.
    severity = (
        rng.normal(0, 1, n)
        + 0.018 * (df["age_years"].to_numpy() - 69)
        + 0.12 * (df["denture_type"].to_numpy() == "both")
    )
    recovery = rng.normal(0.45, 0.95, n)

    ohip_offsets = np.array([0.18, -0.10, 0.10, -0.22, -0.05])
    ohip_baseline = np.column_stack(
        [rng.binomial(4, expit(-0.25 + 0.72 * severity + offset)) for offset in ohip_offsets]
    )
    ohip_followup = np.column_stack(
        [rng.binomial(4, expit(-0.55 + 0.72 * severity - 0.62 * recovery + offset)) for offset in ohip_offsets]
    )
    for i in range(5):
        df[f"OHIP_{i + 1}"] = ohip_baseline[:, i]
        df[f"OHIP_{i + 1}_recall"] = ohip_followup[:, i]

    gohai_offsets = np.array([0.00, -0.12, 0.22, 0.12, -0.08, 0.16, -0.10, 0.06, -0.04, 0.08, 0.03, -0.06])
    gohai_baseline = np.column_stack(
        [1 + rng.binomial(4, expit(0.40 - 0.68 * severity + offset)) for offset in gohai_offsets]
    )
    gohai_followup = np.column_stack(
        [1 + rng.binomial(4, expit(0.72 - 0.68 * severity + 0.58 * recovery + offset)) for offset in gohai_offsets]
    )
    for i in range(12):
        df[f"GOHAI_{i + 1}"] = gohai_baseline[:, i]
        df[f"GOHAI_{i + 1}_recall"] = gohai_followup[:, i]

    df["OHIP_baseline_total"] = ohip_baseline.sum(axis=1)
    df["OHIP_followup_total"] = ohip_followup.sum(axis=1)
    df["OHIP_change_improvement"] = df["OHIP_baseline_total"] - df["OHIP_followup_total"]
    df["GOHAI_baseline_total"] = gohai_baseline.sum(axis=1)
    df["GOHAI_followup_total"] = gohai_followup.sum(axis=1)
    df["GOHAI_change_improvement"] = df["GOHAI_followup_total"] - df["GOHAI_baseline_total"]
    df["GOHAI_change_deterioration"] = -df["GOHAI_change_improvement"]
    df["OHIP_change_deterioration"] = -df["OHIP_change_improvement"]

    df["init_mai_huedegree"] = np.clip(rng.normal(49 + 7.5 * severity, 13.5, n), 0, 100).round(3)
    # Higher MAI is worse, so stronger latent recovery should reduce MAI.
    mai_delta = rng.normal(-3.5 - 4.0 * recovery, 10.5, n)
    df["final_mai_huedegree"] = np.clip(df["init_mai_huedegree"] + mai_delta, 0, 100).round(3)
    df["MAI_change_huedegree"] = (df["final_mai_huedegree"] - df["init_mai_huedegree"]).round(3)
    # Higher MAI means worse performance in this project. Therefore a positive
    # improvement score is baseline minus follow-up; deterioration is the
    # reverse. The anchors are generated as noisy independent measurements of
    # the latent recovery construct, not as recodings of the observed scores.
    df["MAI_change_improvement"] = -df["MAI_change_huedegree"]
    df["MAI_change_deterioration"] = df["MAI_change_huedegree"]
    oral_anchor_signal = recovery + rng.normal(0, 0.85, n)
    chewing_anchor_signal = recovery + rng.normal(0, 0.90, n)
    df["responsiveness_today_situation_recall"] = status_from_ohip(df["OHIP_followup_total"])
    df["responsiveness_change"] = change_category(oral_anchor_signal).astype("string")
    df["chewing_today_situation_recall"] = status_from_gohai(df["GOHAI_followup_total"])
    df["chewing_change"] = change_category(chewing_anchor_signal).astype("string")
    df["F9"] = rng.choice([1, 2, 3], size=n, p=[0.69, 0.24, 0.07])
    df["dropout"] = 0

    for name in ANATOMY_COLUMNS:
        df[f"unfavorable_{name}"] = anatomy_exposure(df[name], name)

    return df


def derive_anchor_outcomes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, object]]]:
    """Derive direction-correct cutoffs using anchor ROC and Youden's J."""
    df = df.copy()
    thresholds: dict[str, dict[str, object]] = {}
    for outcome, meta in OUTCOMES.items():
        anchor_positive = df[meta["anchor"]].isin(meta["anchor_labels"]).astype(int)
        score = pd.to_numeric(df[meta["score"]], errors="coerce")
        if anchor_positive.nunique() != 2 or score.nunique() < 2:
            raise ValueError(f"Insufficient anchor or score variation for {outcome}")
        fpr, tpr, candidate_thresholds = roc_curve(
            anchor_positive,
            score,
            drop_intermediate=False,
        )
        # A clinically directional change must be greater than zero in its
        # named direction. Without this constraint, Youden can select a
        # negative cutoff and label small opposite-direction changes as events.
        admissible = np.isfinite(candidate_thresholds) & (candidate_thresholds > 0)
        if not admissible.any():
            raise ValueError(f"No positive direction-consistent ROC threshold for {outcome}")
        youden = np.where(admissible, tpr - fpr, -np.inf)
        best_index = int(np.argmax(youden))
        threshold = float(candidate_thresholds[best_index])
        df[outcome] = (score >= threshold).astype(int)
        thresholds[outcome] = {
            "score_column": meta["score"],
            "anchor_column": meta["anchor"],
            "anchor_positive_labels": meta["anchor_labels"],
            "threshold_youden": threshold,
            "roc_auc": float(roc_auc_score(anchor_positive, score)),
            "sensitivity_at_threshold": float(tpr[best_index]),
            "specificity_at_threshold": float(1.0 - fpr[best_index]),
            "anchor_positive_rate": float(anchor_positive.mean()),
            "derived_event_rate": float(df[outcome].mean()),
        }
    return df, thresholds


def crude_or(exposure: pd.Series, outcome: pd.Series) -> tuple[float, float, float, int, int, int, int]:
    a = int(((exposure == 1) & (outcome == 1)).sum())
    b = int(((exposure == 1) & (outcome == 0)).sum())
    c = int(((exposure == 0) & (outcome == 1)).sum())
    d = int(((exposure == 0) & (outcome == 0)).sum())
    correction = 0.5 if min(a, b, c, d) == 0 else 0.0
    a_c, b_c, c_c, d_c = a + correction, b + correction, c + correction, d + correction
    estimate = (a_c * d_c) / (b_c * c_c)
    se = np.sqrt(1 / a_c + 1 / b_c + 1 / c_c + 1 / d_c)
    return estimate, np.exp(np.log(estimate) - 1.96 * se), np.exp(np.log(estimate) + 1.96 * se), a, b, c, d


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    p = p_values.to_numpy(dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    m = len(p)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty(m, dtype=float)
    result[order] = np.clip(adjusted, 0, 1)
    return pd.Series(result, index=p_values.index)


def fit_or_models(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    age_centered = (df["age_years"] - df["age_years"].mean()) / 10.0
    female = (df["gender"] == "Female").astype(int)
    denture_both = (df["denture_type"] == "both").astype(int)

    for outcome, outcome_meta in OUTCOMES.items():
        baseline = outcome_meta["baseline"]
        baseline_scaled = (df[baseline] - df[baseline].mean()) / df[baseline].std(ddof=0)
        for predictor in ANATOMY_COLUMNS:
            exposure_name = f"unfavorable_{predictor}"
            exposure = df[exposure_name].astype(int)
            y = df[outcome].astype(int)
            crude, crude_low, crude_high, a, b, c, d = crude_or(exposure, y)
            x = pd.DataFrame(
                {
                    exposure_name: exposure,
                    "age_per_10y": age_centered,
                    "female": female,
                    "denture_both": denture_both,
                    "baseline_score_z": baseline_scaled,
                }
            )
            x = sm.add_constant(x, has_constant="add")
            fit = sm.Logit(y, x).fit(disp=False, maxiter=200)
            coefficient = float(fit.params[exposure_name])
            conf = fit.conf_int().loc[exposure_name]
            rows.append(
                {
                    "data_status": "SYNTHETIC_SIMULATION_RESULT_NOT_EMPIRICAL_EVIDENCE",
                    "simulation_scenario": "NULL_ANATOMY_EFFECT_PIPELINE_TEST",
                    "predictor": predictor,
                    "exposure_definition": EXPOSURE_DEFINITIONS[predictor],
                    "outcome": outcome,
                    "outcome_definition": outcome_meta["label"],
                    "n": len(df),
                    "events_total": int(y.sum()),
                    "event_rate": float(y.mean()),
                    "exposed_total": int(exposure.sum()),
                    "exposure_prevalence": float(exposure.mean()),
                    "events_exposed": a,
                    "non_events_exposed": b,
                    "events_unexposed": c,
                    "non_events_unexposed": d,
                    "crude_or": float(crude),
                    "crude_ci_low": float(crude_low),
                    "crude_ci_high": float(crude_high),
                    "adjusted_or": float(np.exp(coefficient)),
                    "adjusted_ci_low": float(np.exp(conf.iloc[0])),
                    "adjusted_ci_high": float(np.exp(conf.iloc[1])),
                    "p_value": float(fit.pvalues[exposure_name]),
                    "adjustment_set": "age_per_10y + gender + denture_type + baseline_outcome",
                }
            )

    results = pd.DataFrame(rows)
    results["q_value_global_132_tests"] = benjamini_hochberg(results["p_value"])
    results["q_value_within_outcome_22_tests"] = results.groupby("outcome", group_keys=False)["p_value"].apply(
        benjamini_hochberg
    )
    results["fdr_significant_global_0_05"] = (results["q_value_global_132_tests"] < 0.05).astype(int)
    return results


def validate(
    df: pd.DataFrame,
    results: pd.DataFrame,
    thresholds: dict[str, dict[str, object]],
    n: int,
) -> dict[str, object]:
    assert len(df) == n
    assert df["synthetic_subject_id"].is_unique
    assert df["data_status"].eq("SYNTHETIC_DO_NOT_USE_AS_OBSERVED_DATA").all()
    assert df[ANATOMY_COLUMNS].notna().all().all()
    assert df[[f"OHIP_{i}" for i in range(1, 6)]].isin(range(5)).all().all()
    assert df[[f"OHIP_{i}_recall" for i in range(1, 6)]].isin(range(5)).all().all()
    assert df[[f"GOHAI_{i}" for i in range(1, 13)]].isin(range(1, 6)).all().all()
    assert df[[f"GOHAI_{i}_recall" for i in range(1, 13)]].isin(range(1, 6)).all().all()
    assert len(results) == len(ANATOMY_COLUMNS) * len(OUTCOMES)
    assert np.isfinite(results[["adjusted_or", "adjusted_ci_low", "adjusted_ci_high", "p_value"]]).all().all()
    for outcome, meta in OUTCOMES.items():
        expected = (df[meta["score"]] >= thresholds[outcome]["threshold_youden"]).astype(int)
        assert expected.equals(df[outcome])
    significant_rows = results.loc[
        results["fdr_significant_global_0_05"] == 1,
        ["predictor", "outcome", "adjusted_or", "adjusted_ci_low", "adjusted_ci_high", "p_value", "q_value_global_132_tests"],
    ].to_dict(orient="records")
    return {
        "row_count": len(df),
        "unique_synthetic_ids": int(df["synthetic_subject_id"].nunique()),
        "or_result_rows": len(results),
        "missing_cells": int(df.isna().sum().sum()),
        "event_rates": {outcome: float(df[outcome].mean()) for outcome in OUTCOMES},
        "global_fdr_significant_count": int(results["fdr_significant_global_0_05"].sum()),
        "global_fdr_significant_results": significant_rows,
        "adjusted_or_min": float(results["adjusted_or"].min()),
        "adjusted_or_max": float(results["adjusted_or"].max()),
    }


def format_or_cell(row: pd.Series) -> str:
    return f"{row['adjusted_or']:.2f} ({row['adjusted_ci_low']:.2f}–{row['adjusted_ci_high']:.2f})"


def build_report(
    results: pd.DataFrame,
    thresholds: dict[str, dict[str, object]],
    audit: dict[str, object],
    seed: int,
) -> str:
    pivot_rows = []
    for predictor in ANATOMY_COLUMNS:
        row: dict[str, str] = {"Képlet": predictor}
        for outcome, short in [
            ("ohip_meaningful_improvement", "OHIP jav."),
            ("ohip_meaningful_deterioration", "OHIP roml."),
            ("gohai_meaningful_improvement", "GOHAI jav."),
            ("gohai_meaningful_deterioration", "GOHAI roml."),
            ("mai_meaningful_improvement", "MAI jav."),
            ("mai_meaningful_deterioration", "MAI roml."),
        ]:
            selected = results[(results["predictor"] == predictor) & (results["outcome"] == outcome)].iloc[0]
            row[short] = format_or_cell(selected)
        pivot_rows.append(row)
    table_lines = [
        "| Képlet | OHIP jav. | OHIP roml. | GOHAI jav. | GOHAI roml. | MAI jav. | MAI roml. |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    table_lines.extend(
        f"| {row['Képlet']} | {row['OHIP jav.']} | {row['OHIP roml.']} | "
        f"{row['GOHAI jav.']} | {row['GOHAI roml.']} | {row['MAI jav.']} | {row['MAI roml.']} |"
        for row in pivot_rows
    )
    table = "\n".join(table_lines)
    if audit["global_fdr_significant_results"]:
        false_positive_lines = "; ".join(
            f"{row['predictor']}–{row['outcome']}: OR {row['adjusted_or']:.2f}, "
            f"95% CI {row['adjusted_ci_low']:.2f}–{row['adjusted_ci_high']:.2f}, "
            f"globális q={row['q_value_global_132_tests']:.3f}"
            for row in audit["global_fdr_significant_results"]
        )
    else:
        false_positive_lines = "nem volt"
    threshold_lines = "\n".join(
        f"- {OUTCOMES[outcome]['label']}: score `{meta['score_column']}`, "
        f"Youden-küszöb `{meta['threshold_youden']:.3f}`, AUC `{meta['roc_auc']:.3f}`, "
        f"szenzitivitás `{meta['sensitivity_at_threshold']:.3f}`, specificitás `{meta['specificity_at_threshold']:.3f}`"
        for outcome, meta in thresholds.items()
    )
    return f"""# SZINTETIKUS szimuláció – anchor-alapú egyedi anatómiai OR-ok

> **Nem empirikus kutatási eredmény.** A 1000 rekord számítógéppel generált,
> valódi beteget nem reprezentál, és nem használható fel hiányzó utánkövetés
> pótlására vagy anatómiai hatás igazolására.

## Szimulációs cél

Az elemzési pipeline ellenőrzése null-hatás forgatókönyvben. Az OHIP-, GOHAI-
és MAI-változás generálása szándékosan nem függött az anatómiai képletektől.
Ezért a várt valódi OR minden képletnél és kimenetnél 1,00.

- N = {audit['row_count']}
- Seed = {seed}
- Anchorok: OHIP és GOHAI = `responsiveness_change`; MAI = `chewing_change`
- Javulás-anchor: `Kicsit javult` vagy `Sokat javult`
- Romlás-anchor: `Kicsit romlott` vagy `Sokat romlott`
- A score-ok iránya úgy lett megfordítva, hogy a nagyobb érték mindig az adott
  irányú nagyobb változást jelentse; OHIP és MAI esetén az alacsonyabb score jobb
- Modell: képletenként külön logisztikus regresszió, korrigálva életkorra,
  nemre, fogsortípusra és a megfelelő kiindulási kimenetre
- Többszörös összehasonlítás: Benjamini–Hochberg FDR, összesen 132 teszt

## Anchor-alapú ROC/Youden küszöbök

{threshold_lines}

## Korrigált OR (95%-os Wald CI)

{table}

## QA

- Eseményarányok: `{json.dumps(audit['event_rates'], ensure_ascii=False)}`
- Globális FDR után szignifikáns tesztek száma: **{audit['global_fdr_significant_count']} / 132**
- A null-szimulációban kapott FDR-pozitív jel(ek): **{false_positive_lines}**
- Becsült korrigált OR-tartomány: **{audit['adjusted_or_min']:.2f}–{audit['adjusted_or_max']:.2f}**
- Hiányzó cellák száma: **{audit['missing_cells']}**

## Értelmezés

Az egyes OR-ok 1 körüli véletlen eltérései Monte Carlo-ingadozások. Mivel az
adatgenerálásban az anatómia bizonyítottan nem hatott a kimenetekre, az esetleg
szignifikánsnak látszó eredmény is ismert álpozitív. Ezekből nem lehet
kijelenteni, hogy bármely anatómiai képlet növeli vagy csökkenti a valódi
betegek számottevő OHIP-, GOHAI- vagy MAI-változásának esélyét. Ehhez valódi,
prospektív utánkövetési adatokon előre rögzített modell szükséges.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--output-dir", type=Path, default=Path("stat_output"))
    args = parser.parse_args()
    if args.n < 100:
        raise SystemExit("Use at least 100 records for this simulation check.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = simulate_followup(args.n, args.seed)
    data, thresholds = derive_anchor_outcomes(data)
    results = fit_or_models(data)
    audit = validate(data, results, thresholds, args.n)

    data_path = args.output_dir / f"SYNTHETIC_ONLY_followup_n{args.n}.csv"
    results_path = args.output_dir / "SYNTHETIC_ONLY_anatomy_or_results.csv"
    summary_path = args.output_dir / "SYNTHETIC_ONLY_simulation_summary.json"
    report_path = Path("SYNTHETIC_ONLY_anatomy_or_report.md")

    data.to_csv(data_path, index=False)
    results.to_csv(results_path, index=False)
    summary = {
        "warning": "SYNTHETIC SIMULATION ONLY - NOT OBSERVED PATIENT DATA OR EMPIRICAL EVIDENCE",
        "scenario": "NULL_ANATOMY_EFFECT_PIPELINE_TEST",
        "seed": args.seed,
        "n": args.n,
        "outcome_definitions": OUTCOMES,
        "anchor_roc_thresholds": thresholds,
        "exposure_definitions": EXPOSURE_DEFINITIONS,
        "model_adjustment": ["age_per_10y", "gender", "denture_type", "baseline_outcome"],
        "multiple_testing": "Benjamini-Hochberg FDR across all 132 tests and within each outcome",
        "audit": audit,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(build_report(results, thresholds, audit, args.seed), encoding="utf-8")

    print(json.dumps({
        "data": str(data_path),
        "results": str(results_path),
        "summary": str(summary_path),
        "report": str(report_path),
        "audit": audit,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
