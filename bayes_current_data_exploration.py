"""PREDICT current-data Bayesian exploration.

The script is deliberately de-identified and uses only aggregate outputs.
It does not fit the elicited directional priors to OHIP/GOHAI/MAI because the
expert elicitation has not yet assigned those priors to these outcomes.

Primary exploratory model:
    standardized worse outcome ~ one prespecified risk predictor

The coefficient likelihood is approximated by the HC3-robust OLS estimate and
standard error. It is updated with a skeptical/neutral Normal(0, 0.5^2) prior.
This is a small-sample screening analysis, not the final PREDICT model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, rankdata


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "patients.csv"
OUT = ROOT / "stat_output"
OUT.mkdir(exist_ok=True)

NEUTRAL_TAU = 0.50

IDENTIFIER_COLUMNS = {
    "id",
    "TAJ",
    "paciens_neve",
    "data_uploader",
    "init_image_path",
    "final_image_path",
}


def parse_decimal(series: pd.Series) -> pd.Series:
    """Parse plain numbers and Mongo-style {'$decimal': '...'} strings."""
    text = series.astype("string")
    extracted = text.str.extract(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
        expand=False,
    )
    return pd.to_numeric(extracted, errors="coerce")


def zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    sd = values.std(ddof=1)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    return (values - values.mean()) / sd


def rank_normal(series: pd.Series) -> pd.Series:
    """Rank-based inverse-normal transform, retaining missingness."""
    values = pd.to_numeric(series, errors="coerce")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    observed = values.notna()
    n = int(observed.sum())
    if n < 2:
        return result
    ranks = rankdata(values.loc[observed], method="average")
    result.loc[observed] = norm.ppf((ranks - 0.5) / n)
    return zscore(result)


def bilateral_mean(
    frame: pd.DataFrame, variable: str, mapping: dict[int, float]
) -> pd.Series:
    right = frame[f"{variable}_jobb"].map(mapping)
    left = frame[f"{variable}_bal"].map(mapping)
    return pd.concat([right, left], axis=1).mean(axis=1, skipna=True)


def bilateral_any(
    frame: pd.DataFrame, variable: str, present_values: set[int]
) -> pd.Series:
    right_raw = frame[f"{variable}_jobb"]
    left_raw = frame[f"{variable}_bal"]
    right = right_raw.map(lambda x: float(x in present_values) if pd.notna(x) else np.nan)
    left = left_raw.map(lambda x: float(x in present_values) if pd.notna(x) else np.nan)
    return pd.concat([right, left], axis=1).max(axis=1, skipna=True)


def posterior_from_normal_likelihood(
    estimate: float, se: float, tau: float = NEUTRAL_TAU
) -> dict[str, float]:
    """Normal likelihood × Normal(0,tau²) skeptical prior."""
    if not (np.isfinite(estimate) and np.isfinite(se) and se > 0):
        return {key: np.nan for key in ("mean", "sd", "lo", "hi", "p_positive", "p_rope")}
    variance = 1.0 / (1.0 / se**2 + 1.0 / tau**2)
    mean = variance * estimate / se**2
    sd = np.sqrt(variance)
    lo, hi = mean + norm.ppf([0.025, 0.975]) * sd
    p_positive = norm.cdf(mean / sd)
    rope = 0.10
    p_rope = norm.cdf((rope - mean) / sd) - norm.cdf((-rope - mean) / sd)
    return {
        "mean": mean,
        "sd": sd,
        "lo": lo,
        "hi": hi,
        "p_positive": p_positive,
        "p_rope": p_rope,
    }


def fit_one(
    frame: pd.DataFrame,
    outcome: str,
    predictor: pd.Series,
    predictor_name: str,
    scale: str,
    adjusted: bool,
) -> dict[str, float | str | int]:
    work = pd.DataFrame({"y": frame[outcome], "x": predictor})
    controls: list[str] = []
    if adjusted:
        work = work.assign(
            age_z=frame["age_z"],
            male=frame["male"],
            single_arch=frame["single_arch"],
        )
        controls = ["age_z", "male", "single_arch"]
    work = work.dropna()
    n = len(work)
    minimum_n = 12 if adjusted else 8
    base = {
        "predictor": predictor_name,
        "outcome": outcome,
        "outcome_scale": scale,
        "adjusted": adjusted,
        "n": n,
    }
    if n < minimum_n or work["x"].nunique() < 2:
        return base | {"status": "insufficient_data"}

    design = sm.add_constant(work[["x", *controls]], has_constant="add")
    try:
        fit = sm.OLS(work["y"], design).fit(cov_type="HC3")
        estimate = float(fit.params["x"])
        se = float(fit.bse["x"])
    except (ValueError, np.linalg.LinAlgError):
        return base | {"status": "fit_failed"}

    posterior = posterior_from_normal_likelihood(estimate, se)
    return base | {
        "status": "ok",
        "likelihood_estimate": estimate,
        "likelihood_se": se,
        "likelihood_lo": estimate - 1.96 * se,
        "likelihood_hi": estimate + 1.96 * se,
        "neutral_post_mean": posterior["mean"],
        "neutral_post_sd": posterior["sd"],
        "neutral_post_lo": posterior["lo"],
        "neutral_post_hi": posterior["hi"],
        "neutral_post_p_expected_direction": posterior["p_positive"],
        "neutral_post_p_rope_abs_lt_0_1": posterior["p_rope"],
    }


def main() -> None:
    raw = pd.read_csv(DATA)
    # Defensive guarantee: no identifier is ever included in an output table.
    frame = raw.drop(columns=[c for c in IDENTIFIER_COLUMNS if c in raw], errors="ignore").copy()

    # Outcomes: higher is consistently worse.
    ohip_columns = [f"OHIP_{i}" for i in range(1, 6)]
    gohai_columns = [f"GOHAI_{i}" for i in range(1, 13)]
    frame["OHIP_sum"] = frame[ohip_columns].sum(axis=1, min_count=len(ohip_columns))
    frame["GOHAI_sum"] = frame[gohai_columns].sum(axis=1, min_count=len(gohai_columns))
    frame["OHIP_worse"] = zscore(frame["OHIP_sum"])
    frame["GOHAI_worse"] = -zscore(frame["GOHAI_sum"])
    frame["MAI_worse"] = zscore(frame["init_mai_huedegree"])
    for outcome in ("OHIP_worse", "GOHAI_worse", "MAI_worse"):
        frame[f"{outcome}_rank"] = rank_normal(frame[outcome])

    record_date = pd.to_datetime(frame["record_datetime"], errors="coerce")
    birth_date = pd.to_datetime(frame["birthdate"], errors="coerce")
    age = (record_date - birth_date).dt.days / 365.25
    age = age.mask((age < 18) | (age > 105))
    frame["age_z"] = zscore(age)
    frame["male"] = (frame["gender"] == "Male").astype(float)
    frame["single_arch"] = (frame["denture_type"] != "both").astype(float)

    predictors: dict[str, pd.Series] = {}
    meta: dict[str, dict[str, object]] = {}

    def add(name: str, values: pd.Series, certainty: str, q: float | None, note: str) -> None:
        predictors[name] = values.astype(float)
        meta[name] = {"certainty": certainty, "elicited_q": q, "note": note}

    add("F1_low_ridge", -zscore(parse_decimal(frame["F1"])), "strong", 0.98, "per 1 SD lower ridge")
    add("F3_low_palate", -zscore(parse_decimal(frame["F3"])), "strong", 0.98, "per 1 SD lower vault")
    add("F4_low_angle", -zscore(parse_decimal(frame["F4"])), "moderate", 0.90, "per 1 SD lower angle")
    add("F5_any_flabby", frame["F5"].map({1: 0, 2: 1, 3: 1}), "weak", 0.80, "any vs none")
    add("F6_deviation_from_90", zscore((pd.to_numeric(frame["F6"], errors="coerce") - 90).abs()), "strong", 0.98, "per 1 SD absolute deviation")
    add("F7_any_torus", frame["F7"].map({1: 0, 2: 1, 3: 1}), "strong", 0.98, "any vs none")

    add("A1_ridge_saturated", frame["A1_Kaan"].map({1: 0, 2: 1 / 3, 3: 2 / 3, 4: 1, 5: 1}), "strong", 0.98, "best-to-worst saturated contrast")

    # A3 is nominal and exploratory: proportions of sides in categories 2/3.
    for category in (2, 3):
        right = (frame["A3_jobb"] == category).astype(float).where(frame["A3_jobb"].notna())
        left = (frame["A3_bal"] == category).astype(float).where(frame["A3_bal"].notna())
        add(
            f"A3_category_{category}_proportion",
            pd.concat([right, left], axis=1).mean(axis=1, skipna=True),
            "exploratory",
            None,
            f"proportion of sides in category {category}; category 1 reference",
        )

    add("A4_any_torus", bilateral_any(frame, "A4", {2, 3}), "strong", 0.98, "any side vs none")
    add("A5_lingual_pouch_risk", bilateral_mean(frame, "A5", {2: 0, 1: 0.5, 3: 1}), "strong", 0.98, "bilateral mean, best-to-worst")
    add("A6_firm_gingiva_risk", bilateral_mean(frame, "A6", {1: 0, 2: 0.5, 3: 1}), "strong", 0.98, "bilateral mean, best-to-worst")
    add("A7_tuberculum_shape_risk", bilateral_mean(frame, "A7", {1: 0, 2: 0.5, 3: 1}), "strong", 0.98, "bilateral mean, best-to-worst")
    add("A8_inclination_risk", bilateral_mean(frame, "A8", {1: 0, 2: 1, 3: 1}), "strong", 0.98, "bilateral mean; 2 and validation category 3 grouped")
    add("A9_mobility_risk", bilateral_mean(frame, "A9", {1: 0, 2: 0.5, 3: 1}), "strong", 0.98, "bilateral mean, best-to-worst")
    add("A11_floor_of_mouth_risk", frame["A11"].map({2: 0, 1: 0.5, 3: 1}), "strong", 0.98, "best-to-worst")
    add("A12_spinae_present", frame["A12"].map({1: 0, 2: 1, 3: 1}), "strong", 0.98, "palpable/painful vs not palpable")
    add("A13_TMJ_risk", frame["A13"].map({1: 0, 2: 0.5, 3: 1}), "strong", 0.98, "additive clinical factor")

    tuberculum_items = [
        predictors["A6_firm_gingiva_risk"],
        predictors["A7_tuberculum_shape_risk"],
        predictors["A8_inclination_risk"],
        predictors["A9_mobility_risk"],
    ]
    add(
        "A6_A9_tuberculum_composite",
        pd.concat(tuberculum_items, axis=1).mean(axis=1, skipna=True),
        "construct_summary",
        None,
        "simple formative pilot summary; not a validated latent score",
    )

    asymmetry = []
    for number in range(3, 10):
        right = frame[f"A{number}_jobb"]
        left = frame[f"A{number}_bal"]
        both = right.notna() & left.notna()
        asymmetry.append(((right != left) & both).astype(float).where(both))
    add(
        "A3_A9_asymmetry_proportion",
        pd.concat(asymmetry, axis=1).mean(axis=1, skipna=True),
        "exploratory",
        None,
        "proportion of bilateral items with unequal sides",
    )

    predictor_registry = pd.DataFrame(
        [
            {
                "predictor": name,
                "n_nonmissing": int(values.notna().sum()),
                "n_unique": int(values.nunique(dropna=True)),
                **meta[name],
            }
            for name, values in predictors.items()
        ]
    )
    predictor_registry.to_csv(OUT / "bayes_predictor_registry.csv", index=False)

    results: list[dict[str, object]] = []
    for predictor_name, predictor in predictors.items():
        for outcome in ("MAI_worse", "OHIP_worse", "GOHAI_worse"):
            for scale, outcome_column in (
                ("raw_z", outcome),
                ("rank_normal", f"{outcome}_rank"),
            ):
                for adjusted in (False, True):
                    results.append(
                        fit_one(
                            frame,
                            outcome_column,
                            predictor,
                            predictor_name,
                            scale,
                            adjusted,
                        )
                    )
    result_frame = pd.DataFrame(results)
    result_frame = result_frame.merge(predictor_registry, on="predictor", how="left")
    result_frame.to_csv(OUT / "bayes_neutral_item_models.csv", index=False)

    # Follow-up completeness and exact default-pattern audit.
    follow_ohip = [f"OHIP_{i}_recall" for i in range(1, 6)]
    follow_gohai = [f"GOHAI_{i}_recall" for i in range(1, 13)]
    complete_ohip = frame[follow_ohip].notna().all(axis=1)
    complete_gohai = frame[follow_gohai].notna().all(axis=1)
    default_ohip = (frame[follow_ohip] == 4).all(axis=1) & complete_ohip
    default_gohai = (frame[follow_gohai] == 1).all(axis=1) & complete_gohai

    # A7=3 should imply A8=3 according to the expert elicitation.
    validation = {}
    for side in ("jobb", "bal"):
        a7 = frame[f"A7_{side}"]
        a8 = frame[f"A8_{side}"]
        complete = a7.notna() & a8.notna()
        validation[side] = {
            "complete": int(complete.sum()),
            "A7_eq_3": int(((a7 == 3) & complete).sum()),
            "A7_eq_3_and_A8_ne_3": int(((a7 == 3) & (a8 != 3) & complete).sum()),
        }

    audit = {
        "rows": int(len(frame)),
        "denture_type": frame["denture_type"].value_counts(dropna=False).to_dict(),
        "baseline_complete": {
            "OHIP": int(frame[ohip_columns].notna().all(axis=1).sum()),
            "GOHAI": int(frame[gohai_columns].notna().all(axis=1).sum()),
            "MAI": int(frame["init_mai_huedegree"].notna().sum()),
        },
        "followup_complete": {
            "OHIP": int(complete_ohip.sum()),
            "GOHAI": int(complete_gohai.sum()),
            "final_MAI": int(frame["final_mai_huedegree"].notna().sum()),
        },
        "followup_exact_default_worst": {
            "OHIP": int(default_ohip.sum()),
            "GOHAI": int(default_gohai.sum()),
            "both": int((default_ohip & default_gohai).sum()),
        },
        "A7_A8_validation": validation,
        "neutral_prior": {"distribution": "Normal(0, 0.5^2)", "rope": [-0.1, 0.1]},
        "directional_priors_applied": False,
        "directional_priors_not_applied_reason": (
            "Expert directions have not yet been assigned to MAI/OHIP/GOHAI parameters."
        ),
    }
    (OUT / "bayes_current_data_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Construct diagnostics: these are descriptive, not scale validation.
    construct = pd.concat(tuberculum_items, axis=1)
    construct.columns = ["A6", "A7", "A8", "A9"]
    construct_corr = construct.corr(method="spearman")
    construct_corr.to_csv(OUT / "bayes_tuberculum_spearman.csv")
    complete_construct = construct.dropna()
    k = complete_construct.shape[1]
    item_variance = complete_construct.var(axis=0, ddof=1).sum()
    total_variance = complete_construct.sum(axis=1).var(ddof=1)
    alpha = k / (k - 1) * (1 - item_variance / total_variance)
    (OUT / "bayes_construct_diagnostics.json").write_text(
        json.dumps(
            {
                "tuberculum_complete_n": int(len(complete_construct)),
                "cronbach_alpha_descriptive_only": float(alpha),
                "warning": "A6-A9 may form a formative rather than reflective construct.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {OUT / 'bayes_predictor_registry.csv'}")
    print(f"Wrote {OUT / 'bayes_neutral_item_models.csv'}")
    print(f"Wrote {OUT / 'bayes_current_data_audit.json'}")


if __name__ == "__main__":
    main()
