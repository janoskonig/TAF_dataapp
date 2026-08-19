"""Bayesian cross-sectional analysis for the current PREDICT snapshot.

Question
--------
At the examination visit, are unfavorable anatomical findings associated
with worse patient-reported oral-health quality of life or objective chewing
performance?

This is explicitly cross-sectional. It makes no temporal, prognostic, or
causal claim. The informative-prior analysis is a sensitivity scenario that
forces the elicited anatomical directions onto the cross-sectional outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.integrate import cumulative_trapezoid
from scipy.special import logsumexp
from scipy.stats import norm

from bayes_current_data_exploration import (
    bilateral_any,
    bilateral_mean,
    parse_decimal,
    posterior_from_normal_likelihood,
    zscore,
)


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "stat_output"
OUT.mkdir(exist_ok=True)
TAU = 0.50
SEED = 20260819


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probs: list[float]) -> np.ndarray:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    return np.interp(probs, cdf, values)


def sign_weighted_posterior(mean: float, sd: float, q: float) -> dict[str, float]:
    """Reweight a neutral Normal posterior by a sign-only two-piece prior.

    The magnitude prior remains half-Normal(TAU) on both sides. q changes only
    the prior probability of a positive coefficient.
    """
    if not (np.isfinite(mean) and np.isfinite(sd) and sd > 0 and 0 < q < 1):
        return {key: np.nan for key in ("mean", "lo", "hi", "p_positive")}
    low = min(-4.0, mean - 9 * sd)
    high = max(4.0, mean + 9 * sd)
    grid = np.linspace(low, high, 50001)
    density = norm.pdf(grid, mean, sd)
    density *= np.where(grid >= 0, 2 * q, 2 * (1 - q))
    area = np.trapz(density, grid)
    density /= area
    cdf = cumulative_trapezoid(density, grid, initial=0)
    cdf /= cdf[-1]
    post_mean = np.trapz(grid * density, grid)
    lo, hi = np.interp([0.025, 0.975], cdf, grid)
    p_positive = np.trapz(density[grid >= 0], grid[grid >= 0])
    return {"mean": post_mean, "lo": lo, "hi": hi, "p_positive": p_positive}


def fit_item(
    frame: pd.DataFrame,
    outcome: str,
    predictor: pd.Series,
    name: str,
    q: float | None,
    adjusted: bool,
) -> dict[str, object]:
    work = pd.DataFrame({"y": frame[outcome], "x": predictor})
    controls: list[str] = []
    if adjusted:
        work = work.assign(
            age_z=frame["age_z"], male=frame["male"], single_arch=frame["single_arch"]
        )
        controls = ["age_z", "male", "single_arch"]
    work = work.dropna()
    n = len(work)
    minimum = 12 if adjusted else 8
    base = {"predictor": name, "outcome": outcome, "adjusted": adjusted, "n": n, "elicited_q": q}
    if n < minimum or work["x"].nunique() < 2:
        return base | {"status": "insufficient_data"}
    fit = sm.OLS(
        work["y"], sm.add_constant(work[["x", *controls]], has_constant="add")
    ).fit(cov_type="HC3")
    estimate = float(fit.params["x"])
    se = float(fit.bse["x"])
    neutral = posterior_from_normal_likelihood(estimate, se, TAU)
    if q is None:
        directional = {
            "mean": neutral["mean"],
            "lo": neutral["lo"],
            "hi": neutral["hi"],
            "p_positive": neutral["p_positive"],
        }
    else:
        directional = sign_weighted_posterior(neutral["mean"], neutral["sd"], q)
    return base | {
        "status": "ok",
        "likelihood_estimate": estimate,
        "likelihood_se": se,
        "neutral_mean": neutral["mean"],
        "neutral_lo": neutral["lo"],
        "neutral_hi": neutral["hi"],
        "neutral_p_positive": neutral["p_positive"],
        "directional_mean": directional["mean"],
        "directional_lo": directional["lo"],
        "directional_hi": directional["hi"],
        "directional_p_positive": directional["p_positive"],
        "prior_shift_p_positive": directional["p_positive"] - neutral["p_positive"],
        "prior_data_conflict": bool(q is not None and neutral["p_positive"] < 0.20),
    }


def multivariate_normal_update(
    fit: sm.regression.linear_model.RegressionResultsWrapper,
    coefficient_names: list[str],
    directional_q: dict[str, float],
    rng: np.random.Generator,
    draws: int = 300_000,
) -> tuple[pd.DataFrame, float]:
    """Approximate robust likelihood + independent N(0,TAU²) priors.

    Informative sign-only priors are applied by importance reweighting draws
    from the neutral multivariate posterior.
    """
    bhat = fit.params.loc[coefficient_names].to_numpy(float)
    like_cov = fit.cov_params().loc[coefficient_names, coefficient_names].to_numpy(float)
    like_cov = (like_cov + like_cov.T) / 2
    eig = np.linalg.eigvalsh(like_cov)
    if eig.min() <= 1e-10:
        like_cov += np.eye(len(coefficient_names)) * (1e-8 - eig.min())
    like_precision = np.linalg.pinv(like_cov)
    prior_sd = np.array([1.0 if name == "const" else TAU for name in coefficient_names])
    prior_precision = np.diag(1.0 / prior_sd**2)
    post_cov = np.linalg.pinv(like_precision + prior_precision)
    post_mean = post_cov @ like_precision @ bhat
    post_cov = (post_cov + post_cov.T) / 2
    eig = np.linalg.eigvalsh(post_cov)
    if eig.min() <= 0:
        post_cov += np.eye(len(coefficient_names)) * (1e-10 - eig.min())

    samples = rng.multivariate_normal(post_mean, post_cov, size=draws)
    log_weights = np.zeros(draws)
    for j, name in enumerate(coefficient_names):
        q = directional_q.get(name)
        if q is None:
            continue
        log_weights += np.where(samples[:, j] >= 0, np.log(2 * q), np.log(2 * (1 - q)))
    log_weights -= logsumexp(log_weights)
    weights = np.exp(log_weights)
    ess = float(1.0 / np.sum(weights**2))

    rows = []
    for j, name in enumerate(coefficient_names):
        neutral_sd = float(np.sqrt(post_cov[j, j]))
        neutral_mean = float(post_mean[j])
        neutral_lo, neutral_hi = neutral_mean + norm.ppf([0.025, 0.975]) * neutral_sd
        neutral_p = float(norm.cdf(neutral_mean / neutral_sd))
        directional_mean = float(np.sum(samples[:, j] * weights))
        directional_lo, directional_hi = weighted_quantile(
            samples[:, j], weights, [0.025, 0.975]
        )
        directional_p = float(np.sum(weights[samples[:, j] > 0]))
        rows.append(
            {
                "term": name,
                "neutral_mean": neutral_mean,
                "neutral_lo": neutral_lo,
                "neutral_hi": neutral_hi,
                "neutral_p_positive": neutral_p,
                "directional_mean": directional_mean,
                "directional_lo": float(directional_lo),
                "directional_hi": float(directional_hi),
                "directional_p_positive": directional_p,
                "elicited_q": directional_q.get(name),
            }
        )
    return pd.DataFrame(rows), ess


def fit_multivariable(
    frame: pd.DataFrame,
    outcome: str,
    model_name: str,
    predictors: dict[str, pd.Series],
    directional_q: dict[str, float],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, object]]:
    work = pd.DataFrame({"y": frame[outcome], **predictors})
    work = work.assign(
        age_z=frame["age_z"], male=frame["male"], single_arch=frame["single_arch"]
    ).dropna()
    design_names = [*predictors.keys(), "age_z", "male", "single_arch"]
    design = sm.add_constant(work[design_names], has_constant="add")
    fit = sm.OLS(work["y"], design).fit(cov_type="HC3")
    table, ess = multivariate_normal_update(
        fit, list(design.columns), directional_q, rng
    )
    table.insert(0, "n", len(work))
    table.insert(0, "outcome", outcome)
    table.insert(0, "model", model_name)
    info = {
        "model": model_name,
        "outcome": outcome,
        "n": int(len(work)),
        "parameters_including_intercept": int(len(design.columns)),
        "importance_ess": ess,
        "r_squared_likelihood_fit": float(fit.rsquared),
    }
    return table, info


def main() -> None:
    raw = pd.read_csv(ROOT / "patients.csv")
    frame = raw.drop(
        columns=[
            "id", "TAJ", "paciens_neve", "data_uploader", "init_image_path", "final_image_path"
        ],
        errors="ignore",
    ).copy()

    ohip = frame[[f"OHIP_{i}" for i in range(1, 6)]].sum(axis=1, min_count=5)
    gohai = frame[[f"GOHAI_{i}" for i in range(1, 13)]].sum(axis=1, min_count=12)
    frame["OHIP_worse"] = zscore(ohip)
    frame["GOHAI_worse"] = -zscore(gohai)
    frame["QoL_worse"] = zscore((frame["OHIP_worse"] + frame["GOHAI_worse"]) / 2)
    frame["MAI_worse"] = zscore(frame["init_mai_huedegree"])
    chewing_map = {"Nagyon rossz": 1, "Rossz": 2, "Átlagos": 3, "Jó": 4, "Kiváló": 5}
    frame["chewing_worse"] = -zscore(frame["chewing_today_situation"].map(chewing_map))

    record_date = pd.to_datetime(frame["record_datetime"], errors="coerce")
    birth_date = pd.to_datetime(frame["birthdate"], errors="coerce")
    age = (record_date - birth_date).dt.days / 365.25
    age = age.mask((age < 18) | (age > 105))
    frame["age_z"] = zscore(age)
    frame["male"] = (frame["gender"] == "Male").astype(float)
    frame["single_arch"] = (frame["denture_type"] != "both").astype(float)

    p: dict[str, pd.Series] = {}
    q: dict[str, float | None] = {}
    notes: dict[str, str] = {}

    def add(name: str, values: pd.Series, direction_q: float | None, note: str) -> None:
        p[name] = values.astype(float)
        q[name] = direction_q
        notes[name] = note

    add("F1_low_ridge", -zscore(parse_decimal(frame["F1"])), 0.98, "continuous pilot")
    add("F3_low_palate", -zscore(parse_decimal(frame["F3"])), 0.98, "continuous pilot")
    add("F4_low_angle", -zscore(parse_decimal(frame["F4"])), 0.90, "continuous pilot")
    add("F5_any_flabby", frame["F5"].map({1: 0, 2: 1, 3: 1}), 0.80, "any vs none")
    add("F6_deviation_from_90", zscore((pd.to_numeric(frame["F6"], errors="coerce") - 90).abs()), 0.98, "continuous pilot")
    add("F7_any_torus", frame["F7"].map({1: 0, 2: 1, 3: 1}), 0.98, "any vs none")

    add("A1_ridge_saturated", frame["A1_Kaan"].map({1: 0, 2: 1 / 3, 3: 2 / 3, 4: 1, 5: 1}), 0.98, "primary mandibular ridge measure")
    add("A4_any_torus", bilateral_any(frame, "A4", {2, 3}), 0.98, "any side vs none")
    add("A5_lingual_pouch_risk", bilateral_mean(frame, "A5", {2: 0, 1: 0.5, 3: 1}), 0.98, "bilateral mean")
    add("A6_firm_gingiva_risk", bilateral_mean(frame, "A6", {1: 0, 2: 0.5, 3: 1}), 0.98, "bilateral mean")
    add("A7_tuberculum_shape_risk", bilateral_mean(frame, "A7", {1: 0, 2: 0.5, 3: 1}), 0.98, "bilateral mean")
    add("A8_inclination_risk", bilateral_mean(frame, "A8", {1: 0, 2: 1, 3: 1}), 0.98, "validation-sensitive")
    add("A9_mobility_risk", bilateral_mean(frame, "A9", {1: 0, 2: 0.5, 3: 1}), 0.98, "bilateral mean")
    add("A11_floor_of_mouth_risk", frame["A11"].map({2: 0, 1: 0.5, 3: 1}), 0.98, "independent factor")
    add("A12_spinae_present", frame["A12"].map({1: 0, 2: 1, 3: 1}), 0.98, "ridge indicator sensitivity")
    add("A13_TMJ_risk", frame["A13"].map({1: 0, 2: 0.5, 3: 1}), 0.98, "clinical modifier")

    tub_primary = pd.concat(
        [p["A6_firm_gingiva_risk"], p["A7_tuberculum_shape_risk"], p["A9_mobility_risk"]],
        axis=1,
    ).mean(axis=1, skipna=True)
    add("A6_A7_A9_tuberculum_primary", tub_primary, 0.98, "formative primary; A8 excluded for validation inconsistency")
    tub_with_a8 = pd.concat(
        [
            p["A6_firm_gingiva_risk"],
            p["A7_tuberculum_shape_risk"],
            p["A8_inclination_risk"],
            p["A9_mobility_risk"],
        ],
        axis=1,
    ).mean(axis=1, skipna=True)
    add("A6_A9_tuberculum_with_A8", tub_with_a8, 0.98, "sensitivity composite")

    # Nominal A3 and all asymmetry terms remain direction-free.
    for category in (2, 3):
        right = (frame["A3_jobb"] == category).astype(float).where(frame["A3_jobb"].notna())
        left = (frame["A3_bal"] == category).astype(float).where(frame["A3_bal"].notna())
        add(
            f"A3_category_{category}_proportion",
            pd.concat([right, left], axis=1).mean(axis=1, skipna=True),
            None,
            "exploratory nominal contrast",
        )
    asymmetry = []
    for number in range(3, 10):
        right, left = frame[f"A{number}_jobb"], frame[f"A{number}_bal"]
        both = right.notna() & left.notna()
        asymmetry.append(((right != left) & both).astype(float).where(both))
    add(
        "A3_A9_asymmetry_proportion",
        pd.concat(asymmetry, axis=1).mean(axis=1, skipna=True),
        None,
        "exploratory asymmetry",
    )

    registry = pd.DataFrame(
        [
            {
                "predictor": name,
                "n_nonmissing": int(values.notna().sum()),
                "elicited_q": q[name],
                "note": notes[name],
            }
            for name, values in p.items()
        ]
    )
    registry.to_csv(OUT / "cross_sectional_predictor_registry.csv", index=False)

    item_rows = []
    for outcome in ("QoL_worse", "MAI_worse", "chewing_worse"):
        for name, values in p.items():
            for adjusted in (False, True):
                item_rows.append(fit_item(frame, outcome, values, name, q[name], adjusted))
    item_results = pd.DataFrame(item_rows).merge(registry, on=["predictor", "elicited_q"], how="left")
    item_results.to_csv(OUT / "cross_sectional_item_models.csv", index=False)

    # Parsimonious jaw-specific multivariable models.
    rng = np.random.default_rng(SEED)
    multivariable_tables = []
    model_info = []
    lower_predictors = {
        "ridge": p["A1_ridge_saturated"],
        "tuberculum": p["A6_A7_A9_tuberculum_primary"],
        "floor_mouth": p["A11_floor_of_mouth_risk"],
        "lingual_pouch": p["A5_lingual_pouch_risk"],
        "mandibular_torus": p["A4_any_torus"],
        "TMJ": p["A13_TMJ_risk"],
    }
    lower_q = {name: 0.98 for name in lower_predictors}
    upper_predictors = {
        "flabby": p["F5_any_flabby"],
        "palatal_torus": p["F7_any_torus"],
        "F8_cat2": (frame["F8"] == 2).astype(float).where(frame["F8"].notna()),
        "F8_cat3": (frame["F8"] == 3).astype(float).where(frame["F8"].notna()),
    }
    upper_q = {"flabby": 0.80, "palatal_torus": 0.98}
    for outcome in ("QoL_worse", "MAI_worse", "chewing_worse"):
        for model_name, predictors, qs in (
            ("lower_expanded", lower_predictors, lower_q),
            ("upper_core", upper_predictors, upper_q),
        ):
            table, info = fit_multivariable(frame, outcome, model_name, predictors, qs, rng)
            multivariable_tables.append(table)
            model_info.append(info)
    pd.concat(multivariable_tables, ignore_index=True).to_csv(
        OUT / "cross_sectional_multivariable_models.csv", index=False
    )

    # Prespecified weak interactions in separate, deliberately small models.
    interaction_rows = []
    for interaction_name, x_name, z_name in (
        ("A1_x_A11", "A1_ridge_saturated", "A11_floor_of_mouth_risk"),
        ("A5_x_A11", "A5_lingual_pouch_risk", "A11_floor_of_mouth_risk"),
    ):
        x, z = p[x_name], p[z_name]
        for outcome in ("QoL_worse", "MAI_worse", "chewing_worse"):
            work = pd.DataFrame(
                {
                    "y": frame[outcome],
                    "x": x,
                    "z": z,
                    "age_z": frame["age_z"],
                    "male": frame["male"],
                    "single_arch": frame["single_arch"],
                }
            ).dropna()
            work["x_c"] = work["x"] - work["x"].mean()
            work["z_c"] = work["z"] - work["z"].mean()
            work["interaction"] = work["x_c"] * work["z_c"]
            fit = sm.OLS(
                work["y"],
                sm.add_constant(
                    work[["x_c", "z_c", "interaction", "age_z", "male", "single_arch"]],
                    has_constant="add",
                ),
            ).fit(cov_type="HC3")
            estimate = float(fit.params["interaction"])
            se = float(fit.bse["interaction"])
            neutral = posterior_from_normal_likelihood(estimate, se, TAU)
            directional = sign_weighted_posterior(neutral["mean"], neutral["sd"], 0.80)
            interaction_rows.append(
                {
                    "interaction": interaction_name,
                    "outcome": outcome,
                    "n": len(work),
                    "neutral_mean": neutral["mean"],
                    "neutral_lo": neutral["lo"],
                    "neutral_hi": neutral["hi"],
                    "neutral_p_positive": neutral["p_positive"],
                    "directional_p_positive": directional["p_positive"],
                }
            )
    pd.DataFrame(interaction_rows).to_csv(OUT / "cross_sectional_interactions.csv", index=False)

    # Scale diagnostics and study metadata.
    component_r = float(frame[["OHIP_worse", "GOHAI_worse"]].corr().iloc[0, 1])
    two_item_alpha = float(2 * component_r / (1 + component_r))
    metadata = {
        "design": "cross-sectional",
        "primary_outcome": "QoL_worse = z(mean(z(OHIP), -z(GOHAI)))",
        "secondary_outcomes": ["MAI_worse", "chewing_worse"],
        "qol_component_pearson_r": component_r,
        "qol_two_item_alpha": two_item_alpha,
        "neutral_prior": "Normal(0, 0.5^2)",
        "directional_sensitivity_prior": (
            "two-piece sign-weighted Normal magnitude prior; q weak=.80, moderate=.90, strong=.98"
        ),
        "directional_sensitivity_warning": (
            "Elicited mechanical directions are forced onto cross-sectional outcomes only as sensitivity."
        ),
        "multivariable_models": model_info,
    }
    (OUT / "cross_sectional_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Wrote {OUT / 'cross_sectional_item_models.csv'}")
    print(f"Wrote {OUT / 'cross_sectional_multivariable_models.csv'}")
    print(f"Wrote {OUT / 'cross_sectional_interactions.csv'}")
    print(f"Wrote {OUT / 'cross_sectional_metadata.json'}")


if __name__ == "__main__":
    main()
