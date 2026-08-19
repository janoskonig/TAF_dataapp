"""Preregistered longitudinal analyses for the PREDICT two-jaw cohort."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.miscmodels.ordinal_model import OrderedModel


MIN_MODEL_N = 10
REAL_MODEL_SWITCH_N = 20
SIMULATION_N = 3000
SIMULATION_SEED = 20260819
ANCHOR_ORDER = {
    "Sokat romlott": 1,
    "Kicsit romlott": 2,
    "Változatlan maradt": 3,
    "Kicsit javult": 4,
    "Sokat javult": 5,
}

PREDICTORS = [
    ("f5_risk", "F5 · Lötyögő gerinc", "0 = nincs; 1 = jelen"),
    ("f7_risk", "F7 · Torus palatinus", "0 = nincs; 1 = jelen"),
    ("a1_risk", "A1 · Kaán-gerincforma", "0–1 súlyosság; 1 = legkedvezőtlenebb"),
    ("a3_risk", "A3 · Buccinator tasak", "0 = kétoldalt kedvező; 1 = legalább egy kedvezőtlen oldal"),
    ("a4_risk", "A4 · Torus mandibularis", "0 = nincs; 1 = legalább egy oldalon jelen"),
    ("a5_risk", "A5 · Lingualis tasak", "két oldal 0–1 súlyossági átlaga"),
    ("a6_risk", "A6 · Tuberculum feszes ínyborítása", "két oldal 0–1 súlyossági átlaga"),
    ("a7_risk", "A7 · Tuberculum alakja", "két oldal 0–1 súlyossági átlaga"),
    ("a8_risk", "A8 · Tuberculum–gerinc szög", "két oldal 0–1 kedvezőtlenségi átlaga"),
    ("a9_risk", "A9 · Tuberculum mozgékonysága", "két oldal 0–1 súlyossági átlaga"),
    ("a11_risk", "A11 · Szublingualis tájék", "0–1 súlyosság; 1 = tömött, elődomborodó"),
    ("a12_risk", "A12 · Spinae mentales", "0–1 súlyosság; 1 = nyomásérzékeny"),
    ("a13_risk", "A13 · TMI-funkció", "0–1 súlyosság; 1 = fájdalom/mozgáskorlátozottság"),
]

OUTCOMES = [
    ("ohip_followup", "ohip_baseline", "OHIP-5", "0–20; magasabb = rosszabb"),
    ("gohai_followup", "gohai_baseline", "GOHAI", "12–60; magasabb = jobb"),
    ("mai_followup", "mai_baseline", "MAI hue-degree", "magasabb = rosszabb"),
]

ANCHORS = [
    ("oral_anchor", "Szájüregi egészség változása", "OHIP-5 és GOHAI anchor"),
    ("chewing_anchor", "Rágóképesség változása", "MAI anchor"),
]


def load_longitudinal_dataframe(connection):
    ohip_baseline = [f'p."OHIP_{i}" AS ohip_{i}' for i in range(1, 6)]
    gohai_baseline = [f'p."GOHAI_{i}" AS gohai_{i}' for i in range(1, 13)]
    ohip_followup = [
        f'COALESCE(f.ohip_{i}_recall, p."OHIP_{i}_recall") AS ohip_{i}_recall'
        for i in range(1, 6)
    ]
    gohai_followup = [
        f'COALESCE(f.gohai_{i}_recall, p."GOHAI_{i}_recall") AS gohai_{i}_recall'
        for i in range(1, 13)
    ]
    selected = ",\n".join(
        [
            'p."id" AS patient_id',
            'p."birthdate" AS birthdate',
            'p."gender" AS gender',
            'p."F5" AS f5',
            'p."F7" AS f7',
            'p."A1_Kaan" AS a1_kaan',
            'p."A3_jobb" AS a3_jobb',
            'p."A3_bal" AS a3_bal',
            'p."A4_jobb" AS a4_jobb',
            'p."A4_bal" AS a4_bal',
            'p."A5_jobb" AS a5_jobb',
            'p."A5_bal" AS a5_bal',
            *[f'p."A{i}_{side}" AS a{i}_{side}' for i in range(6, 10) for side in ("jobb", "bal")],
            'p."A11" AS a11',
            'p."A12" AS a12',
            'p."A13" AS a13',
            *ohip_baseline,
            *gohai_baseline,
            'p."init_mai_huedegree" AS mai_baseline',
            *ohip_followup,
            *gohai_followup,
            'COALESCE(f.final_mai_huedegree, p."final_mai_huedegree") AS mai_followup',
            'COALESCE(f.responsiveness_change, p."responsiveness_change") AS oral_anchor_text',
            'COALESCE(f.chewing_change, p."chewing_change") AS chewing_anchor_text',
            'f.completed_at AS new_completed_at',
        ]
    )
    query = f"""
        WITH latest AS (
            SELECT DISTINCT ON ("TAJ") *
            FROM patients
            WHERE "TAJ" IS NOT NULL
            ORDER BY "TAJ", "id" DESC
        )
        SELECT {selected}
        FROM latest p
        LEFT JOIN followup_visits f
          ON f.patient_id = p."id" AND f.visit_round = 1
        WHERE LOWER(TRIM(p."denture_type")) = 'both'
          AND p."OHIP_1" IS NOT NULL
        ORDER BY p."id"
    """
    with connection.cursor() as cursor:
        cursor.execute(query)
        columns = [description[0] for description in cursor.description]
        records = cursor.fetchall()
    frame = pd.DataFrame(records, columns=columns)
    return prepare_analysis_frame(frame)


def prepare_analysis_frame(frame):
    df = frame.copy()
    numeric_columns = [
        column
        for column in df.columns
        if column.startswith(("a", "f5", "f7", "ohip_", "gohai_", "mai_"))
        and column not in {"oral_anchor_text", "chewing_anchor_text"}
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["ohip_baseline"] = _complete_sum(df, [f"ohip_{i}" for i in range(1, 6)])
    df["ohip_followup"] = _complete_sum(df, [f"ohip_{i}_recall" for i in range(1, 6)])
    df["gohai_baseline"] = _complete_sum(df, [f"gohai_{i}" for i in range(1, 13)])
    df["gohai_followup"] = _complete_sum(df, [f"gohai_{i}_recall" for i in range(1, 13)])
    df["oral_anchor"] = df["oral_anchor_text"].map(ANCHOR_ORDER)
    df["chewing_anchor"] = df["chewing_anchor_text"].map(ANCHOR_ORDER)

    df["f5_risk"] = df["f5"].map({1: 0.0, 2: 1.0, 3: 1.0})
    df["f7_risk"] = df["f7"].map({1: 0.0, 2: 1.0, 3: 1.0})
    df["a1_risk"] = df["a1_kaan"].map({1: 0.0, 2: 1 / 3, 3: 2 / 3, 4: 1.0, 5: 1.0})
    df["a3_risk"] = df.apply(
        lambda row: _any_adverse(row, ["a3_jobb", "a3_bal"], lambda value: value in {1, 3}), axis=1
    )
    df["a4_risk"] = df.apply(
        lambda row: _any_adverse(row, ["a4_jobb", "a4_bal"], lambda value: value in {2, 3}), axis=1
    )
    df["a5_risk"] = df.apply(
        lambda row: _bilateral_mean_score(row, "a5", {1: 0.0, 2: 0.5, 3: 1.0}), axis=1
    )
    df["a6_risk"] = df.apply(
        lambda row: _bilateral_mean_score(row, "a6", {1: 0.0, 2: 0.5, 3: 1.0}), axis=1
    )
    df["a7_risk"] = df.apply(
        lambda row: _bilateral_mean_score(row, "a7", {1: 0.0, 2: 0.5, 3: 1.0}), axis=1
    )
    df["a8_risk"] = df.apply(
        lambda row: _bilateral_mean_score(row, "a8", {1: 0.0, 2: 1.0, 3: 1.0}), axis=1
    )
    df["a9_risk"] = df.apply(
        lambda row: _bilateral_mean_score(row, "a9", {1: 0.0, 2: 0.5, 3: 1.0}), axis=1
    )
    df["a11_risk"] = df["a11"].map({1: 0.0, 2: 0.5, 3: 1.0})
    df["a12_risk"] = df["a12"].map({1: 0.0, 2: 0.5, 3: 1.0})
    df["a13_risk"] = df["a13"].map({1: 0.0, 2: 0.5, 3: 1.0})

    df["ridge_atrophy"] = df.apply(_ridge_atrophy, axis=1)
    df["torus_mandibularis"] = df.apply(
        lambda row: _any_adverse(row, ["a4_jobb", "a4_bal"], lambda value: value in {2, 3}), axis=1
    )
    df["lingual_pouch"] = df.apply(
        lambda row: _any_adverse(row, ["a5_jobb", "a5_bal"], lambda value: value == 3), axis=1
    )
    df["tuberculum_score"] = df.apply(_tuberculum_score, axis=1)
    df["tuberculum_adverse"] = np.where(
        df["tuberculum_score"].notna(),
        (df["tuberculum_score"] >= 0.5).astype(float),
        np.nan,
    )
    df["mouth_floor"] = np.where(df["a11"].notna(), (df["a11"] == 3).astype(float), np.nan)
    components = ["ridge_atrophy", "torus_mandibularis", "lingual_pouch", "tuberculum_adverse", "mouth_floor"]
    df["anatomical_burden_0_5"] = df[components].sum(axis=1, min_count=len(components))
    return df


def build_longitudinal_report(connection):
    df = load_longitudinal_dataframe(connection)
    report = {
        "cohort": cohort_summary(df),
        "descriptive": descriptive_summary(df),
        "anchor_distributions": anchor_distributions(df),
        "continuous_models": [],
        "ordinal_models": [],
        "minimum_model_n": MIN_MODEL_N,
        "real_model_switch_n": REAL_MODEL_SWITCH_N,
    }

    for outcome, baseline, outcome_label, scale_note in OUTCOMES:
        for predictor, predictor_label, predictor_scale in PREDICTORS:
            report["continuous_models"].append(
                fit_continuous_model(
                    df,
                    outcome=outcome,
                    baseline=baseline,
                    outcome_label=outcome_label,
                    scale_note=scale_note,
                    predictor=predictor,
                    predictor_label=predictor_label,
                    predictor_scale=predictor_scale,
                )
            )

    for anchor, anchor_label, anchor_role in ANCHORS:
        for predictor, predictor_label, predictor_scale in PREDICTORS:
            report["ordinal_models"].append(
                fit_ordinal_model(
                    df,
                    anchor=anchor,
                    anchor_label=anchor_label,
                    anchor_role=anchor_role,
                    predictor=predictor,
                    predictor_label=predictor_label,
                    predictor_scale=predictor_scale,
                )
            )

    _add_fdr(report["continuous_models"], group_key="outcome")
    _add_fdr(report["ordinal_models"], group_key="anchor")

    report["real_continuous_models"] = report["continuous_models"]
    report["real_ordinal_models"] = report["ordinal_models"]
    real_models_ready = all(
        all(
            row.get("status") == "ok" and row.get("n", 0) >= REAL_MODEL_SWITCH_N
            for row in report["continuous_models"]
            if row["outcome"] == outcome
        )
        for outcome, _, _, _ in OUTCOMES
    )
    if real_models_ready:
        report.update(model_source="real", model_n=None, simulation=None)
    else:
        simulated = simulate_followup_frame(df, n=SIMULATION_N, seed=SIMULATION_SEED)
        simulated_continuous = []
        simulated_ordinal = []
        for outcome, baseline, outcome_label, scale_note in OUTCOMES:
            for predictor, predictor_label, predictor_scale in PREDICTORS:
                simulated_continuous.append(
                    fit_continuous_model(
                        simulated,
                        outcome=outcome,
                        baseline=baseline,
                        outcome_label=outcome_label,
                        scale_note=scale_note,
                        predictor=predictor,
                        predictor_label=predictor_label,
                        predictor_scale=predictor_scale,
                    )
                )
        for anchor, anchor_label, anchor_role in ANCHORS:
            for predictor, predictor_label, predictor_scale in PREDICTORS:
                simulated_ordinal.append(
                    fit_ordinal_model(
                        simulated,
                        anchor=anchor,
                        anchor_label=anchor_label,
                        anchor_role=anchor_role,
                        predictor=predictor,
                        predictor_label=predictor_label,
                        predictor_scale=predictor_scale,
                    )
                )
        _add_fdr(simulated_continuous, group_key="outcome")
        _add_fdr(simulated_ordinal, group_key="anchor")
        for row in simulated_continuous + simulated_ordinal:
            if row.get("status") == "ok":
                row["caution"] = "Szimulált demonstráció; nem valódi betegadatból származó kutatási eredmény."
        report.update(
            model_source="simulated",
            model_n=SIMULATION_N,
            continuous_models=simulated_continuous,
            ordinal_models=simulated_ordinal,
            simulation={
                "n": SIMULATION_N,
                "seed": SIMULATION_SEED,
                "source_pool_n": int(len(df[[key for key, _, _ in PREDICTORS] + ["ohip_baseline", "gohai_baseline", "mai_baseline"]].dropna())),
                "assumptions": [
                    "A kiindulási pontszámok és a 13 külön anatómiai képlet kódolt értékei a teljes meglévő kétállcsontos esetek visszatevéses mintavételéből származnak.",
                    "A generáló egyenletben egy képlet teljes 0→1 kedvezőtlenségi változásának feltételezett hatása: OHIP +0,45; GOHAI −0,75; MAI hue-degree +2,6 pont, véletlen zaj mellett.",
                    "Az anchor-generálás feltételezett log-odds hatása képletenkénti teljes 0→1 kedvezőtlenségre: szájüregi egészség −0,18; rágóképesség −0,15.",
                    "A szimuláció rögzített maggal reprodukálható, kizárólag memóriában fut, és nem kerül egyik adatbázistáblába sem.",
                ],
            },
        )
    report["charts"] = build_visualizations(df, report)
    return report


def simulate_followup_frame(real_df, n=SIMULATION_N, seed=SIMULATION_SEED):
    """Create a deterministic in-memory demonstration cohort without DB writes."""
    rng = np.random.default_rng(seed)
    columns = [key for key, _, _ in PREDICTORS] + ["ohip_baseline", "gohai_baseline", "mai_baseline"]
    pool = real_df[columns].dropna().reset_index(drop=True)
    if len(pool) < 5:
        raise ValueError("Legalább öt teljes kiindulási eset szükséges a demonstrációs szimulációhoz.")
    sampled = pool.iloc[rng.integers(0, len(pool), size=n)].reset_index(drop=True).copy()
    risk_columns = [key for key, _, _ in PREDICTORS]
    risk_sum = sampled[risk_columns].sum(axis=1).astype(float)

    sampled["ohip_followup"] = np.clip(
        sampled["ohip_baseline"] - 4.0 + 0.45 * risk_sum + rng.normal(0, 2.7, n),
        0,
        20,
    )
    sampled["gohai_followup"] = np.clip(
        sampled["gohai_baseline"] + 6.0 - 0.75 * risk_sum + rng.normal(0, 4.2, n),
        12,
        60,
    )
    sampled["mai_followup"] = np.clip(
        sampled["mai_baseline"] - 18.0 + 2.6 * risk_sum + rng.normal(0, 9.0, n),
        0,
        None,
    )
    oral_latent = 4.2 - 0.18 * risk_sum + rng.logistic(0, 1, n)
    chewing_latent = 4.0 - 0.15 * risk_sum + rng.logistic(0, 1, n)
    sampled["oral_anchor"] = np.digitize(oral_latent, [1.5, 2.5, 3.5, 4.5]) + 1
    sampled["chewing_anchor"] = np.digitize(chewing_latent, [1.5, 2.5, 3.5, 4.5]) + 1
    return sampled


def build_visualizations(real_df, report):
    """Build aggregate, identifier-free interactive charts for the results page."""
    charts = {
        "coverage": _coverage_chart(report["cohort"]),
        "anatomy": _anatomy_distribution_chart(real_df),
        "paired": _paired_outcomes_chart(real_df),
        "beta_forest": _beta_forest_chart(report["continuous_models"], report["model_source"]),
        "or_forest": _or_forest_chart(report["ordinal_models"], report["model_source"]),
    }
    return charts


def _coverage_chart(cohort):
    labels = [
        "Kétállcsontos kohorsz",
        "Bármilyen utánkövetés",
        "Teljes OHIP–GOHAI + anchor",
        "Teljes MAI + anchor",
    ]
    values = [
        cohort["eligible"],
        cohort["any_followup"],
        cohort["questionnaire_pairs"],
        cohort["mai_pairs"],
    ]
    denominator = max(cohort["eligible"], 1)
    text_values = [f"{value} ({100 * value / denominator:.0f}%)" for value in values]
    fig = go.Figure(
        go.Bar(
            x=values[::-1],
            y=labels[::-1],
            orientation="h",
            text=text_values[::-1],
            textposition="outside",
            cliponaxis=False,
            marker=dict(color=["#d77a36", "#5d8fa3", "#37a29d", "#0d5f62"]),
            hovertemplate="%{y}: %{x} beteg<extra></extra>",
        )
    )
    fig.update_layout(
        title="A valódi utánkövetési adatok lefedettsége",
        xaxis_title="Betegek száma",
        yaxis_title=None,
        xaxis=dict(range=[0, max(values) * 1.25 if max(values) else 1], rangemode="tozero"),
        height=330,
        margin=dict(l=20, r=55, t=55, b=45),
        showlegend=False,
    )
    return _chart_html(fig, include_plotlyjs="cdn")


def _anatomy_distribution_chart(real_df):
    labels = [label for _, label, _ in PREDICTORS]
    means = [float(real_df[key].mean() * 100) for key, _, _ in PREDICTORS]
    counts = [int(real_df[key].notna().sum()) for key, _, _ in PREDICTORS]
    fig = go.Figure(
        go.Bar(
            x=means[::-1],
            y=labels[::-1],
            orientation="h",
            text=[f"{value:.0f}% · n={count}" for value, count in zip(means, counts)][::-1],
            textposition="outside",
            marker_color="#086f70",
            hovertemplate="%{y}<br>Átlagos kedvezőtlenség: %{x:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title="Valódi anatómiai képletek 0–1 kedvezőtlenségi szintje",
        xaxis_title="Átlagos kedvezőtlenségi score (%)",
        yaxis_title=None,
        xaxis=dict(range=[0, 115], ticksuffix="%"),
        height=560,
        margin=dict(l=205, r=70, t=55, b=50),
        showlegend=False,
    )
    return _chart_html(fig)


def _paired_outcomes_chart(real_df):
    panels = [
        ("ohip_baseline", "ohip_followup", "OHIP-5", "0–20; ↑ rosszabb"),
        ("gohai_baseline", "gohai_followup", "GOHAI", "12–60; ↑ jobb"),
        ("mai_baseline", "mai_followup", "MAI hue-degree", "↑ rosszabb"),
    ]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{label}<br><sup>{note}</sup>" for _, _, label, note in panels])
    for column, (baseline, followup, _, _) in enumerate(panels, 1):
        paired = real_df[[baseline, followup]].dropna()
        for _, row in paired.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=["Kiindulás", "Utánkövetés"],
                    y=[row[baseline], row[followup]],
                    mode="lines+markers",
                    line=dict(color="rgba(96,117,130,.35)", width=1.5),
                    marker=dict(color="#607582", size=6),
                    showlegend=False,
                    hovertemplate="%{x}: %{y:.2f}<extra></extra>",
                ),
                row=1,
                col=column,
            )
        if not paired.empty:
            fig.add_trace(
                go.Scatter(
                    x=["Kiindulás", "Utánkövetés"],
                    y=[paired[baseline].mean(), paired[followup].mean()],
                    mode="lines+markers",
                    line=dict(color="#d77a36", width=4),
                    marker=dict(color="#d77a36", size=10, symbol="diamond"),
                    name="Átlag",
                    showlegend=column == 1,
                    hovertemplate="Átlag · %{x}: %{y:.2f}<extra></extra>",
                ),
                row=1,
                col=column,
            )
    fig.update_layout(
        title="Valódi párosított kiindulási és utánkövetési pontszámok",
        height=410,
        margin=dict(l=45, r=20, t=80, b=45),
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
    )
    return _chart_html(fig)


def _beta_forest_chart(rows, source):
    outcome_order = [("ohip_followup", "OHIP-5"), ("gohai_followup", "GOHAI"), ("mai_followup", "MAI hue-degree")]
    fig = make_subplots(rows=3, cols=1, subplot_titles=[label for _, label in outcome_order], vertical_spacing=0.09)
    for panel, (outcome, _) in enumerate(outcome_order, 1):
        selected = [row for row in rows if row["outcome"] == outcome and row.get("status") == "ok"]
        selected = list(reversed(selected))
        if not selected:
            continue
        estimates = [row["beta"] for row in selected]
        fig.add_trace(
            go.Scatter(
                x=estimates,
                y=[row["predictor_label"] for row in selected],
                mode="markers",
                marker=dict(
                    size=[12 if row["predictor"] == "anatomical_burden_0_5" else 9 for row in selected],
                    color=["#d77a36" if row["predictor"] == "anatomical_burden_0_5" else "#086f70" for row in selected],
                    symbol=["diamond" if row["predictor"] == "anatomical_burden_0_5" else "circle" for row in selected],
                ),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[row["ci_high"] - row["beta"] for row in selected],
                    arrayminus=[row["beta"] - row["ci_low"] for row in selected],
                    color="#607582",
                    thickness=1.5,
                ),
                text=[row["predictor_scale"] for row in selected],
                hovertemplate="%{y}<br>β=%{x:.2f}<br>%{text}<extra></extra>",
                showlegend=False,
            ),
            row=panel,
            col=1,
        )
        fig.add_vline(x=0, line_dash="dash", line_color="#8396a0", row=panel, col=1)
        fig.update_xaxes(title_text="β és 95%-os CI", row=panel, col=1)
    source_label = "SZIMULÁLT DEMONSTRÁCIÓ" if source == "simulated" else "VALÓDI ADAT"
    fig.update_layout(
        title=f"{source_label} · Korrigált folytonos regressziós hatások",
        height=1320,
        margin=dict(l=285, r=35, t=75, b=45),
    )
    return _chart_html(fig)


def _or_forest_chart(rows, source):
    anchor_order = [("oral_anchor", "Szájüregi egészség anchor"), ("chewing_anchor", "Rágóképesség anchor")]
    fig = make_subplots(rows=2, cols=1, subplot_titles=[label for _, label in anchor_order], vertical_spacing=0.13)
    for panel, (anchor, _) in enumerate(anchor_order, 1):
        selected = [row for row in rows if row["anchor"] == anchor and row.get("status") == "ok"]
        selected = list(reversed(selected))
        if not selected:
            continue
        estimates = [row["odds_ratio"] for row in selected]
        fig.add_trace(
            go.Scatter(
                x=estimates,
                y=[row["predictor_label"] for row in selected],
                mode="markers",
                marker=dict(
                    size=[12 if row["predictor"] == "anatomical_burden_0_5" else 9 for row in selected],
                    color=["#d77a36" if row["predictor"] == "anatomical_burden_0_5" else "#5d8fa3" for row in selected],
                    symbol=["diamond" if row["predictor"] == "anatomical_burden_0_5" else "circle" for row in selected],
                ),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=[row["ci_high"] - row["odds_ratio"] for row in selected],
                    arrayminus=[row["odds_ratio"] - row["ci_low"] for row in selected],
                    color="#607582",
                    thickness=1.5,
                ),
                text=[row["predictor_scale"] for row in selected],
                hovertemplate="%{y}<br>OR=%{x:.2f}<br>%{text}<extra></extra>",
                showlegend=False,
            ),
            row=panel,
            col=1,
        )
        fig.add_vline(x=1, line_dash="dash", line_color="#8396a0", row=panel, col=1)
        fig.update_xaxes(title_text="OR és 95%-os CI (log skála)", type="log", row=panel, col=1)
    source_label = "SZIMULÁLT DEMONSTRÁCIÓ" if source == "simulated" else "VALÓDI ADAT"
    fig.update_layout(
        title=f"{source_label} · Másodlagos ordinális anchor-OR-ok",
        height=940,
        margin=dict(l=285, r=35, t=75, b=45),
    )
    return _chart_html(fig)


def _chart_html(fig, include_plotlyjs=False):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, ui-sans-serif, system-ui, sans-serif", color="#132c3b", size=12),
        hoverlabel=dict(font_size=12),
    )
    return fig.to_html(
        full_html=False,
        include_plotlyjs=include_plotlyjs,
        config={"displayModeBar": False, "responsive": True},
    )


def cohort_summary(df):
    has_questionnaire = df[["ohip_followup", "gohai_followup", "oral_anchor"]].notna().all(axis=1)
    has_mai = df[["mai_baseline", "mai_followup", "chewing_anchor"]].notna().all(axis=1)
    any_followup = df[["ohip_followup", "gohai_followup", "mai_followup", "oral_anchor", "chewing_anchor"]].notna().any(axis=1)
    return {
        "eligible": int(len(df)),
        "any_followup": int(any_followup.sum()),
        "questionnaire_pairs": int(has_questionnaire.sum()),
        "mai_pairs": int(has_mai.sum()),
        "complete_burden": int(df["anatomical_burden_0_5"].notna().sum()),
    }


def descriptive_summary(df):
    rows = []
    for outcome, baseline, label, scale_note in OUTCOMES:
        paired = df[[baseline, outcome]].dropna()
        row = {
            "outcome": outcome,
            "label": label,
            "scale_note": scale_note,
            "baseline_n": int(df[baseline].notna().sum()),
            "followup_n": int(df[outcome].notna().sum()),
            "paired_n": int(len(paired)),
        }
        for prefix, series in (("baseline", paired[baseline]), ("followup", paired[outcome])):
            row.update(_series_summary(series, prefix))
        row.update(_series_summary(paired[outcome] - paired[baseline], "change"))
        rows.append(row)
    return rows


def anchor_distributions(df):
    rows = []
    labels = list(ANCHOR_ORDER)
    for column, label, role in ANCHORS:
        counts = df[column].value_counts().reindex(range(1, 6), fill_value=0)
        rows.append(
            {
                "anchor": column,
                "label": label,
                "role": role,
                "n": int(df[column].notna().sum()),
                "counts": [{"label": text, "count": int(counts[index])} for index, text in enumerate(labels, 1)],
            }
        )
    return rows


def fit_continuous_model(
    df,
    *,
    outcome,
    baseline,
    outcome_label,
    scale_note,
    predictor,
    predictor_label,
    predictor_scale,
):
    model_df = df[[outcome, baseline, predictor]].dropna()
    result = _model_row(
        kind="continuous",
        outcome=outcome,
        outcome_label=outcome_label,
        predictor=predictor,
        predictor_label=predictor_label,
        predictor_scale=predictor_scale,
        n=len(model_df),
    )
    result["scale_note"] = scale_note
    result["adjustment"] = f"{baseline}"
    issue = _model_eligibility_issue(model_df, predictor)
    if issue:
        result.update(status="insufficient", message=issue)
        return result
    try:
        design = sm.add_constant(model_df[[baseline, predictor]].astype(float), has_constant="add")
        fitted = sm.OLS(model_df[outcome].astype(float), design).fit(cov_type="HC3")
        beta = float(fitted.params[predictor])
        ci_low, ci_high = [float(value) for value in fitted.conf_int().loc[predictor]]
        result.update(
            status="ok",
            beta=beta,
            ci_low=ci_low,
            ci_high=ci_high,
            p_value=float(fitted.pvalues[predictor]),
            r_squared=float(fitted.rsquared),
            caution=_sample_caution(len(model_df)),
        )
    except Exception as exc:
        result.update(status="unstable", message=f"A modell nem volt stabilan illeszthető: {type(exc).__name__}.")
    return result


def fit_ordinal_model(
    df,
    *,
    anchor,
    anchor_label,
    anchor_role,
    predictor,
    predictor_label,
    predictor_scale,
):
    model_df = df[[anchor, predictor]].dropna()
    result = _model_row(
        kind="ordinal",
        outcome=anchor,
        outcome_label=anchor_label,
        predictor=predictor,
        predictor_label=predictor_label,
        predictor_scale=predictor_scale,
        n=len(model_df),
    )
    result.update(anchor=anchor, anchor_role=anchor_role, categories=int(model_df[anchor].nunique()))
    issue = _model_eligibility_issue(model_df, predictor, ordinal_outcome=anchor)
    if issue:
        result.update(status="insufficient", message=issue)
        return result
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            model = OrderedModel(
                model_df[anchor].astype(int),
                model_df[[predictor]].astype(float),
                distr="logit",
            )
            fitted = model.fit(method="bfgs", disp=False, maxiter=500)
        coefficient = float(fitted.params[predictor])
        standard_error = float(fitted.bse[predictor])
        if not all(math.isfinite(value) for value in (coefficient, standard_error)) or standard_error > 10:
            raise ValueError("non-finite or extreme uncertainty")
        ci_low = coefficient - 1.96 * standard_error
        ci_high = coefficient + 1.96 * standard_error
        result.update(
            status="ok",
            odds_ratio=float(math.exp(coefficient)),
            ci_low=float(math.exp(ci_low)),
            ci_high=float(math.exp(ci_high)),
            p_value=float(fitted.pvalues[predictor]),
            caution=_sample_caution(len(model_df)),
            po_diagnostic=_threshold_slope_diagnostic(model_df, anchor, predictor),
        )
    except Exception as exc:
        result.update(status="unstable", message=f"Az ordinális modell nem volt stabilan illeszthető: {type(exc).__name__}.")
    return result


def _model_eligibility_issue(model_df, predictor, ordinal_outcome=None):
    if len(model_df) < MIN_MODEL_N:
        return f"Csak {len(model_df)} teljes eset van; legalább {MIN_MODEL_N} szükséges már a technikai modellillesztéshez is."
    if model_df[predictor].nunique() < 2:
        return "A prediktornak nincs elegendő variabilitása."
    if ordinal_outcome and model_df[ordinal_outcome].nunique() < 2:
        return "Az anchor válaszai nem tartalmaznak legalább két kategóriát."
    counts = model_df[predictor].value_counts()
    if set(model_df[predictor].dropna().unique()).issubset({0, 1}) and counts.min() < 3:
        return "A bináris anatómiai konstrukció egyik csoportjában háromnál kevesebb beteg van."
    return None


def _threshold_slope_diagnostic(model_df, outcome, predictor):
    slopes = []
    for threshold in sorted(model_df[outcome].unique())[:-1]:
        binary = (model_df[outcome] > threshold).astype(int)
        if binary.sum() < 3 or (1 - binary).sum() < 3:
            continue
        try:
            fit = sm.Logit(binary, sm.add_constant(model_df[[predictor]], has_constant="add")).fit(disp=False)
            slope = float(fit.params[predictor])
            if math.isfinite(slope):
                slopes.append(slope)
        except Exception:
            continue
    if len(slopes) < 2:
        return "A proporcionális odds feltétel küszöbspecifikus ellenőrzéséhez kevés az adat."
    same_direction = all(value >= 0 for value in slopes) or all(value <= 0 for value in slopes)
    spread = max(slopes) - min(slopes)
    direction_text = "azonos" if same_direction else "eltérő"
    return f"Érzékenységi ellenőrzés: {len(slopes)} küszöbmodell, {direction_text} előjel; meredekségtartomány {min(slopes):.2f}–{max(slopes):.2f} (terjedelem {spread:.2f})."


def _add_fdr(rows, group_key):
    groups = {}
    for row in rows:
        if row.get("status") == "ok" and row["predictor"] != "anatomical_burden_0_5":
            groups.setdefault(row[group_key], []).append(row)
    for group in groups.values():
        p_values = np.array([row["p_value"] for row in group], dtype=float)
        order = np.argsort(p_values)
        adjusted = np.empty(len(group), dtype=float)
        running = 1.0
        for reverse_rank in range(len(group) - 1, -1, -1):
            original_index = order[reverse_rank]
            rank = reverse_rank + 1
            running = min(running, p_values[original_index] * len(group) / rank)
            adjusted[original_index] = min(running, 1.0)
        for row, q_value in zip(group, adjusted):
            row["q_value"] = float(q_value)


def _model_row(**values):
    return {**values, "status": "pending"}


def _sample_caution(n):
    if n < 20:
        return "Nagyon kis mintás, hipotézisgeneráló becslés; az intervallum és a modell stabilitása elsődleges."
    if n < 25:
        return "Leíró, hipotézisgeneráló pilot becslés."
    if n < 30:
        return "Értékelhető, de továbbra is bizonytalan pilot becslés."
    return "A jelenlegi kohorszmérethez közeli pilot becslés."


def _complete_sum(df, columns):
    return df[columns].sum(axis=1, min_count=len(columns))


def _series_summary(series, prefix):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_median": None,
            f"{prefix}_q1": None,
            f"{prefix}_q3": None,
        }
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(values.median()),
        f"{prefix}_q1": float(values.quantile(0.25)),
        f"{prefix}_q3": float(values.quantile(0.75)),
    }


def _any_adverse(row, columns, adverse):
    values = [row[column] for column in columns]
    observed = [value for value in values if pd.notna(value)]
    if any(adverse(value) for value in observed):
        return 1.0
    if len(observed) == len(columns):
        return 0.0
    return np.nan


def _bilateral_mean_score(row, base, mapping):
    values = []
    for side in ("jobb", "bal"):
        value = row[f"{base}_{side}"]
        if pd.notna(value) and value in mapping:
            values.append(mapping[value])
    return float(np.mean(values)) if values else np.nan


def _ridge_atrophy(row):
    values = [row["a1_kaan"], row["a12"]]
    if (pd.notna(values[0]) and values[0] >= 3) or (pd.notna(values[1]) and values[1] >= 2):
        return 1.0
    if all(pd.notna(value) for value in values):
        return 0.0
    return np.nan


def _tuberculum_score(row):
    recoded = []
    for number in range(6, 10):
        for side in ("jobb", "bal"):
            value = row[f"a{number}_{side}"]
            if pd.isna(value):
                return np.nan
            if number == 8:
                mapping = {1: 0.0, 2: 1.0, 3: 1.0}
            else:
                mapping = {1: 0.0, 2: 0.5, 3: 1.0}
            if value not in mapping:
                return np.nan
            recoded.append(mapping[value])
    return float(np.mean(recoded))
