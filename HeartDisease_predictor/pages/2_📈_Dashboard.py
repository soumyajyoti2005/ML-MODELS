"""pages/2_📈_Dashboard.py — Interactive Clinical Data Explorer & Insights.

Plotly charts themed in the PulseBloom cream / pink / violet palette, with
sidebar filters (sex, age band, chest pain type) that update instantly.
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plotly.graph_objects as go  # noqa: E402
import utils  # noqa: E402

st.set_page_config(page_title="PulseBloom · Dashboard", page_icon="📈", layout="wide")
_THEME = utils.apply_theme()  # day/night claymorphism CSS (theme_fig auto-detects)


def kpi_strip(df):
    """Top KPI summary strip."""
    k1, k2, k3, k4 = st.columns(4, gap="large")
    total = len(df)
    prev = df["HeartDisease"].mean() * 100
    avg_age = df["Age"].mean()
    mean_chol = df["Cholesterol"].mean()
    kpis = [
        (f"{total:,}", "Total Cohort Size", "patients"),
        (f"{prev:.1f}%", "High Risk Prevalence", "of cohort"),
        (f"{avg_age:.1f}", "Average Patient Age", "years"),
        (f"{mean_chol:.1f}", "Mean Serum Cholesterol", "mg/dL"),
    ]
    for col, (num, label, hint) in zip((k1, k2, k3, k4), kpis):
        with col:
            st.markdown(
                f'<div class="kpi-card"><span class="kpi-num">{num}</span>'
                f'<span class="kpi-label">{label} · {hint}</span></div>',
                unsafe_allow_html=True,
            )


def sidebar_filters(df):
    """Filter controls; returns the filtered dataframe."""
    with st.sidebar:
        st.markdown('<div class="card-title">🔎 Filters</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="card-sub">Slice the cohort without reloading — the charts '
            "re-render instantly.</p>",
            unsafe_allow_html=True,
        )
        sex = st.multiselect("Gender", options=["Male", "Female"], default=["Male", "Female"])
        age_bands = st.multiselect(
            "Age bracket",
            options=[str(b) for b in df["AgeBand"].cat.categories],
            default=[str(b) for b in df["AgeBand"].cat.categories],
        )
        chest = st.multiselect(
            "Chest pain classification",
            options=["ATA", "NAP", "ASY", "TA"],
            default=["ATA", "NAP", "ASY", "TA"],
        )

        mask = (
            df["SexLabel"].isin(sex)
            & df["AgeBand"].astype(str).isin(age_bands)
            & df["ChestPainType"].isin(chest)
        )
        filtered = df[mask]
        st.markdown(
            f'<div class="pill-badge">📊 {len(filtered):,} '
            f"/ {len(df):,} rows shown</div>",
            unsafe_allow_html=True,
        )
        return filtered


def section(title, sub):
    st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="nb-section">{title}</div>', unsafe_allow_html=True)
def chart_1_age_sex(df):
    """Distribution of Heart Disease across Age & Sex (violin)."""
    fig = go.Figure()
    colors = {"No Heart Disease": utils.MINT, "Heart Disease": utils.PINK}
    for risk in ["No Heart Disease", "Heart Disease"]:
        for sex in ["Female", "Male"]:
            subset = df[(df["Risk"] == risk) & (df["SexLabel"] == sex)]["Age"]
            if subset.empty:
                continue
            fig.add_trace(
                go.Violin(
                    y=subset,
                    x=[f"{sex} · {risk}"],
                    name=f"{sex} · {risk}",
                    side="positive" if sex == "Male" else "negative",
                    line_color=colors[risk],
                    fillcolor=utils.PINK_SOFT if sex == "Female" else utils.VIOLET_SOFT,
                    box_visible=True,
                    meanline_visible=True,
                    opacity=0.75,
                )
            )
    fig.update_layout(title="Heart Disease Distribution across Age & Sex", height=460, yaxis_title="Age (years)")
    utils.theme_fig(fig)
    st.plotly_chart(fig, use_container_width=True)


def chart_2_chest_risk(df):
    """Chest Pain Type vs risk likelihood (bar)."""
    order = ["ATA", "NAP", "ASY", "TA"]
    rates, counts = [], []
    for cpt in order:
        sub = df[df["ChestPainType"] == cpt]
        if len(sub) == 0:
            rates.append(0)
            counts.append(0)
            continue
        rates.append(sub["HeartDisease"].mean() * 100)
        counts.append(len(sub))
    fig = go.Figure(
        go.Bar(
            x=order,
            y=rates,
            customdata=counts,
            marker_color=[utils.MINT, utils.VIOLET, utils.PINK, utils.CORAL_DEEP],
            text=[f"{r:.0f}%" for r in rates],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Risk likelihood: %{y:.1f}%<br>Cohort: %{customdata} patients<extra></extra>",
        )
    )
    fig.update_layout(
        title="Chest Pain Type vs Risk Likelihood",
        height=430,
        yaxis_title="Heart Disease rate (%)",
        xaxis_title="Chest Pain Type",
        showlegend=False,
    )
    utils.theme_fig(fig)
    st.plotly_chart(fig, use_container_width=True)


def chart_3_hr_stpeak(df):
    """MaxHR vs Oldpeak scatter with gentle risk clustering."""
    fig = go.Figure()
    for risk, color in [("No Heart Disease", utils.VIOLET_DEEP), ("Heart Disease", utils.PINK)]:
        sub = df[df["Risk"] == risk]
        fig.add_trace(
            go.Scatter(
                x=sub["MaxHR"],
                y=sub["Oldpeak"],
                mode="markers",
                name=risk,
                marker=dict(
                    size=7,
                    color=color,
                    opacity=0.55,
                    line=dict(color="white", width=0.6),
                ),
                hovertemplate="MaxHR %{x} bpm · Oldpeak %{y}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Peak Heart Rate vs ST Depression (risk clustering)",
        height=440,
        xaxis_title="Maximum Heart Rate (bpm)",
        yaxis_title="Oldpeak — ST Depression",
        legend=dict(orientation="h", y=1.06, x=0.2),
    )
    utils.theme_fig(fig)
    st.plotly_chart(fig, use_container_width=True)


def chart_4_cholesterol(df):
    """Cholesterol density curves across risk classes."""
    fig = go.Figure()
    for risk, color in [("No Heart Disease", utils.VIOLET), ("Heart Disease", utils.PINK)]:
        sub = df[df["Risk"] == risk]["Cholesterol"]
        if sub.empty:
            continue
        fig.add_trace(
            go.Histogram(
                x=sub,
                name=risk,
                nbinsx=28,
                marker_color=color,
                opacity=0.55,
                hovertemplate="Cholesterol %{x} mg/dL<br>Count %{y}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Cholesterol Levels: Healthy vs High-Risk Cases",
        height=440,
        xaxis_title="Serum Cholesterol (mg/dL)",
        yaxis_title="Patient Count",
        barmode="overlay",
        legend=dict(orientation="h", y=1.06, x=0.25),
    )
    utils.theme_fig(fig)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page body
# ---------------------------------------------------------------------------
utils.side_brand()
utils.clay_topbar()

st.markdown(
    '<span class="hero-eyebrow">📈&nbsp;CLINICAL DATA EXPLORER</span>',
    unsafe_allow_html=True,
)
st.markdown(
    '<h1 class="hero-title" style="font-size:2.3rem">The numbers behind '
    '<span class="hl">the model</span></h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-sub">Rendered from heart.csv — the same 918 clinical '
    "records used to train the classifier in main.ipynb.</p>",
    unsafe_allow_html=True,
)

df = utils.load_dataset()
filtered = sidebar_filters(df)
kpi_strip(filtered)

section("Age & Sex Profile", "Older cohorts and male patients skew toward higher prevalence.")
chart_1_age_sex(filtered)

section("Symptom Profile", "Asymptomatic and typical-angina profiles carry the highest likelihood.")
chart_2_chest_risk(filtered)

c3, c4 = st.columns(2, gap="large")
with c3:
    chart_3_hr_stpeak(filtered)
with c4:
    chart_4_cholesterol(filtered)

m = utils.model_metrics()
if m:
    acc, f1, n, name = m
    note = f"{name} · Accuracy {acc*100:.1f}% · F1 {f1*100:.1f}% · {n:,} records"
else:
    note = "Run `python train_model.py` to reload model metrics."
st.markdown(f'<p class="nb-section-sub" style="margin-top:1rem">Model: {note}</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)
utils.safe_page_link("pages/1_💓_Predict.py", label="Back to the Risk Assessment →", icon="💓", key="foot_predict")
st.markdown(
    '<div class="nb-footer">PulseBloom · CardiaCare v1.0 · all numbers derive from heart.csv · '
    '<span class="pill-badge">DATA ROOM</span></div>',
    unsafe_allow_html=True,
)