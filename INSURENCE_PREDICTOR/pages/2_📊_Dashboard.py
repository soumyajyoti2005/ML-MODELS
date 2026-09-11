"""pages/2_📊_Dashboard.py — Suncover dashboard ("Data Room").

Interactive Plotly charts themed to the neo-brutalist look: cream background,
pure-black axis lines, flat fills, thin black frame around each chart.
"""

import base64
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plotly.graph_objects as go  # noqa: E402
import utils  # noqa: E402

ILI = os.path.join(utils.ROOT, "assets", "illustrations")
CSS = os.path.join(utils.ROOT, "assets", "style.css")

st.set_page_config(page_title="Suncover · Dashboard", page_icon="📊", layout="wide")
utils.load_css(CSS)


def svg_uri(name):
    """Base64 data-URI for an SVG in assets/illustrations."""
    with open(os.path.join(ILI, name), "rb") as f:
        return "data:image/svg+xml;base64," + base64.b64encode(f.read()).decode("ascii")


@st.cache_data
def load_data():
    """Deduped insurance.csv (same rows as the model) with derived columns."""
    df = pd.read_csv(os.path.join(utils.ROOT, "insurance.csv"))
    df = df.drop_duplicates().reset_index(drop=True)
    df["is_female"] = df["sex"].map({"male": 0, "female": 1})
    df["is_smoker"] = df["smoker"].map({"no": 0, "yes": 1})
    for r in utils.REGION_COLS:
        df[r] = (df["region"] == r).astype(int)
    df["bmi_category"] = pd.cut(df["bmi"], bins=utils.BMI_BINS, labels=utils.BMI_LABELS)
    return df


def group_header(title, sub, doodle=None):
    """Bold black divider + section title, with a hand-drawn SVG corner accent."""
    st.markdown('<div class="nb-group">', unsafe_allow_html=True)
    st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="nb-section">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="nb-section-sub">{sub}</p>', unsafe_allow_html=True)
    if doodle:
        st.markdown(
            f'<img class="nb-doodle" src="{svg_uri(doodle)}" alt=""/>',
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<div class="nb-eyebrow">SUNCOVER // DATA ROOM</div>', unsafe_allow_html=True)
st.markdown(
    '<h1 class="hero-title" style="font-size:2.15rem">The numbers behind '
    '<span class="hl">the model</span></h1>',
    unsafe_allow_html=True,
)
st.markdown(
    "<p class=\"hero-sub\">RENDERED FROM insurance.csv — THE SAME ROWS THE MODEL WAS TRAINED ON.</p>",
    unsafe_allow_html=True,
)

df = load_data()
m = utils.model_metrics()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4, gap="medium")
kp = [
    (f"{len(df):,}", "TOTAL RECORDS"),
    (f"${df['charges'].mean():,.0f}", "AVERAGE CHARGE"),
    (f"{df['age'].mean():.1f}", "AVERAGE AGE"),
    (f"{df['is_smoker'].mean() * 100:.1f}%", "SMOKERS"),
]
for col, (num, lab) in zip((k1, k2, k3, k4), kp):
    with col:
        st.markdown(
            f'<div class="nb-kpi"><span class="nb-kpi-num">{num}</span>'
            f'<span class="nb-kpi-label">{lab}</span></div>',
            unsafe_allow_html=True,
        )

if m:
    st.markdown(
        f'<p class="nb-section-sub" style="padding-top:6px">MODEL <span class="nb-stamp" '
        f'style="transform:rotate(-4deg)">R² = {m[0]:.3f} · ADJ R² = {m[1]:.3f}</span></p>',
        unsafe_allow_html=True,
    )
# ---------------------------------------------------------------------------
# Group 1 — spread of charges
# ---------------------------------------------------------------------------
group_header(
    "VALUE SPREAD",
    "WHERE DO THE CHARGES LIVE? THE RIGHT TAIL IS LONG — A FEW VERY EXPENSIVE CLAIMS.",
    "accent_star.svg",
)

c1, c2 = st.columns(2, gap="large")

with c1:
    mean_charge = df["charges"].mean()
    fig = go.Figure(
        go.Histogram(
            x=df["charges"],
            nbinsx=30,
            marker=dict(color=utils.YELLOW, line=dict(color=utils.INK, width=2)),
        )
    )
    fig.add_vline(x=mean_charge, line_dash="dash", line_color=utils.CORAL, line_width=2)
    fig.update_layout(title="CHARGES · HISTOGRAM", height=360, showlegend=False)
    utils.theme_fig(fig)
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = go.Figure()
    for lbl, color in (("no", utils.YELLOW), ("yes", utils.CORAL)):
        sub = df.loc[df["smoker"] == lbl, "charges"]
        fig.add_trace(
            go.Box(
                y=sub,
                name="SMOKER" if lbl == "yes" else "NON-SMOKER",
                marker=dict(color=color, line=dict(color=utils.INK, width=2)),
                line=dict(color=utils.INK, width=2),
                boxpoints=False,
            )
        )
    fig.update_layout(title="CHARGES BY SMOKER · BOX", height=360)
    utils.theme_fig(fig)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Group 2 — age vs charges
# ---------------------------------------------------------------------------
group_header(
    "AGE & SMOKING",
    "SMOKING LIFTS THE ENTIRE DISTRIBUTION — SEE IT IN THE CORAL CLOUD.",
    "accent_arrow.svg",
)

fig = go.Figure()
for sm, name, color in (
    ("no", "non-smoker", utils.YELLOW),
    ("yes", "smoker", utils.CORAL),
):
    sub = df[df["smoker"] == sm]
    fig.add_trace(
        go.Scatter(
            x=sub["age"], y=sub["charges"], mode="markers", name=name,
            marker=dict(color=color, size=7, line=dict(color=utils.INK, width=1)),
        )
    )
fig.update_layout(title="AGE vs CHARGES · COLOURED BY SMOKER", height=420)
utils.theme_fig(fig)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Group 3 — linear relationships & composition
# ---------------------------------------------------------------------------
group_header(
    "MAKE-UP & CORRELATION",
    "SMOKING CORRELATES HARDEST WITH CHARGES; REGION AND SEX ARE NEARLY DEAF.",
    "accent_squiggle.svg",
)

c3, c4 = st.columns(2, gap="large")

with c3:
    numeric_cols = ["age", "bmi", "children", "charges", "is_female", "is_smoker"] + utils.REGION_COLS
    corr = df[numeric_cols].corr().round(2)
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            text=corr.values,
            texttemplate="%{text}",
            textfont=dict(family="IBM Plex Mono", color=utils.INK, size=10),
            colorscale=[[0, "#FFFDF6"], [0.5, utils.YELLOW], [1, utils.CORAL]],
            zmin=-1,
            zmax=1,
            showscale=False,
            hovertemplate="%{x} / %{y}: %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(title="CORRELATION HEATMAP", height=470, margin=dict(t=50, l=20, r=20, b=20))
    utils.theme_fig(fig)
    st.plotly_chart(fig, use_container_width=True)

with c4:
    region_counts = df["region"].value_counts()
    fig = go.Figure(
        go.Pie(
            labels=region_counts.index,
            values=region_counts.values,
            hole=0.55,
            sort=False,
            textinfo="percent",
            textfont=dict(family="IBM Plex Mono", color=utils.INK, size=12),
            marker=dict(
                colors=["#FFD54F", "#FFC107", "#FFE49A", "#FFB300"],
                line=dict(color=utils.INK, width=2),
            ),
        )
    )
    fig.update_layout(
        title="REGION SPLIT",
        height=470,
        showlegend=False,
        annotations=[
            dict(
                text=f"{len(df):,} rows",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(family="IBM Plex Mono", color=utils.INK, size=13),
            )
        ],
    )
    utils.theme_fig(fig)
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Group 4 — BMI make-up
# ---------------------------------------------------------------------------
group_header(
    "BMI MAKE-UP",
    "MOSTLY HEALTHY — BUT THE MODEL KEEPS A SPECIAL EYE ON THE OBESE BUCKET.",
    "accent_arrow.svg",
)

vals = [int((df["bmi_category"] == lab).sum()) for lab in utils.BMI_LABELS]
fig = go.Figure(
    go.Bar(
        x=list(utils.BMI_LABELS),
        y=vals,
        marker=dict(color=utils.YELLOW, line=dict(color=utils.INK, width=2)),
        text=vals,
        texttemplate="%{text}",
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", color=utils.INK, size=12),
    )
)
fig.update_layout(title="BMI CATEGORY BREAKDOWN", height=380, showlegend=False)
utils.theme_fig(fig)
st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)
st.page_link("pages/1_🔮_Predict.py", label="Back to the Predictor →", icon="🔮")
st.markdown(
    '<div class="nb-footer">SUNCOVER v1.0 · all numbers derive from insurance.csv · '
    '<span class="nb-stamp">DATA ROOM</span></div>',
    unsafe_allow_html=True,
)