"""pages/1_🔮_Predict.py — Suncover prediction page.

Form inputs mirror the RAW insurance.csv columns (age, sex, bmi, children,
smoker, region). All derived features (is_female / is_smoker / region dummies /
bmi_category_* / scaled values) are computed internally by utils.preprocess_input
in the exact order the model was trained on.
"""

import base64
import os
import sys

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plotly.graph_objects as go  # noqa: E402
import utils  # noqa: E402

ILI = os.path.join(utils.ROOT, "assets", "illustrations")
CSS = os.path.join(utils.ROOT, "assets", "style.css")

st.set_page_config(page_title="Suncover · Predict", page_icon="🔮", layout="wide")
utils.load_css(CSS)


def svg_uri(name):
    """Base64 data-URI for an SVG in assets/illustrations."""
    with open(os.path.join(ILI, name), "rb") as f:
        return "data:image/svg+xml;base64," + base64.b64encode(f.read()).decode("ascii")


@st.cache_data
def load_dataset():
    """Raw insurance.csv, deduped exactly like train_model.py, plus bmi category."""
    df = pd.read_csv(os.path.join(utils.ROOT, "insurance.csv"))
    df = df.drop_duplicates().reset_index(drop=True)
    df["bmi_category"] = pd.cut(df["bmi"], bins=utils.BMI_BINS, labels=utils.BMI_LABELS)
    return df


def countup_html(value):
    """HTML component with a small inline-JS count-up for the predicted number.

    Streamlit strips <script> from st.markdown, so the count-up lives in its own
    sandboxed component (pure JS, no parent-DOM access needed).
    """
    return f"""
<link rel="stylesheet"
      href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700&family=IBM+Plex+Mono:wght@500;700&display=swap">
<div style="background:linear-gradient(135deg,#FFD54F 0%,#FFC107 100%);
            border:3px solid #1A1A1A;border-radius:4px;
            box-shadow:7px 7px 0 #1A1A1A;padding:16px 14px;">
  <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;letter-spacing:1px;
              text-transform:uppercase;color:#111111;margin-bottom:8px;">Estimated annual charge</div>
  <div class="nb-countup-value" style="font-family:'Space Grotesk','Archivo',sans-serif;
              font-size:44px;font-weight:700;color:#111111;line-height:1.25;">$0</div>
</div>
<script>
(function () {{
  var el = document.querySelector('.nb-countup-value');
  var target = {int(round(value))};
  var dur = 900; var t0 = null;
  function fmt(v) {{ return '$' + Math.round(v).toLocaleString('en-US'); }}
  function ease(t) {{ return 1 - Math.pow(1 - t, 3); }}
  function step(ts) {{
    if (!t0) t0 = ts;
    var p = Math.min((ts - t0) / dur, 1);
    el.textContent = fmt(target * ease(p));
    if (p < 1) requestAnimationFrame(step);
  }}
  requestAnimationFrame(step);
}})();
</script>
"""


def effect_chart(smoker_value, bmi_int):
    """Horizontal bar: how this profile's smoker/BMI status shifts the estimate."""
    df = load_dataset()
    overall = df["charges"].mean()

    sm_sub = df.loc[df["smoker"] == smoker_value, "charges"]
    user_cat = pd.cut([bmi_int], utils.BMI_BINS, labels=utils.BMI_LABELS)[0]
    bm_sub = df.loc[df["bmi_category"] == user_cat, "charges"]

    sm_effect = sm_sub.mean() - overall
    bm_effect = bm_sub.mean() - overall

    sm_label = "Smoker" if smoker_value == "yes" else "Non-smoker"
    bm_label = f"BMI · {user_cat}"

    fig = go.Figure(
        go.Bar(
            x=[bm_effect, sm_effect],
            y=[bm_label, sm_label],
            orientation="h",
            marker=dict(
                color=[
                    utils.AMBER if bm_effect >= 0 else utils.CORAL,
                    utils.CORAL if sm_effect > 0 else utils.YELLOW,
                ],
                line=dict(color=utils.INK, width=2),
            ),
        )
    )
    fig.add_vline(x=0, line_width=2, line_color=utils.INK)
    fig.update_layout(title="PULL VS DATASET AVERAGE ($)", height=300, showlegend=False)
    utils.theme_fig(fig)
    return fig


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.markdown('<div class="nb-eyebrow">SUNCOVER // PREDICTOR</div>', unsafe_allow_html=True)
st.markdown(
    '<h1 class="hero-title" style="font-size:2.15rem">Your profile, <span class="hl">scored</span> in seconds</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-sub">THE FORM ASKS ONLY WHAT insurance.csv ASKS — '
    "the model derives everything else internally.</p>",
    unsafe_allow_html=True,
)

ready = utils.artifacts_ready()

# ---------------------------------------------------------------------------
# Form (left)  /  Result panel (right)
# ---------------------------------------------------------------------------
left, right = st.columns([5, 6], gap="large", vertical_alignment="top")

with left:
    st.markdown('<div class="nb-card">', unsafe_allow_html=True)
    st.markdown('<div class="nb-eyebrow">STEP 1 · YOUR PROFILE</div>', unsafe_allow_html=True)
    st.markdown('<div class="nb-card-title">Raw inputs — no jargon</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="nb-section-sub">SAME FIELDS AS THE DATASET. NOTHING DERIVED, NOTHING HIDDEN.</p>',
        unsafe_allow_html=True,
    )

    with st.form("predict_form"):
        age = st.slider("Age", min_value=18, max_value=64, value=34, step=1)
        sex = st.radio("Biological sex", ["Female", "Male"], horizontal=True)
        bmi = st.number_input(
            "BMI", min_value=10.0, max_value=70.0, value=27.0, step=0.1, format="%.1f"
        )
        children = st.number_input("Children", min_value=0, max_value=5, value=0, step=1)
        smoker = st.toggle("Smoker", value=False)
        region = st.selectbox(
            "Region", ["southwest", "southeast", "northwest", "northeast"]
        )
        submitted = st.form_submit_button(
            "GET MY ESTIMATE →",
            disabled=not ready,
            help="Train the model first: run python train_model.py" if not ready else None,
        )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="nb-eyebrow">STEP 2 · YOUR ESTIMATE</div>', unsafe_allow_html=True)

    if not ready:
        st.markdown(
            f'<div class="nb-empty">'
            f'<img src="{svg_uri("empty_state.svg")}" alt="Model not trained yet"/>'
            f'<div class="nb-card-title">MODEL NOT TRAINED YET</div>'
            f'<p class="nb-section-sub">Open a terminal in this folder and run '
            f"<b>python train_model.py</b>, then refresh. It writes "
            f"<b>models/model.pkl</b>, <b>models/scaler.pkl</b> and "
            f"<b>models/feature_columns.pkl</b> by replaying main.ipynb.</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="nb-compare" style="padding-top:6px">'
            "THE FORM STAYS READY — YOUR PROFILE IS WAITING FOR THE MODEL.</p>",
            unsafe_allow_html=True,
        )

    elif not submitted:
        st.markdown(
            '<div class="nb-empty" style="border-style:solid">'
            f'<img src="{svg_uri("accent_arrow.svg")}" style="max-height:90px" alt=""/>'
            '<div class="nb-card-title">YOUR ESTIMATE LANDS HERE</div>'
            '<p class="nb-section-sub">FILL THE FORM ON THE LEFT AND HIT '
            "<b>GET MY ESTIMATE →</b>.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    else:
        model, _, _ = utils.load_artifacts()
        feature = utils.preprocess_input(
            age=age,
            sex="female" if sex == "Female" else "male",
            bmi=bmi,
            children=children,
            smoker="yes" if smoker else "no",
            region=region,
        )
        pred = float(model.predict(feature)[0])
        df = load_dataset()
        overall = float(df["charges"].mean())
        delta = pred - overall
        arrow = "UP" if delta >= 0 else "DOWN"
        sign = "+" if delta >= 0 else "−"

        components.html(countup_html(pred), height=185)

        st.markdown(
            f'<p class="nb-compare" style="padding-top:8px">'
            f'<span class="nb-badge" style="transform:rotate(1deg)">'
            f"{arrow} {sign}${abs(delta):,.0f} vs dataset avg ${overall:,.0f}</span>"
            f"</p>",
            unsafe_allow_html=True,
        )

        m = utils.model_metrics()
        if m:
            note = f"LINEAR REGRESSION · R² = {m[0]:.2f} · {m[2]:,} RECORDS"
        else:
            note = "LINEAR REGRESSION"
        st.markdown(f'<p class="nb-section-sub">{note}</p>', unsafe_allow_html=True)

        st.plotly_chart(
            effect_chart("yes" if smoker else "no", int(float(bmi))),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)
st.markdown('<p class="nb-eyebrow">WANT THE BIGGER PICTURE?</p>', unsafe_allow_html=True)
st.page_link("pages/2_📊_Dashboard.py", label="Open the Data Room →", icon="📊")
st.markdown(
    '<div class="nb-footer">SUNCOVER v1.0 · prediction is an estimate, not medical advice · '
    '<span class="nb-stamp">LINEAR REGRESSION</span></div>',
    unsafe_allow_html=True,
)