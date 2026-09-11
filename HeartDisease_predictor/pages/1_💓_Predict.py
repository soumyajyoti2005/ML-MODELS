"""pages/1_💓_Predict.py — Personal Heart Risk Assessment.

Collects the 11 clinical features from heart.csv through a calm, two-column
card interface and returns an instant risk probability + lifestyle advice
from the trained classifier (models/heart_disease_model.pkl).
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
import utils  # noqa: E402

st.set_page_config(page_title="PulseBloom · Risk Assessment", page_icon="💓", layout="wide")
_THEME = utils.apply_theme()  # day/night claymorphism CSS (night-aware svgs below)

META = utils.load_artifacts()[2] if utils.artifacts_ready() else None
LIMITS = (META or {}).get(
    "input_limits",
    {
        "Age": {"min": 20, "max": 80, "step": 1},
        "RestingBP": {"min": 70, "max": 200, "step": 1},
        "Cholesterol": {"min": 85, "max": 600, "step": 1},
        "MaxHR": {"min": 60, "max": 202, "step": 1},
        "Oldpeak": {"min": 0.0, "max": 6.0, "step": 0.1},
    },
)


def svg_uri(name):
    """Night-aware base64 data-URI for an SVG in assets/illustrations."""
    return utils.svg_uri(name)


CHEST_OPTIONS = ["ATA", "NAP", "ASY", "TA"]
CHEST_LABEL = {
    "ATA": "ATA · Atypical Angina",
    "NAP": "NAP · Non-Anginal Pain",
    "ASY": "ASY · Asymptomatic",
    "TA": "TA · Typical Angina",
}
ECG_OPTIONS = ["Normal", "ST", "LVH"]
ECG_LABEL = {
    "Normal": "Normal",
    "ST": "ST-T wave abnormality",
    "LVH": "Left Ventricular Hypertrophy",
}
SLOPE_OPTIONS = ["Up", "Flat", "Down"]
FS_OPTIONS = ["Normal (<= 120 mg/dL)", "Elevated (> 120 mg/dL)"]
EXERCISE_OPTIONS = ["No", "Yes"]


def fmt_chest(c):
    return CHEST_LABEL.get(c, c)


def fmt_ecg(c):
    return ECG_LABEL.get(c, c)


def fmt_fs(c):
    return "≤ 120 mg/dL · Normal" if c.startswith("Normal") else "> 120 mg/dL · Elevated"


def fmt_ex(c):
    return "No — no chest tightness" if c == "No" else "Yes — exercise-induced tightness"


def header_banner():
    """Calm page header."""
    st.markdown(
        '<span class="hero-eyebrow">💓&nbsp;PERSONAL RISK ASSESSMENT</span>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<h1 class="hero-title" style="font-size:2.3rem">Personal Heart '
        '<span class="hl">Risk Assessment</span></h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="hero-sub">Fill in your clinical indicators below to calculate '
        "your risk likelihood — instantly, privately, on this machine.</p>",
        unsafe_allow_html=True,
    )


def input_sidebar(col, title, subtitle, tint):
    """Open a soft white card inside a column and return nothing (renders via markdown)."""
    col.markdown(
        f'<div class="card-container-tint">'
        f'<div class="card-title">{title}</div>'
        f'<p class="card-sub">{subtitle}</p>'
        f"</div>",
        unsafe_allow_html=True,
    )


def probability_gauge(prob):
    """Soft gradient progress bar for the risk probability."""
    pct = max(0.0, min(100.0, prob * 100))
    color = "var(--mint-bg)" if prob < 0.5 else "var(--coral-bg)"
    return f"""
    <div style="background:{color};border:1px solid var(--pink-border);
                border-radius:9999px;height:20px;margin:0.4rem 0">
      <div style="width:{pct:.0f}%;height:20px;border-radius:9999px;
                  background:linear-gradient(90deg,#F9BACB,#D9CEF7)"></div>
    </div>
    """
def main():
    utils.side_brand()
    utils.clay_topbar()
    header_banner()

    if not utils.artifacts_ready():
        st.markdown(
            f'<div class="result-elevated" style="padding:2rem">'
            f'<img src="{svg_uri("empty_state.svg")}" style="width:170px" alt="Model not trained yet"/>'
            f'<div class="card-title" style="font-size:1.4rem">MODEL NOT TRAINED YET</div>'
            f'<p class="card-sub">Open a terminal in this folder and run '
            f"<b>python train_model.py</b> — it replays main.ipynb and writes "
            f"<b>models/heart_disease_model.pkl</b>, <b>models/scaler.pkl</b> and "
            f"<b>models/feature_metadata.json</b>. Then refresh this page.</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    left, right = st.columns(2, gap="large")

    with left:
        input_sidebar(left, "Vitals & Demographics", "Who you are & how your heart is doing right now.", "rose")

        sex = st.segmented_control("Sex", options=["Female", "Male"], default="Female", selection_mode="single")
        age = st.slider("Age (years)", min_value=20, max_value=80, value=47, step=1)
        resting_bp = st.slider("Resting Blood Pressure (mm Hg)", min_value=70, max_value=200, value=120, step=1)
        max_hr = st.slider("Maximum Heart Rate achieved (bpm)", min_value=60, max_value=202, value=140, step=1)
        exercise_angina = st.segmented_control(
            "Exercise Angina (induce chest tightness?)",
            options=EXERCISE_OPTIONS,
            default="No",
            selection_mode="single",
            format_func=fmt_ex,
        )

    with right:
        input_sidebar(right, "Laboratory & Diagnostic Findings", "Your lab panel, ECG signal & symptom profile.", "lavender")

        chest_pain_type = st.segmented_control(
            "Chest Pain Type", options=CHEST_OPTIONS, default="ATA",
            selection_mode="single", format_func=fmt_chest,
        )
        cholesterol = st.slider("Serum Cholesterol (mg/dL)", min_value=85, max_value=600, value=223, step=1)
        fasting_bs = st.segmented_control(
            "Fasting Blood Sugar (> 120 mg/dL?)",
            options=FS_OPTIONS,
            default=FS_OPTIONS[0],
            selection_mode="single",
            format_func=fmt_fs,
        )
        resting_ecg = st.selectbox("Resting ECG", options=ECG_OPTIONS, index=0, format_func=fmt_ecg)
        oldpeak = st.slider("Oldpeak — ST Depression (0.0 – 6.0)", min_value=0.0, max_value=6.0, value=0.8, step=0.1)
        st_slope = st.segmented_control("ST Slope", options=SLOPE_OPTIONS, default="Up", selection_mode="single")

    st.markdown('<div style="text-align:center;margin-top:0.2rem">', unsafe_allow_html=True)
    analyze = st.button("✨ Analyze Cardiovascular Risk", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    if not analyze:
        st.markdown(
            f'<div class="card-container-tint" style="text-align:center;padding:2.2rem">'
            f'<img src="{svg_uri("heart_shield.svg")}" style="width:130px" alt="Heart shield"/>'
            f'<div class="card-title" style="font-size:1.3rem">Your result will appear here</div>'
            f'<p class="card-sub">Tune the inputs on the left &amp; right, then press '
            f"<b>Analyze Cardiovascular Risk</b> for an instant, private reading.</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        feature = utils.preprocess_input(
            age=age,
            sex=sex,
            chest_pain_type=chest_pain_type,
            resting_bp=resting_bp,
            cholesterol=cholesterol,
            fasting_bs=fasting_bs,
            resting_ecg=resting_ecg,
            max_hr=max_hr,
            exercise_angina=exercise_angina,
            oldpeak=oldpeak,
            st_slope=st_slope,
        )
        model, _, metadata = utils.load_artifacts()
        prob = float(model.predict_proba(feature)[0][1])
        label = "Low Risk" if prob < 0.5 else "Elevated Risk"
        risk_prob = (1 - prob) * 100 if prob < 0.5 else prob * 100

        if label == "Low Risk":
            st.markdown(
                f'<div class="result-low">'
                f'<span class="pill-badge pill-badge-mint">🤍&nbsp;LOW RISK OUTCOME</span>'
                f'<div style="margin-top:0.4rem">'
                f'<p class="result-metric">Low Risk Probability</p>'
                f'<div class="result-num">{risk_prob:.0f}%</div>'
                f'<p class="result-sub">Your profile currently aligns with '
                f'<span class="result-hl">healthy cardiovascular patterns</span>. '
                f"Keep it up — consistency matters more than perfection.</p>"
                f'<div style="margin-top:0.6rem">{probability_gauge(prob)}</div>'
                f'<div class="card-title" style="font-size:1.05rem;margin-top:0.6rem">🌿 Wellness Notes</div>'
                f'<div class="advice-item"><span class="dot">·</span> Favour a '
                f"<b>Mediterranean-style diet</b>: olive oil, oily fish, legumes, "
                f"whole grains and plenty of vegetables.</div>"
                f'<div class="advice-item"><span class="dot">·</span> Target your '
                f"<b>aerobic zone</b>: roughly 60–75% of max HR "
                f"({0.60 * max_hr:.0f}–{0.75 * max_hr:.0f} bpm) for 150 minutes weekly.</div>"
                f'<div class="advice-item"><span class="dot">·</span> Keep resting BP '
                f"guided below 120/80 and cholesterol in check — small daily choices compound.</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="result-elevated">'
                f'<span class="pill-badge pill-badge-coral">💗&nbsp;ELEVATED RISK FACTOR DETECTED</span>'
                f'<div style="margin-top:0.4rem">'
                f'<p class="result-metric">Elevated Risk Probability</p>'
                f'<div class="result-num">{risk_prob:.0f}%</div>'
                f'<p class="result-sub">This profile shows '
                f'<span class="result-hl-coral">signs consistent with elevated cardiac risk</span>. '
                f"Please review these findings with a certified cardiologist.</p>"
                f'<div style="margin-top:0.6rem">{probability_gauge(prob)}</div>'
                f'<div class="card-title" style="font-size:1.05rem;margin-top:0.6rem">🩺 Recommended Next Steps</div>'
                f'<div class="advice-item"><span class="dot">·</span> Review with a '
                f"<b>certified cardiologist</b>; bring a copy of your vitals &amp; ECG report.</div>"
                f'<div class="advice-item"><span class="dot">·</span> Monitor '
                f"<b>blood pressure &amp; cholesterol</b> — discuss medication or lifestyle "
                f"changes before starting them.</div>"
                f'<div class="advice-item"><span class="dot">·</span> Gentle aerobic activity '
                f"(walking, cycling) with clinician approval may improve fitness safely.</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        with st.expander("🔬 What drove this readout?", expanded=False):
            contrib = top_contributors(feature, model, metadata)
            items = "".join(
                f'<div class="advice-item"><span class="dot">›</span> <b>{name}</b> — {desc}</div>'
                for name, desc in contrib
            )
            st.markdown(
                '<div class="card-container-tint">' + items + f'<p class="card-sub" style="margin-top:0.4rem">Model: '
                f"<b>{metadata.get('model_name','Random Forest')}</b> · applied on top of "
                f"the same 11 clinical features as <b>main.ipynb</b>.</p></div>",
                unsafe_allow_html=True,
            )


FEATURE_LABELS = {
    "Age": "Age",
    "RestingBP": "Resting Blood Pressure",
    "Cholesterol": "Serum Cholesterol",
    "MaxHR": "Max Heart Rate",
    "Oldpeak": "ST Depression (Oldpeak)",
    "is_female": "Female sex",
    "ChestPainType_ASY": "Asymptomatic chest pain",
    "ChestPainType_ATA": "Atypical angina",
    "ChestPainType_NAP": "Non-anginal pain",
    "RestingECG_Normal": "Normal resting ECG",
    "RestingECG_ST": "ST-T wave abnormality",
    "is_ExerciseAngina": "Exercise angina",
    "FastingBS": "Fasting blood sugar > 120",
    "ST_Slope_Flat": "Flat ST slope",
    "ST_Slope_Up": "Upward ST slope",
}


def top_contributors(feature, model, metadata):
    """Pick the 3 features that most influenced this prediction.

    Uses Random Forest feature importances (or absolute coefficients as a
    fallback) weighted by this patient's scaled input values, then returns
    friendly, human-readable explanations.
    """
    import numpy as np

    try:
        row = feature.iloc[0]
        imp = np.asarray(getattr(model, "feature_importances_", None), dtype=float)
        if imp is None or imp.shape[0] != len(utils.FINAL_FEATURES):
            coef = np.asarray(getattr(model, "coef_", None), dtype=float)
            imp = np.abs(coef).ravel() if coef is not None else np.ones(len(utils.FINAL_FEATURES))
        scores = {}
        for name, w, value in zip(utils.FINAL_FEATURES, imp, row.values):
            scores[name] = float(w) * abs(float(value))
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        picked = [name for name, _ in ranked[:3] if scores[name] > 0]
        if not picked:
            picked = [name for name, _ in ranked[:3]]

        explain = {
            "Age": "age stands out as a meaningful factor in this reading",
            "RestingBP": "resting blood pressure influences the model strongly",
            "Cholesterol": "serum cholesterol level is a key input in this readout",
            "MaxHR": "maximum achieved heart rate contributed to this score",
            "Oldpeak": "exercise-induced ST depression moved the prediction",
            "ChestPainType_ASY": "asymptomatic chest pain is a strong signal in this profile",
            "ChestPainType_ATA": "atypical angina was part of the signal",
            "ChestPainType_NAP": "non-anginal pain helped shape the prediction",
            "RestingECG_ST": "an ST-T wave abnormality on resting ECG is noted",
            "RestingECG_Normal": "a normal resting ECG supports the reading",
            "is_ExerciseAngina": "exercise angina is a recognised risk contributor",
            "FastingBS": "elevated fasting blood sugar added to the signal",
            "ST_Slope_Flat": "a flat ST slope is associated with risk",
            "ST_Slope_Up": "an upward ST slope is a favourable sign here",
            "is_female": "sex was part of the model's context",
        }
        return [(FEATURE_LABELS.get(n, n), explain.get(n, "contributed to the prediction")) for n in picked]
    except Exception:
        return [("Feature explanation unavailable", "retrain with `python train_model.py`.")]


main()

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)
utils.safe_page_link("pages/2_📈_Dashboard.py", label="Explore Dataset & Insights →", icon="📈", key="foot_dash")
st.markdown(
    '<div class="nb-footer">PulseBloom · CardiaCare v1.0 · prediction is a '
    "screening aid, not a diagnosis · "
    '<span class="pill-badge">RANDOM FOREST</span></div>',
    unsafe_allow_html=True,
)