"""utils.py — shared helpers for the PulseBloom Heart Disease Streamlit app.

Contains:
    load_css(path)               inject assets/style.css
    svg_uri(name)                base64 data-URI for assets/illustrations/*
    load_artifacts()             cached model / scaler / feature metadata
    artifacts_ready()            are the trained artifacts present?
    preprocess_input(...)        build a model-ready row from raw UI inputs
    load_dataset()               deduped heart.csv with derived columns
    model_metrics()              replay hold-out metrics with saved artifacts
    theme_fig(fig, title)        soft cream/pink/violet Plotly theme

The preprocessing constants below MUST stay identical to train_model.py
(same one-hot maps, same column order, same scaler) — change both together.
"""

import base64
import json
import os

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(ROOT, "models", "heart_disease_model.pkl")
SCALER_PATH = os.path.join(ROOT, "models", "scaler.pkl")
META_PATH = os.path.join(ROOT, "models", "feature_metadata.json")

# Shared preprocessing contract — keep identical to train_model.py.
FINAL_FEATURES = [
    "is_female",
    "ChestPainType_ASY",
    "ChestPainType_ATA",
    "ChestPainType_NAP",
    "RestingECG_Normal",
    "RestingECG_ST",
    "is_ExerciseAngina",
    "FastingBS",
    "ST_Slope_Flat",
    "ST_Slope_Up",
    "Age",
    "RestingBP",
    "Cholesterol",
    "MaxHR",
    "Oldpeak",
]
SCALED_COLS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]

# Design tokens (mirrored in assets/style.css)
CREAM = "#FAF8F5"
WHITE = "#FFFFFF"
PINK = "#F9BACB"
PINK_SOFT = "#FFF0F5"
PINK_BORDER = "#FAD0DD"
VIOLET = "#D9CEF7"
VIOLET_SOFT = "#F3EEFF"
VIOLET_DEEP = "#7D679E"
INK = "#2E2633"
MAUVE = "#7E7385"
MINT = "#D9F5E3"
MINT_DEEP = "#237346"
CORAL = "#FFE2E6"
CORAL_DEEP = "#B83248"

# Night palette (mirrored in assets/theme_night.css)
N_BG = "#221C2B"
N_BG2 = "#2A2338"
N_SURFACE = "#2E2740"
N_SURFACE2 = "#362C4A"
N_PINK = "#EBA6C4"
N_PINK_SOFT = "#3C2238"
N_PINK_BORDER = "#5E3448"
N_VIOLET = "#B9A1E8"
N_VIOLET_SOFT = "#2E2547"
N_VIOLET_DEEP = "#B39BEA"
N_VIOLET_BORDER = "#5A4A80"
N_INK = "#F4EFFA"
N_MAUVE = "#BCB0CE"
N_GRID = "#433A55"
N_MINT = "#72DFA6"
N_MINT_BG = "#1F3A2C"
N_CORAL = "#FF9DB4"
N_CORAL_BG = "#48202E"

# Theme state (persists across pages inside one browser session)
THEME_KEY = "pulse_theme"
BASE_CSS = os.path.join(ROOT, "assets", "style.css")
NIGHT_CSS = os.path.join(ROOT, "assets", "theme_night.css")
HAND_CSS = os.path.join(ROOT, "assets", "handwriting.css")


def _svg_uri(path):
    """Base64 data-URI for an SVG file (no server request needed)."""
    with open(path, "rb") as f:
        return "data:image/svg+xml;base64," + base64.b64encode(f.read()).decode("ascii")


def load_css(path):
    """Read a CSS file and inject it into the page."""
    with open(path, "r", encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    """Load trained model artifacts once and cache them.

    sklearn / joblib are imported lazily so the pages still render a friendly
    \"run train_model.py\" state before the training deps are fully available.
    """
    import joblib

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, scaler, metadata


def artifacts_ready():
    """True when all three model artifacts exist on disk."""
    return all(os.path.isfile(p) for p in (MODEL_PATH, SCALER_PATH, META_PATH))


def preprocess_input(
    age,
    sex,
    chest_pain_type,
    resting_bp,
    cholesterol,
    fasting_bs,
    resting_ecg,
    max_hr,
    exercise_angina,
    oldpeak,
    st_slope,
):
    """Turn raw, user-entered inputs into the model's exact feature row.

    Mirrors train_model.py's one-hot mapping and column order, then applies
    the SAME StandardScaler the model was trained with.

    sex            : "Female" | "Male"
    chest_pain_type: "ATA" | "NAP" | "ASY" | "TA"
    resting_ecg    : "Normal" | "ST" | "LVH"
    exercise_angina: "Yes" | "No"
    fasting_bs     : "Normal (<= 120 mg/dL)" | "Elevated (> 120 mg/dL)"
    st_slope       : "Up" | "Flat" | "Down"
    """
    _, scaler, _ = load_artifacts()

    row = pd.DataFrame(
        [
            {
                "is_female": 1 if sex == "Female" else 0,
                "ChestPainType_ASY": int(chest_pain_type == "ASY"),
                "ChestPainType_ATA": int(chest_pain_type == "ATA"),
                "ChestPainType_NAP": int(chest_pain_type == "NAP"),
                "RestingECG_Normal": int(resting_ecg == "Normal"),
                "RestingECG_ST": int(resting_ecg == "ST"),
                "is_ExerciseAngina": 1 if exercise_angina == "Yes" else 0,
                "FastingBS": 1 if fasting_bs == "Elevated (> 120 mg/dL)" else 0,
                "ST_Slope_Flat": int(st_slope == "Flat"),
                "ST_Slope_Up": int(st_slope == "Up"),
                "Age": float(age),
                "RestingBP": float(resting_bp),
                "Cholesterol": float(cholesterol),
                "MaxHR": float(max_hr),
                "Oldpeak": float(oldpeak),
            }
        ]
    )

    row[SCALED_COLS] = scaler.transform(row[SCALED_COLS])
    return row[FINAL_FEATURES]


@st.cache_data
def load_dataset():
    """Deduped heart.csv with friendly derived columns for the UI/dashboard."""
    df = pd.read_csv(os.path.join(ROOT, "heart.csv"))
    df = df.drop_duplicates().reset_index(drop=True)

    df["is_female"] = (df["Sex"] == "F").astype(int)
    df["is_ExerciseAngina"] = (df["ExerciseAngina"] == "Y").astype(int)
    df["Risk"] = df["HeartDisease"].map({0: "No Heart Disease", 1: "Heart Disease"})
    df["SexLabel"] = df["Sex"].map({"M": "Male", "F": "Female"})
    df["ChestPainLabel"] = df["ChestPainType"].map(
        {"ATA": "Atypical Angina", "NAP": "Non-Anginal", "ASY": "Asymptomatic", "TA": "Typical Angina"}
    )
    df["AgeBand"] = pd.cut(
        df["Age"],
        bins=[0, 39, 49, 59, 69, 100],
        labels=["< 40", "40–49", "50–59", "60–69", "70+"],
    )
    return df


def theme_fig(fig, title=None, dark=None):
    """Apply the PulseBloom clay palette to a plotly figure.

    dark=None auto-detects the active session theme; True/False forces it.
    """
    if dark is None:
        dark = get_theme() == "night"

    if dark:
        paper, plot, grid, ink, soft = N_BG, N_BG2, N_GRID, N_INK, N_MAUVE
        border, colorway = N_VIOLET_BORDER, [N_PINK, N_VIOLET, "#7FD6B4", N_MINT_BG, N_VIOLET_DEEP, "#C98BB0"]
        mint_bg, coral_bg = N_MINT_BG, N_CORAL_BG
    else:
        paper, plot, grid, ink, soft = CREAM, WHITE, "#F0E6EC", INK, MAUVE
        border, colorway = PINK_BORDER, [PINK, VIOLET, "#A9C9B6", MINT, VIOLET_DEEP, "#E9B0C1"]
        mint_bg, coral_bg = MINT, CORAL

    fig.update_layout(
        paper_bgcolor=paper,
        plot_bgcolor=plot,
        font=dict(family="Plus Jakarta Sans, -apple-system, BlinkMacSystemFont, sans-serif", color=ink, size=12),
        title=title,
        title_font=dict(family="Plus Jakarta Sans, sans-serif", size=15, color=ink),
        margin=dict(l=40, r=30, t=60 if title else 35, b=40),
        legend=dict(
            bgcolor=plot,
            bordercolor=border,
            borderwidth=1,
            font=dict(family="Plus Jakarta Sans, sans-serif", color=soft, size=11),
            orientation="h",
        ),
        colorway=colorway,
        hoverlabel=dict(bgcolor=plot, bordercolor=border, font=dict(color=ink, size=12)),
    )
    fig.update_xaxes(
        showgrid=True, gridcolor=grid, gridwidth=0.6, zeroline=False,
        tickfont=dict(family="Plus Jakarta Sans, sans-serif", color=soft),
        linecolor=border,
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=grid, gridwidth=0.6, zeroline=False,
        tickfont=dict(family="Plus Jakarta Sans, sans-serif", color=soft),
        linecolor=border,
    )
    return fig


@st.cache_resource
def model_metrics():
    """Replay the training pipeline with the SAVED artifacts and evaluate.

    Returns (accuracy, f1, n_rows, model_name) or None when artifacts are missing.
    """
    if not artifacts_ready():
        return None
    try:
        import joblib  # noqa: F401
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        return None

    model, scaler, metadata = load_artifacts()
    df = pd.read_csv(os.path.join(ROOT, "heart.csv"))

    ch_mean = df.loc[df["Cholesterol"] != 0, "Cholesterol"].mean().round(0)
    df["Cholesterol"] = df["Cholesterol"].replace(to_replace=0, value=ch_mean)
    bp_mean = df.loc[df["RestingBP"] != 0, "RestingBP"].mean().round(0)
    df["RestingBP"] = df["RestingBP"].replace(to_replace=0, value=bp_mean)

    enc = pd.get_dummies(data=df)
    enc.drop(labels=["Sex_M", "ExerciseAngina_N"], axis=1, inplace=True)
    enc.rename(columns={"Sex_F": "is_female", "ExerciseAngina_Y": "is_ExerciseAngina"}, inplace=True)
    enc = enc.astype(int)
    enc[SCALED_COLS] = scaler.transform(enc[SCALED_COLS])

    framed = enc[FINAL_FEATURES + ["HeartDisease"]]
    X = framed.drop("HeartDisease", axis=1)
    y = framed["HeartDisease"]
    _, x_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    y_pred = model.predict(x_test)
    acc = float(accuracy_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    return acc, f1, len(framed), metadata.get("model_name", "Random Forest")


# ---------------------------------------------------------------------------
# Theme system — day / night clay switching
# ---------------------------------------------------------------------------
def get_theme():
    """Active theme: 'day' | 'night' (persists across pages in the session)."""
    if THEME_KEY not in st.session_state:
        st.session_state[THEME_KEY] = "day"
    return st.session_state[THEME_KEY]


def set_theme(mode):
    st.session_state[THEME_KEY] = "night" if mode == "night" else "day"


def apply_theme():
    """Inject the claymorphism base (day) CSS + night overrides when active.

    Returns the active theme so pages can theme charts/art accordingly.
    """
    theme = get_theme()
    load_css(BASE_CSS)
    if theme == "night":
        load_css(NIGHT_CSS)
    load_css(HAND_CSS)  # handwritten (Excalidraw-style) type + text-size polish, wins over both themes
    return theme


def theme_switch():
    """Squishy clay day/night toggle (top-right on every page)."""
    theme = get_theme()
    night = theme == "night"
    label = "🌙 Night" if not night else "☀️ Day"
    if st.button(label, key="pulse_theme_switch", help="Toggle the clay theme"):
        set_theme("night" if not night else "day")
        st.rerun()
    return get_theme()


def clay_topbar():
    """Clay navigation bar: logo + AI pill left, theme toggle at the RIGHT corner.

    Returns the active theme after the switch has been rendered.
    """
    left, mid, right = st.columns([1.25, 2.4, 0.62], vertical_alignment="center")
    with left:
        st.markdown(
            '<div class="clay-logo"><span class="clay-logo-mark">💓</span>'
            "PulseBloom&nbsp;·&nbsp;CardiaCare</div>",
            unsafe_allow_html=True,
        )
    with mid:
        st.markdown(
            '<span class="clay-pill clay-pill-violet">✨&nbsp;AI&nbsp;Mode</span> '
            '<span class="clay-pill">🔔</span> '
            '<span class="clay-avatar">SO</span>',
            unsafe_allow_html=True,
        )
    with right:
        theme_switch()  # ☀️ / 🌙 clay toggle, top-right corner
    return get_theme()


def safe_page_link(page, label, icon=None, key=None):
    """st.page_link with a switch_page-button fallback.

    When the target page is not registered in the app's page map (e.g. a page
    file run standalone, or AppTest page-entry runs) st.page_link raises
    KeyError('url_pathname'); a plain switch_page button keeps navigation alive.
    """
    button_key = key or f"nav_{label}"
    try:
        st.page_link(page, label=label, icon=icon)
    except Exception:
        st.button(
            label,
            key=button_key,
            icon=icon,
            on_click=lambda p=page: st.switch_page(p),
            use_container_width=True,
        )


def side_brand():
    """Clay sidebar shared by every page: vector illustration brand + nav + note."""
    with st.sidebar:
        theme = get_theme()
        illo = "side_illustration.svg" if theme == "day" else "side_illustration_night.svg"
        st.markdown(
            f'<div class="side-brand">'
            f'<img class="side-illo" src="{svg_uri(illo)}" alt="PulseBloom heart illustration"/>'
            f'<div class="side-brand-name"><span class="clay-logo-mark">💓</span> PulseBloom</div>'
            f'<div class="side-brand-sub">CardiaCare Studio · clay edition</div>'
            f"</div>",
            unsafe_allow_html=True,
        )
        safe_page_link("ui.py", label="🏠  Home", key="nav_home")
        safe_page_link("pages/1_💓_Predict.py", label="💓  Risk Predict", key="nav_predict")
        safe_page_link("pages/2_📈_Dashboard.py", label="📈  Insights", key="nav_insights")
        st.markdown(
            '<div class="side-note">🫀 Trained on the <b>918-patient</b> heart dataset — '
            "same pipeline as <b>main.ipynb</b>, wrapped in a soft clay UI. "
            "Flip the ☀️/🌙 switch in the top-right corner to change mood.</div>",
            unsafe_allow_html=True,
        )


def page_heading(eyebrow, title_html, sub, size="2.4rem"):
    """Consistent clay page header."""
    st.markdown(f'<span class="clay-kicker">{eyebrow}</span>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="clay-title" style="font-size:{size};margin-top:0.1rem">{title_html}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<p class="clay-sub">{sub}</p>', unsafe_allow_html=True)


def gauge_html(prob, label_prefix="Risk Probability"):
    """Soft clay gauge bar (mint when low risk, coral when elevated)."""
    pct = max(0.0, min(100.0, prob * 100))
    track = "var(--mint-bg)" if prob < 0.5 else "var(--coral-bg)"
    return (
        f'<div style="display:flex;justify-content:space-between;margin-bottom:4px">'
        f'<span style="color:var(--ink-soft);font-size:1.05rem;font-weight:600">{label_prefix}</span>'
        f'<span style="color:var(--ink);font-size:1.1rem;font-weight:700">{pct:.0f}%</span></div>'
        f'<div class="clay-gauge" style="--gauge-bg:{track}"><div style="width:{pct:.0f}%"></div></div>'
    )


def svg_uri(name, dark=None):
    """Base64 data-URI for an SVG in assets/illustrations.

    Night-aware: when the session theme is 'night', prefers the
    <name>_night.svg variant when it exists (e.g. hero_cardio_night.svg).
    """
    base = os.path.join(ROOT, "assets", "illustrations", name)
    if dark is None:
        dark = get_theme() == "night"
    if dark:
        stem, ext = os.path.splitext(name)
        variant = os.path.join(ROOT, "assets", "illustrations", f"{stem}_night{ext}")
        if os.path.isfile(variant):
            return _svg_uri(variant)
    return _svg_uri(base)