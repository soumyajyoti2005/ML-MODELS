"""utils.py — shared helpers for the Suncover Streamlit app.

Contains:
    load_css(path)                       inject assets/style.css
    preprocess_input(...)                build a model-ready row from raw inputs
    load_artifacts()                     cached model / scaler / feature columns

The preprocessing constants below MUST stay identical to train_model.py
(same bmi bins, same column order, same scaler) — change both together.
"""

import os

import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Shared preprocessing contract — keep identical to train_model.py
# ---------------------------------------------------------------------------
BMI_BINS = [0.0, 18.5, 24.9, 29.9, float("inf")]
BMI_LABELS = ["underweight", "Normal", "overweight", "obese"]
REGION_COLS = ["southwest", "southeast", "northwest", "northeast"]
SCALED_COLS = ["age", "bmi", "children"]
FINAL_FEATURES = [
    "is_smoker",
    "bmi_category_obese",
    "southeast",
    "is_female",
    "age",
    "bmi",
    "children",
]

MODEL_PATH = os.path.join(ROOT, "models", "model.pkl")
SCALER_PATH = os.path.join(ROOT, "models", "scaler.pkl")
FEATURES_PATH = os.path.join(ROOT, "models", "feature_columns.pkl")

# Design tokens (shared with assets/style.css)
CREAM = "#FFFDF6"
YELLOW = "#FFD54F"
AMBER = "#FFC107"
CORAL = "#FF6F59"
INK = "#1A1A1A"


def load_css(path):
    """Read a CSS file and inject it into the page."""
    with open(path, "r", encoding="utf-8") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    """Load trained model artifacts once and cache them.

    joblib / scikit-learn are imported lazily so pages still render (with a
    friendly \"run train_model.py\" state) before the training dependencies
    are installed.
    """
    import joblib

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = list(joblib.load(FEATURES_PATH))
    return model, scaler, feature_columns


def artifacts_ready():
    """True when all three model artifacts exist on disk."""
    return all(os.path.isfile(p) for p in (MODEL_PATH, SCALER_PATH, FEATURES_PATH))


def preprocess_input(age, sex, bmi, children, smoker, region):
    """Turn raw, user-entered inputs into the model's exact feature row.

    Mirrors train_model.py's cleaning pipeline step-for-step:
    sex/smoker mapping, region one-hot, bmi binning, int truncation of the
    bmi column, then StandardScaler transform, in FINAL_FEATURES order.
    """
    row = pd.DataFrame(
        [
            {
                "age": int(age),
                "sex": sex,  # "male" | "female"
                "bmi": float(bmi),
                "children": int(children),
                "smoker": smoker,  # "yes" | "no"
                "region": region,
            }
        ]
    )

    # Same mappings + renames as train_model.py
    row["sex"] = row["sex"].map({"male": 0, "female": 1})
    row["smoker"] = row["smoker"].map({"no": 0, "yes": 1})
    row.rename(columns={"sex": "is_female", "smoker": "is_smoker"}, inplace=True)

    # Region one-hot, then int conversion (truncates bmi exactly like training)
    for r in REGION_COLS:
        row[r] = row["region"] == r
    row.drop(labels=["region"], axis=1, inplace=True)
    row = row.astype(int)

    # BMI binning + one-hot + int conversion
    row["bmi_category"] = pd.cut(row["bmi"], bins=BMI_BINS, labels=BMI_LABELS)
    row = pd.get_dummies(row, columns=["bmi_category"])
    row = row.astype(int)

    # Scale with the SAME scaler the model was trained with
    _, scaler, feature_columns = load_artifacts()
    row[SCALED_COLS] = scaler.transform(row[SCALED_COLS])

    # Guarantee the exact column order the model was fit on
    return row[feature_columns]


def theme_fig(fig, title=None):
    """Apply the Suncover neo-brutalist look to a plotly figure.

    Cream backgrounds, pure-black axis lines, flat fills. Meant for the
    Interactive Ecosystem of the dashboard + the predict-page effect chart.
    """
    import plotly.graph_objects as go  # noqa: F401  (kept for parity)

    fig.update_layout(
        paper_bgcolor=CREAM,
        plot_bgcolor=CREAM,
        font=dict(family="IBM Plex Mono, monospace", color="#111111", size=12),
        title_font=dict(
            family="Space Grotesk, sans-serif", size=16, color="#111111"
        ),
        title=title,
        margin=dict(l=20, r=20, t=60 if title else 30, b=20),
        legend=dict(
            bgcolor=CREAM,
            bordercolor=INK,
            borderwidth=2,
            font=dict(family="IBM Plex Mono, monospace", color="#111111", size=12),
        ),
        colorway=[YELLOW, AMBER, CORAL, "#FFE49A", "#FFB300"],
    )
    fig.update_xaxes(
        showline=True, linewidth=2, linecolor=INK,
        gridcolor=INK, gridwidth=0.4, zeroline=False,
        tickfont=dict(family="IBM Plex Mono, monospace", color="#111111"),
    )
    fig.update_yaxes(
        showline=True, linewidth=2, linecolor=INK,
        gridcolor=INK, gridwidth=0.4, zeroline=False,
        tickfont=dict(family="IBM Plex Mono, monospace", color="#111111"),
    )
    return fig


@st.cache_resource
def model_metrics():
    """Replay the training pipeline with the SAVED artifacts and evaluate.

    Returns (r2, adjusted_r2, n_rows) or None when artifacts are missing.
    """
    if not artifacts_ready():
        return None
    try:
        from sklearn.linear_model import LinearRegression  # noqa: F401  (parity check)
        from sklearn.metrics import r2_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        return None

    model, scaler, feature_columns = load_artifacts()
    df = pd.read_csv(os.path.join(ROOT, "insurance.csv"))
    df = df.copy()
    df.drop_duplicates(inplace=True)

    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})
    df.rename(columns={"sex": "is_female", "smoker": "is_smoker"}, inplace=True)
    for r in REGION_COLS:
        df[r] = df["region"] == r
    df.drop(labels=["region"], axis=1, inplace=True)
    df = df.astype(int)
    df["bmi_category"] = pd.cut(df["bmi"], bins=BMI_BINS, labels=BMI_LABELS)
    df = pd.get_dummies(df, columns=["bmi_category"])
    df = df.astype(int)
    df[SCALED_COLS] = scaler.transform(df[SCALED_COLS])

    X = df[feature_columns]
    y = df["charges"]
    _, x_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    r2 = r2_score(y_test, model.predict(x_test))
    n = x_test.shape[0]
    p = x_test.shape[1]
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    return r2, adjusted_r2, len(df)