"""train_model.py — train & export the Heart Disease classifier.

Replays main.ipynb's cleaning + feature-engineering pipeline EXACTLY:

    1. Read heart.csv (918 Cleveland/Hungarian/Swiss/VA records).
    2. Zero-impute Cholesterol and RestingBP with the non-zero mean (rounded),
       exactly as the notebook does.
    3. One-hot encode every categorical column, then drop Sex_M /
       ExerciseAngina_N, rename Sex_F -> is_female and ExerciseAngina_Y ->
       is_ExerciseAngina, and cast the whole frame to int.
    4. Fit a StandardScaler over [Age, RestingBP, Cholesterol, MaxHR, Oldpeak]
       on the FULL dataset (same as the notebook).
    5. Benchmark the five models from main.ipynb (Logistic Regression, KNN,
       Decision Tree, Naive Bayes, SVM) plus Random Forest on the same
       80/20 hold-out (random_state=42) and keep the best by F1-score.

Artifacts written to models/:
    heart_disease_model.pkl     trained classifier (filename kept from uiprompt.md)
    scaler.pkl                  fitted StandardScaler
    feature_metadata.json       feature order, one-hot maps, slider limits, metrics

Run:  python train_model.py
"""

import json
import os

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(ROOT, "heart.csv")
MODELS_DIR = os.path.join(ROOT, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "heart_disease_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
META_PATH = os.path.join(MODELS_DIR, "feature_metadata.json")

# Final feature order used by the notebook (final_feature, minus the target).
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

# Human-readable slider / input limits (used by the Streamlit UI + metadata).
INPUT_LIMITS = {
    "Age": {"min": 20, "max": 80, "step": 1, "label": "Age (years)"},
    "RestingBP": {"min": 70, "max": 200, "step": 1, "label": "Resting Blood Pressure (mm Hg)"},
    "Cholesterol": {"min": 85, "max": 600, "step": 1, "label": "Serum Cholesterol (mg/dL)"},
    "MaxHR": {"min": 60, "max": 202, "step": 1, "label": "Maximum Heart Rate (bpm)"},
    "Oldpeak": {"min": 0.0, "max": 6.0, "step": 0.1, "label": "ST Depression — Oldpeak (0.0 – 6.0)"},
}


def load_cleaned_data(csv_path):
    """Reproduce main.ipynb cleaning + one-hot encoding + scaling.

    Returns (framed, scaler) where `framed` holds FINAL_FEATURES + target.
    """
    df = pd.read_csv(csv_path)

    # Notebook: zero-impute with the rounded non-zero mean.
    ch_mean = df.loc[df["Cholesterol"] != 0, "Cholesterol"].mean().round(0)
    df["Cholesterol"] = df["Cholesterol"].replace(to_replace=0, value=ch_mean)

    bp_mean = df.loc[df["RestingBP"] != 0, "RestingBP"].mean().round(0)
    df["RestingBP"] = df["RestingBP"].replace(to_replace=0, value=bp_mean)

    # One-hot encoding (alphabetical column order, same as pd.get_dummies).
    df_encode = pd.get_dummies(data=df)

    df_encode.drop(labels=["Sex_M", "ExerciseAngina_N"], axis=1, inplace=True)
    df_encode.rename(
        columns={"Sex_F": "is_female", "ExerciseAngina_Y": "is_ExerciseAngina"},
        inplace=True,
    )
    df_encode = df_encode.astype(int)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    df_encode[SCALED_COLS] = scaler.fit_transform(df_encode[SCALED_COLS])

    framed = df_encode[FINAL_FEATURES + ["HeartDisease"]]
    return framed, scaler


def build_models():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier

    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "KNN (n=5)": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": CalibratedClassifierCV(SVC(random_state=42)),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42
        ),
    }
def main():
    import joblib
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    framed, scaler = load_cleaned_data(CSV_PATH)

    X = framed.drop("HeartDisease", axis=1)
    y = framed["HeartDisease"]

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    results = []
    for name, model in build_models().items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        results.append(
            {
                "model": name,
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "f1_score": round(float(f1_score(y_test, y_pred)), 4),
            }
        )

    bench = pd.DataFrame(results).sort_values(
        by=["f1_score", "accuracy"], ascending=False
    )
    print("=== Benchmark (80/20 hold-out, random_state=42) ===")
    print(bench.to_string(index=False))

    # Best by F1, falling back to accuracy as a tie-break.
    best_row = bench.iloc[0].to_dict()
    best_name = best_row["model"]
    best_model = build_models()[best_name]

    # Refit the winner on the same 80% training split as the notebook.
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    accuracy = round(float(accuracy_score(y_test, y_pred)), 4)
    f1 = round(float(f1_score(y_test, y_pred)), 4)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    metadata = {
        "model_name": best_name,
        "task": "binary-classification",
        "classes": [0, 1],
        "class_names": {0: "No Heart Disease (Low Risk)", 1: "Heart Disease (Elevated Risk)"},
        "n_records": int(len(framed)),
        "test_size": 0.20,
        "random_state": 42,
        "accuracy": accuracy,
        "f1_score": f1,
        "final_features": FINAL_FEATURES,
        "scaled_features": SCALED_COLS,
        "input_limits": INPUT_LIMITS,
        "categorical_map": {
            "Sex": {"Female": 1, "Male": 0},
            "ChestPainType": {
                "ATA": {"ChestPainType_ASY": 0, "ChestPainType_ATA": 1, "ChestPainType_NAP": 0},
                "NAP": {"ChestPainType_ASY": 0, "ChestPainType_ATA": 0, "ChestPainType_NAP": 1},
                "ASY": {"ChestPainType_ASY": 1, "ChestPainType_ATA": 0, "ChestPainType_NAP": 0},
                "TA": {"ChestPainType_ASY": 0, "ChestPainType_ATA": 0, "ChestPainType_NAP": 0},
            },
            "RestingECG": {
                "Normal": {"RestingECG_Normal": 1, "RestingECG_ST": 0},
                "ST": {"RestingECG_Normal": 0, "RestingECG_ST": 1},
                "LVH": {"RestingECG_Normal": 0, "RestingECG_ST": 0},
            },
            "ExerciseAngina": {"No": 0, "Yes": 1},
            "FastingBS": {"Normal (<= 120 mg/dL)": 0, "Elevated (> 120 mg/dL)": 1},
            "ST_Slope": {
                "Up": {"ST_Slope_Flat": 0, "ST_Slope_Up": 1},
                "Flat": {"ST_Slope_Flat": 1, "ST_Slope_Up": 0},
                "Down": {"ST_Slope_Flat": 0, "ST_Slope_Up": 0},
            },
        },
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n=== Saved artifacts ===")
    print(f"  {MODEL_PATH}   <- {best_name}")
    print(f"  {SCALER_PATH}")
    print(f"  {META_PATH}")
    print(f"\nBest model: {best_name}  |  Accuracy: {accuracy:.4f}  |  F1: {f1:.4f}")
    print(f"Rows: {len(framed)}  |  Positive rate: {y.mean() * 100:.1f}%")


if __name__ == "__main__":
    main()