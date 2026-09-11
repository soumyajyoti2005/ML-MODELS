"""train_model.py — train Suncover's Linear Regression model and export artifacts.

Replicates main.ipynb's pipeline byte-for-byte and saves:
    models/model.pkl            (scikit-learn LinearRegression)
    models/scaler.pkl           (StandardScaler fit on age, bmi, children)
    models/feature_columns.pkl  (ordered final feature list)

Run:  python train_model.py
Prints R^2 and adjusted R^2 on the same 80/20 hold-out (random_state=42)
so you can confirm parity with the notebook.
"""

import os

import pandas as pd

# ---------------------------------------------------------------------------
# Shared preprocessing contract — keep identical to utils.py
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

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")


def load_cleaned_data(csv_path):
    """Reproduce main.ipynb cells 27-45: clean + engineer features."""
    df = pd.read_csv(csv_path)

    # Notebook: df_cleaned = df  then  drop_duplicates(inplace=True)
    df = df.copy()
    df.drop_duplicates(inplace=True)

    # sex / smoker mappings + renames (cells 34-36)
    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})
    df.rename(columns={"sex": "is_female", "smoker": "is_smoker"}, inplace=True)

    # region one-hot (cell 38) + drop + int conversion (cell 39)
    for region in REGION_COLS:
        df[region] = df["region"] == region
    df.drop(labels=["region"], axis=1, inplace=True)
    df = df.astype(int)  # NOTE: truncates the float BMI column, exactly like the notebook

    # BMI binning (cell 42) + one-hot (cell 43) + int conversion (cell 45)
    df["bmi_category"] = pd.cut(df["bmi"], bins=BMI_BINS, labels=BMI_LABELS)
    df = pd.get_dummies(df, columns=["bmi_category"])
    df = df.astype(int)

    return df


def main():
    import joblib
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "insurance.csv")
    df = load_cleaned_data(csv_path)

    # Scale age / bmi / children on the FULL dataset (notebook cell 47)
    scaler = StandardScaler()
    df[SCALED_COLS] = scaler.fit_transform(df[SCALED_COLS])

    final_df = df[FINAL_FEATURES + ["charges"]]
    X = final_df.drop("charges", axis=1)
    y = final_df["charges"]

    # Same split as the notebook (cell 69)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    # Evaluation on the hold-out (cells 73-76)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    n = x_test.shape[0]
    p = x_test.shape[1]
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(FINAL_FEATURES, FEATURES_PATH)

    print(f"Rows after dedup: {len(df)}")
    print(f"R^2  = {r2:.4f}")
    print(f"Adjusted R^2 = {adjusted_r2:.4f}")
    print("Saved:")
    print(f"  {MODEL_PATH}")
    print(f"  {SCALER_PATH}")
    print(f"  {FEATURES_PATH}")


if __name__ == "__main__":
    main()