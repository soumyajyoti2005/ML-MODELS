<p align="center">
  <img src="assets/banner.svg" alt="Suncover — Smart Insurance Charge Predictor" width="100%"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-FFD54F?style=for-the-badge&logo=python&logoColor=111111" alt="Python"/>
  <img src="https://img.shields.io/badge/Streamlit-App-FFC107?style=for-the-badge&logo=streamlit&logoColor=111111" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Model-Linear%20Regression-FF6F59?style=for-the-badge" alt="Model"/>
  <img src="https://img.shields.io/badge/UI-Neo--Brutalist-111111?style=for-the-badge" alt="UI Style"/>
</p>

# Suncover

**Suncover** is a Streamlit web app that estimates annual medical insurance charges from a Linear Regression model, wrapped in a bold **Neo-Brutalist** interface — thick black borders, hard offset shadows, and a warm light-yellow / creamy-white palette.

---

## 🧭 Pages

| Page | What it does |
|---|---|
| 🏠 **Landing** | Hero intro, project pitch, quick links to Predict & Dashboard |
| 🔮 **Predict** | Enter your details → get an instant estimated charge |
| 📊 **Dashboard** | Interactive plots exploring the underlying insurance dataset |

---

## ✨ Features

- Instant charge estimate from a trained Linear Regression model
- Inputs match the raw dataset (age, sex, bmi, children, smoker, region) — no confusing engineered fields
- Interactive Plotly dashboard: distributions, correlations, smoker vs. non-smoker comparisons
- Distinct hand-crafted Neo-Brutalist look — no generic Streamlit defaults

---

## 🛠 Tech Stack

- **Python** · pandas · numpy · scikit-learn
- **Streamlit** for the UI
- **Plotly** for interactive charts
- **joblib** for model persistence

---

## 📂 Project Structure

```
insurance-predictor/
├── main.ipynb                  # original notebook (model exploration)
├── insurance.csv                # dataset
├── train_model.py               # trains + exports model artifacts
├── utils.py                     # shared preprocessing + CSS loader
├── models/
│   ├── model.pkl
│   ├── scaler.pkl
│   └── feature_columns.pkl
├── ui.py                        # Streamlit entry point (Landing page)
├── pages/
│   ├── 1_🔮_Predict.py
│   └── 2_📊_Dashboard.py
├── assets/
│   ├── banner.svg                # this README header
│   ├── illustrations/
│   └── style.css
└── requirements.txt
```

---

## 🚀 Getting Started

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (generates the .pkl files in models/)
python train_model.py

# 3. Launch the app
streamlit run ui.py
```

---

## 🔮 How Prediction Works

1. You enter the **raw** attributes: age, sex, bmi, children, smoker status, region.
2. `utils.py` converts them into the model's real feature set — sex/smoker mapping, region one-hot, BMI category one-hot, and scaling on age/bmi/children — using the same logic as `train_model.py`.
3. The saved Linear Regression model returns an estimated annual insurance charge.

---

## 📊 Dashboard Insights

- Distribution of charges across the dataset
- Charges by smoker vs. non-smoker
- Age vs. charges, colored by smoking status
- Correlation heatmap of numeric features
- Region and BMI-category breakdowns

---

## 🎨 Design Language

**Neo-Brutalism** on a **light yellow (`#FFD54F`) + creamy white (`#FFFDF6`)** palette: thick black borders, hard offset shadows (no blur), flat color blocks, oversized bold typography, and slightly tilted "scrapbook" elements — deliberately un-templated.

---

## 📄 License

MIT — feel free to adapt for your own dataset or model.

---

<p align="center"><sub>Built by Soumya</sub></p>