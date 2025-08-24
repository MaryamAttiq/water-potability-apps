import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Water Potability Predictor", page_icon="💧", layout="centered")

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)

st.title("💧 Water Potability Prediction")
st.write("Enter water quality values and predict whether the water is **Potable (1)** or **Not Potable (0)**.")

model_path = "outputs/model.pkl"  # default relative path
try:
    model = load_model(model_path)
    model_loaded = True
except Exception as e:
    st.warning("Model not found. Please upload a trained model.pkl or train locally and include it in the repo under `outputs/model.pkl`.")
    model = None
    model_loaded = False

# Feature inputs
st.subheader("Input Features")
col1, col2 = st.columns(2)

with col1:
    ph = st.number_input("pH (0–14)", min_value=0.0, max_value=14.0, value=7.0, step=0.1, help="Acidity/alkalinity level")
    hardness = st.number_input("Hardness (mg/L)", min_value=0.0, value=150.0, step=1.0)
    solids = st.number_input("Solids / TDS (ppm)", min_value=0.0, value=10000.0, step=100.0)
    chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, value=7.0, step=0.1)
    sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, value=300.0, step=1.0)
with col2:
    conductivity = st.number_input("Conductivity (μS/cm)", min_value=0.0, value=400.0, step=1.0)
    organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, value=12.0, step=0.1)
    trihalo = st.number_input("Trihalomethanes (μg/L)", min_value=0.0, value=60.0, step=0.1)
    turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=3.0, step=0.1)

input_df = pd.DataFrame([{
    'ph': ph,
    'Hardness': hardness,
    'Solids': solids,
    'Chloramines': chloramines,
    'Sulfate': sulfate,
    'Conductivity': conductivity,
    'Organic_carbon': organic_carbon,
    'Trihalomethanes': trihalo,
    'Turbidity': turbidity
}])

predict_btn = st.button("Predict")

if predict_btn:
    if not model_loaded:
        st.error("Model file missing. Please add `outputs/model.pkl` to run predictions.")
    else:
        proba = model.predict_proba(input_df)[0, 1]
        pred = int(proba >= 0.5)
        st.success(f"Prediction: **{pred}** (Probability of Potable: {proba:.2%})")

        # Show feature importances if available
        try:
            rf = model.named_steps['model']
            importances = rf.feature_importances_
            fi = pd.DataFrame({'feature': input_df.columns, 'importance': importances}).sort_values('importance', ascending=False).set_index('feature')
            st.subheader("Feature Importances")
            st.bar_chart(fi)
        except Exception:
            st.info("Feature importances not available for this model.")

st.divider()

st.subheader("Batch Prediction (optional)")
st.write("Upload a CSV with columns: ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity")
file = st.file_uploader("Upload CSV", type=['csv'])

if file and model_loaded:
    try:
        df = pd.read_csv(file)
        proba = model.predict_proba(df)[:, 1]
        preds = (proba >= 0.5).astype(int)
        out = df.copy()
        out['potability_proba'] = proba
        out['potability_pred'] = preds
        st.dataframe(out.head(50))
        st.download_button("Download Predictions CSV", data=out.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
    except Exception as e:
        st.error(f"CSV error: {e}")
