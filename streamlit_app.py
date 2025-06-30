# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------------
# Load Trained Model and PCA
# ------------------------
@st.cache_resource
def load_model_and_pca():
    model = joblib.load("xgboost_face_recognizer.pkl")
    pca = joblib.load("pca_transformer.pkl")
    return model, pca

model, pca = load_model_and_pca()

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Face Identity Verifier", layout="centered")
st.title("🧠 Facial Recognition Identity Verifier")
st.markdown("Upload a **face embedding CSV file** to verify identity (Arnie vs Non-Arnie).")

uploaded_file = st.file_uploader("📤 Upload embedding file (CSV format)", type=["csv"])

if uploaded_file is not None:
    try:
        # Read and display the uploaded embedding
        embedding_df = pd.read_csv(uploaded_file)
        st.subheader("📊 Uploaded Embedding Preview:")
        st.dataframe(embedding_df.head())

        # Ensure it's one row or reshape accordingly
        if embedding_df.shape[0] == 1:
            embedding = embedding_df.values
        elif embedding_df.shape[1] == 1:
            embedding = embedding_df.T.values
        else:
            st.warning("⚠️ Make sure the CSV has only one face embedding (single row or column).")
            st.stop()

        # PCA Transformation
        transformed = pca.transform(embedding)

        # Predict using the model
        prediction = model.predict(transformed)[0]
        proba = model.predict_proba(transformed)[0]

        # Show result
        label = "🟢 Arnie ✅" if prediction == 1 else "🔴 Not Arnie ❌"
        st.success(f"**Prediction:** {label}")
        st.write(f"**Confidence:** {round(np.max(proba) * 100, 2)}%")

    except Exception as e:
        st.error(f"🚫 Error reading the file: {e}")

# ------------------------
# Footer
# ------------------------
st.markdown("---")
st.caption("Developed by Khizar Saifee · Streamlit App · 2025")

