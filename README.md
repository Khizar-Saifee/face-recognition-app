
# 🔒 Facial Recognition Identity Verification App

A machine learning-powered facial recognition system that verifies user identities based on face embeddings.  
Built using **XGBoost**, **SVM**, and **Random Forest**, the system processes facial feature vectors and predicts whether the face belongs to a known identity (e.g., "Arnie") or not.

This project includes:

- 📦 Trained models (`xgboost_face_recognizer.pkl`, `pca_transformer.pkl`)
- 🧠 Dimensionality reduction using **PCA**
- 🌐 A Streamlit app for interactive, real-time predictions
- 📈 Visual comparisons (confusion matrix, ROC curve, F1 vs accuracy)
- 🛡️ Designed with ethics and privacy in mind

---

## 🚀 Live App (Streamlit Cloud)

👉 [https://face-recognition-app-as84vygzju7gouac84cgah.streamlit.app/]

---

## 📁 Folder Contents

| File/Folder            | Description                                  |
|------------------------|----------------------------------------------|
| `streamlit_app.py`     | Main Streamlit interface for model inference |
| `xgboost_face_recognizer.pkl` | Trained XGBoost model file             |
| `pca_transformer.pkl`  | PCA transformer used to reduce dimensions    |
| `requirements.txt`     | All Python packages needed for deployment    |

---

## 📤 How to Use

1. Clone this repository or download ZIP
2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app locally:  
   ```bash
   streamlit run streamlit_app.py
   ```
4. Upload a CSV file with face embeddings to get predictions

---

## 🙌 Credits

Developed by **Khizar Saifee**  
2025 · For educational & secure identity verification use
