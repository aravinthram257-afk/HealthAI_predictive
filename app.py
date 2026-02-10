import streamlit as st
import torch
import pandas as pd
from pathlib import Path
import joblib
from dl_module import LosNet, load_los
# ===== Load LOS model + scaler =====
@st.cache_resource
def load_los_model_and_scaler():
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    model_path = models_dir / "los_net.pth"
    df = load_los()
    target_col = "lengthofstay"
    y = df[target_col]
    X = (
        df.select_dtypes(include=["int64", "float64"])
        .drop(columns=[target_col], errors="ignore")
        .fillna(0)
    )
from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X)
    model = LosNet(X.shape[1])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    feature_columns = X.columns.tolist()
    return model, scaler, feature_columns
# ===== Load Sentiment TF-IDF + LogisticRegression =====
@st.cache_resource
def load_sentiment_ml_model():
    base_dir = Path(__file__).resolve().parent.parent
    models_dir = base_dir / "models"
    vec_path = models_dir / "sentiment_tfidf.joblib"
    clf_path = models_dir / "sentiment_logreg.joblib"
    vectorizer = joblib.load(vec_path)
    clf = joblib.load(clf_path)
    return vectorizer, clf
# ===== Streamlit UI =====
st.set_page_config(page_title="HealthAI Demo", layout="wide")
st.title("HealthAI Predictive Suite – Demo")
tab_los, tab_nlp = st.tabs(["LOS Prediction", "Sentiment Analysis"])
# --- LOS tab ---
with tab_los:
    st.header("Length of Stay Prediction")

    model_los, scaler_los, feature_cols = load_los_model_and_scaler()

    st.write("Enter numeric feature values to predict **length of stay** (days).")

    user_inputs = {}
    cols = st.columns(3)
    for i, col_name in enumerate(feature_cols):
        with cols[i % 3]:
            user_inputs[col_name] = st.number_input(
                col_name, value=0.0, step=1.0, format="%.2f"
            )

    if st.button("Predict LOS"):
        x_df = pd.DataFrame([user_inputs])[feature_cols]
        x_scaled = scaler_los.transform(x_df)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

        with torch.no_grad():
            pred = model_los(x_tensor).squeeze().item()

        st.success(f"Predicted length of stay: **{pred:.1f} days**")
# --- Sentiment tab ---
with tab_nlp:
    st.header("Patient Review Sentiment (TF-IDF + Logistic Regression)")

    vectorizer, clf = load_sentiment_ml_model()

    threshold = st.sidebar.slider(
        "Sentiment positive threshold (probability)",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
    )

    review_text = st.text_area(
        "Enter patient review text",
        height=150,
        placeholder="Type or paste a patient review here...",
    )

    if st.button("Analyze Sentiment"):
        if not review_text.strip():
            st.warning("Please enter some text.")
        else:
            X_vec = vectorizer.transform([review_text])
            prob_pos = clf.predict_proba(X_vec)[0, 1]
            label = "Positive" if prob_pos >= threshold else "Negative"
            st.write(
                f"Predicted sentiment: **{label}** (p_pos={prob_pos:.3f}, thr={threshold:.2f})"
            )

