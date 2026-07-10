import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap

from src.config import PIPELINE_PATH

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide"
)

# ── Load model and explainer (cached so they load once) ────────────────────────
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load(PIPELINE_PATH)
    explainer = joblib.load(
        os.path.join(os.path.dirname(str(PIPELINE_PATH)), "shap_explainer.joblib")
    )
    return pipeline, explainer

pipeline, explainer = load_artifacts()
preprocessor = pipeline.named_steps["preprocessor"]
feature_names = list(preprocessor.get_feature_names_out())

# ── Helpers ────────────────────────────────────────────────────────────────────
def risk_color(prob):
    if prob < 0.4:
        return "🟢", "green", "Low Risk"
    elif prob < 0.7:
        return "🟠", "orange", "Medium Risk"
    else:
        return "🔴", "red", "High Risk"


def build_input_df(inputs: dict) -> pd.DataFrame:
    """Turn form values into a single-row DataFrame matching training columns."""
    row = {
        "tenure":             inputs["tenure"],
        "MonthlyCharges":     inputs["MonthlyCharges"],
        "TotalCharges":       inputs["tenure"] * inputs["MonthlyCharges"],
        "avg_monthly_spend":  inputs["MonthlyCharges"],
        "gender":             inputs["gender"],
        "SeniorCitizen":      "1" if inputs["SeniorCitizen"] else "0",
        "Partner":            inputs["Partner"],
        "Dependents":         inputs["Dependents"],
        "PhoneService":       inputs["PhoneService"],
        "MultipleLines":      inputs["MultipleLines"],
        "InternetService":    inputs["InternetService"],
        "OnlineSecurity":     inputs["OnlineSecurity"],
        "OnlineBackup":       inputs["OnlineBackup"],
        "DeviceProtection":   inputs["DeviceProtection"],
        "TechSupport":        inputs["TechSupport"],
        "StreamingTV":        inputs["StreamingTV"],
        "StreamingMovies":    inputs["StreamingMovies"],
        "Contract":           inputs["Contract"],
        "PaperlessBilling":   inputs["PaperlessBilling"],
        "PaymentMethod":      inputs["PaymentMethod"],
    }
    return pd.DataFrame([row])


# ── Layout ─────────────────────────────────────────────────────────────────────
st.title("📉 Customer Churn Predictor")
st.caption("Enter a customer profile to get a real-time churn risk score with SHAP explanation.")

tab1, tab2 = st.tabs(["🔍 Predict Customer Risk", "📊 Global Feature Importance"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        st.subheader("Customer Profile")

        tenure          = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 65)

        st.divider()

        gender          = st.selectbox("Gender", ["Male", "Female"])
        senior          = st.checkbox("Senior Citizen")
        partner         = st.selectbox("Partner", ["Yes", "No"])
        dependents      = st.selectbox("Dependents", ["No", "Yes"])

        st.divider()

        phone_service   = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines  = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet        = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        online_sec      = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup   = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        device_prot     = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support    = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv    = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_mov   = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        st.divider()

        contract        = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment         = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

        predict_btn = st.button("⚡ Predict Churn Risk", type="primary", use_container_width=True)

    # ── Results column ─────────────────────────────────────────────────────────
    with col_result:
        st.subheader("Prediction")

        if predict_btn:
            inputs = {
                "tenure": tenure, "MonthlyCharges": monthly_charges,
                "gender": gender, "SeniorCitizen": senior,
                "Partner": partner, "Dependents": dependents,
                "PhoneService": phone_service, "MultipleLines": multiple_lines,
                "InternetService": internet, "OnlineSecurity": online_sec,
                "OnlineBackup": online_backup, "DeviceProtection": device_prot,
                "TechSupport": tech_support, "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_mov, "Contract": contract,
                "PaperlessBilling": paperless, "PaymentMethod": payment,
            }

            input_df = build_input_df(inputs)
            prob = pipeline.predict_proba(input_df)[0, 1]
            icon, color, label = risk_color(prob)

            # Risk score display
            st.markdown(f"""
            <div style="
                background: {'#ffeaea' if color=='red' else '#fff8e7' if color=='orange' else '#eafff2'};
                border-left: 6px solid {color};
                border-radius: 8px;
                padding: 24px;
                margin-bottom: 16px;
            ">
                <h1 style="margin:0; color:{color};">{icon} {prob*100:.1f}%</h1>
                <h3 style="margin:4px 0 0 0; color:{color};">{label}</h3>
                <p style="margin:8px 0 0 0; color:#555;">
                    Churn probability for this customer profile.
                </p>
            </div>
            """, unsafe_allow_html=True)

            # SHAP waterfall
            st.markdown("**Why this score?**")
            X_transformed = preprocessor.transform(input_df)
            shap_vals = explainer.shap_values(X_transformed)

            explanation = shap.Explanation(
                values=shap_vals[0],
                base_values=explainer.expected_value,
                data=X_transformed[0],
                feature_names=feature_names
            )

            fig, ax = plt.subplots(figsize=(8, 5))
            shap.plots.waterfall(explanation, max_display=12, show=False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        else:
            st.info("👈 Fill in the customer profile and click **Predict Churn Risk**")
           st.image("docs/shap_waterfall.png", caption="Example SHAP explanation", use_column_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Global Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Global Feature Importance")
    st.caption("Based on mean absolute SHAP values across the test set.")

    col_bar, col_bee = st.columns(2)
    with col_bar:
        st.image("docs/shap_summary_bar.png", caption="Mean |SHAP| bar chart", use_container_width=True)
    with col_bee:
        st.image("docs/shap_beeswarm.png", caption="Beeswarm — direction of impact", use_container_width=True)

    st.divider()
    st.subheader("ROC Curves — Model Comparison")
    st.image("docs/roc_curves.png", use_container_width=False, width=600)
