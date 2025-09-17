import streamlit as st
import pandas as pd
import joblib

st.title("📊 Patient Adherence Prediction Dashboard")

# Load pipeline model
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# -------------------------
# Single Prediction Section
# -------------------------
st.header("🔹 Single Patient Prediction")

gender = st.selectbox("Gender", ["Male", "Female", "Unknown"])
age = st.number_input("Age", min_value=0, max_value=120)
medication_type = st.selectbox("Medication Type", ["Type1", "Type2"])
missed_doses = st.number_input("Missed Doses", min_value=0)
last_visit_gap = st.number_input("Days since Last Visit", min_value=0)
app_usage = st.selectbox("App Usage", ["Yes", "No"])

input_df = pd.DataFrame([{
    "Gender": gender,
    "Age": age,
    "Medication_Type": medication_type,
    "Missed_Doses": missed_doses,
    "Last_Visit_Gap": last_visit_gap,
    "App_Usage": app_usage
}])

if st.button("Predict Single Patient"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    st.write("Prediction probabilities:", proba)
    st.success("✅ Adherent" if prediction == 1 else "❌ Non-Adherent")

# -------------------------
# Batch Prediction Section
# -------------------------
st.header("📂 Batch Prediction from CSV")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Dataset Preview")
    st.dataframe(data.head())

    if st.button("Run Batch Prediction"):
        preds = model.predict(data)
        probs = model.predict_proba(data)[:, 1]

        data["Predicted_Adherence"] = ["Adherent" if p == 1 else "Non-Adherent" for p in preds]
        data["Adherence_Probability"] = probs

        st.write("### Dataset with Predictions")
        st.dataframe(data)

        st.write("### Adherence Count")
        st.bar_chart(data["Predicted_Adherence"].value_counts())

        st.download_button(
            label="⬇ Download Predictions as CSV",
            data=data.to_csv(index=False).encode("utf-8"),
            file_name="patient_predictions.csv",
            mime="text/csv"
        )


