```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os


st.set_page_config(
    page_title="Patient Adherence AI Dashboard",
    page_icon="🏥",
    layout="wide"
)


st.markdown("""
<style>
.metric-card {
    background-color: #111;
    padding: 20px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_components():

    model = joblib.load("model.pkl")

    encoders = joblib.load("encoders.pkl")

    scaler = joblib.load("scaler.pkl")

    features = joblib.load("features.pkl")

    return model, encoders, scaler, features


model, encoders, scaler, features = load_components()


def encode(df):

    for col in encoders:

        if col in df.columns:

            df[col] = df[col].astype(str)

            df[col] = df[col].apply(
                lambda x: encoders[col].transform([x])[0]
                if x in encoders[col].classes_
                else 0
            )

    return df


def preprocess(df):

    df = encode(df)

    for col in features:

        if col not in df.columns:

            df[col] = 0

    df = df[features]

    scaled = scaler.transform(df)

    return scaled


st.title("🏥 Patient Medication Adherence AI Dashboard")


st.sidebar.title("Control Panel")

file = st.sidebar.file_uploader("Upload Patient Dataset", type=["csv"])


if file:

    data = pd.read_csv(file)

else:

    data = pd.read_csv("patient_adherence_dataset.csv")


X = preprocess(data)

pred = model.predict(X)

prob = model.predict_proba(X)[:,1]


data["Prediction"] = pred

data["Risk"] = prob

data["Prediction"] = data["Prediction"].map({
    0:"Adherent",
    1:"Non-Adherent"
})


total = len(data)

adherent = len(data[data["Prediction"]=="Adherent"])

non_adherent = len(data[data["Prediction"]=="Non-Adherent"])

avg_risk = data["Risk"].mean()


col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Patients", total)

col2.metric("Adherent", adherent)

col3.metric("Non-Adherent", non_adherent)

col4.metric("Average Risk", f"{avg_risk:.2f}")


st.divider()


col1, col2 = st.columns(2)


fig1 = px.pie(
    data,
    names="Prediction",
    title="Adherence Distribution",
    color="Prediction"
)

col1.plotly_chart(fig1, use_container_width=True)


fig2 = px.histogram(
    data,
    x="Risk",
    nbins=20,
    title="Risk Distribution"
)

col2.plotly_chart(fig2, use_container_width=True)


st.divider()


st.subheader("High Risk Patients")


threshold = st.slider(
    "Risk Threshold",
    0.0,
    1.0,
    0.7
)


high_risk = data[data["Risk"] >= threshold]


st.dataframe(high_risk, use_container_width=True)


st.subheader("All Patients")


st.dataframe(data, use_container_width=True)


st.subheader("Single Patient Prediction")


input_data = {}

cols = st.columns(3)

for i, col in enumerate(features):

    input_data[col] = cols[i%3].text_input(col)


if st.button("Predict Patient Risk"):

    df = pd.DataFrame([input_data])

    X = preprocess(df)

    pred = model.predict(X)[0]

    prob = model.predict_proba(X)[0][1]

    if pred == 1:

        st.error(f"Non-Adherent Risk: {prob:.2f}")

    else:

        st.success(f"Adherent Risk: {prob:.2f}")


st.subheader("Download Predictions")


csv = data.to_csv(index=False)

st.download_button(
    "Download CSV",
    csv,
    "predictions.csv",
    "text/csv"
)
```






















