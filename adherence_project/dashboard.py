import streamlit as st
import pandas as pd
import os
from joblib import load
import pickle
from io import BytesIO
import smtplib
from email.mime.text import MIMEText
from twilio.rest import Client
from datetime import datetime
import re

st.set_page_config(page_title="Patient Adherence Dashboard", layout="wide")
st.title("Patient Adherence Prediction Dashboard")

def show_toast(message, color="green"):
    toast_html = f"""
    <div style="
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: {color};
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        font-size: 16px;
        z-index: 9999;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    ">{message}</div>
    <script>
        setTimeout(function(){{
            var toasts = document.querySelectorAll('[style*="position: fixed; bottom: 20px;"]');
            toasts.forEach(function(toast){{ toast.style.display = 'none'; }});
        }}, 3000);
    </script>
    """
    st.markdown(toast_html, unsafe_allow_html=True)

def encode_dataframe(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes
    return df


sender_email = st.secrets["gmail"]["email"]
app_password = st.secrets["gmail"]["app_password"]

twilio_sid = st.secrets["twilio"]["account_sid"]
twilio_token = st.secrets["twilio"]["auth_token"]
twilio_from = st.secrets["twilio"]["from_number"]
twilio_client = Client(twilio_sid, twilio_token)


def send_email(patient_id, probability, recipient_email):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = MIMEText(
        f"🚨 Hospital Alert\n\n"
        f"Patient ID: {patient_id}\n"
        f"Status: NON-ADHERENT\n"
        f"Estimated Risk: {probability*100:.1f}%\n"
        f"Timestamp: {current_time}\n\n"
        f"Please schedule a follow-up consultation today.\n\n"
        f"— Automated Patient Monitoring System"
    )
    msg["Subject"] = "🚨 Hospital Alert: Non-Adherence Detected"
    msg["From"] = sender_email
    msg["To"] = recipient_email
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender_email, app_password)
            server.send_message(msg)
        return True
    except Exception as e:
        return f"Email error for Patient {patient_id}: {e}"

def send_sms(patient_id, probability, recipient_number):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        twilio_client.messages.create(
            body=(f"🚨 Hospital Alert: Patient {patient_id} shows high non-adherence risk "
                  f"({probability*100:.1f}%).\n"
                  f"Time: {current_time}\n"
                  f"Please schedule a follow-up consultation today.\n"
                  f"— Automated Patient Monitoring System"),
            from_=twilio_from,
            to=recipient_number
        )
        return True
    except Exception as e:
        return f"SMS error for Patient {patient_id}: {e}"


model = None
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')

if os.path.exists(model_path):
    try:
        model = load(model_path)
        show_toast("✅ Model loaded successfully!", color="green")
    except:
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
                show_toast("✅ Model loaded successfully!", color="green")
        except:
            model = None

if model is None:
    uploaded_model = st.file_uploader("Upload your model file (.pkl)", type=["pkl"])
    if uploaded_model:
        try:
            model = load(uploaded_model)
            show_toast("✅ Model uploaded successfully!", color="green")
        except:
            try:
                model = pickle.load(uploaded_model)
                show_toast("✅ Model uploaded successfully!", color="green")
            except:
                model = None

if model is None:
    st.stop()


st.sidebar.header("Upload Dataset (CSV)")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    show_toast("✅ Dataset uploaded successfully!", color="green")
    st.dataframe(data.head())
else:
    default_path = os.path.join(os.path.dirname(__file__), 'patient_adherence_dataset.csv')
    if os.path.exists(default_path):
        data = pd.read_csv(default_path)
        st.dataframe(data.head())
    else:
        st.stop()

def clean_phone(number):
    """
    Cleans and formats Indian phone numbers to +91XXXXXXXXXX format.
    Accepts numbers like 9876543210, 09876543210, 919876543210, +919876543210.
    """
    if pd.isna(number):
        return None

    number = str(number).strip()
    number = re.sub(r'\D', '', number)  # remove non-digits

    # Remove leading 0 if present
    if number.startswith("0") and len(number) == 11:
        number = number[1:]

    # Handle cases
    if len(number) == 10:
        return f"+91{number}"
    elif len(number) == 12 and number.startswith("91"):
        return f"+{number}"
    elif len(number) == 13 and number.startswith("+91"):
        return number
    else:
        return None

def clean_email(email):
    if pd.isna(email):
        return None
    email = str(email).strip()
    pattern = r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$"
    return email.lower() if re.match(pattern, email) else None

if "Phone" in data.columns:
    data["Phone"] = data["Phone"].apply(clean_phone)
    st.info(f"📱 {data['Phone'].notna().sum()} valid phone numbers formatted for Twilio")

if "Email" in data.columns:
    data["Email"] = data["Email"].apply(clean_email)
    st.info(f"📧 {data['Email'].notna().sum()} valid email addresses cleaned")

if "Adherence" in data.columns:
    y = data["Adherence"].fillna("").apply(lambda x: 0 if str(x).strip().lower() == "adherent" else 1)
else:
    y = None
X = data.drop(columns=["Adherence"], errors='ignore')

st.sidebar.header("Single Prediction")
input_data = {}
for col in X.columns:
    if col not in ["Patient_ID", "Email", "Phone"]:
        value = st.sidebar.text_input(f"{col}")
        input_data[col] = value

if st.sidebar.button("Predict"):
    try:
        input_df = pd.DataFrame([input_data])
        input_df = encode_dataframe(input_df)
        for col in input_df.columns:
            try:
                input_df[col] = pd.to_numeric(input_df[col])
            except:
                pass
        trained_features = model.feature_names_in_
        for col in trained_features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[trained_features]
        prediction = model.predict(input_df)[0]
        result = "Adherent" if prediction == 0 else "Non-Adherent"
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0][1]
            st.write(f"Prediction probability (Non-Adherent): {proba*100:.1f}%")
        show_toast(f"✅ Single prediction: {result}", color="green")
        st.success(f"Prediction: {result}")
    except:
        st.error("Error during prediction")

st.subheader("Batch Prediction")
threshold = st.slider("Non-Adherence Probability Threshold", 0.5, 1.0, 0.7, 0.01)
recipient_email = st.text_input("Default Recipient Email (if patient email missing)")
recipient_number = st.text_input("Default Recipient Phone Number (if patient phone missing, e.g. +919876543210)")

if st.button("Run Batch Prediction"):
    try:
        X_copy = X.copy()
        X_copy = encode_dataframe(X_copy)
        trained_features = model.feature_names_in_
        for col in trained_features:
            if col not in X_copy.columns:
                X_copy[col] = 0
        X_copy = X_copy[trained_features]

        preds = model.predict(X_copy)
        probs = model.predict_proba(X_copy)[:, 1] if hasattr(model, "predict_proba") else [0]*len(X_copy)

        data["Predicted_Adherence"] = ["Adherent" if p == 0 else "Non-Adherent" for p in preds]
        data["Non_Adherence_Prob"] = probs
        st.dataframe(data)

        high_risk = data[data["Non_Adherence_Prob"] >= threshold]

        if not high_risk.empty:
            st.error(f"{len(high_risk)} HIGH-RISK NON-ADHERENT patients")
            st.dataframe(high_risk)

            st.info("📧📱 Sending alerts (Email + SMS)...")
            progress_bar = st.progress(0)
            total = len(high_risk)

            for i, row in enumerate(high_risk.itertuples(), start=1):
                patient_id = getattr(row, "Patient_ID", f"Patient-{i}")
                patient_email = getattr(row, "Email", None)
                patient_phone = getattr(row, "Phone", None)
                probability = getattr(row, "Non_Adherence_Prob", 0)

                target_email = patient_email if patient_email else recipient_email
                target_phone = patient_phone if patient_phone else recipient_number

                email_result = send_email(patient_id, probability, target_email) if target_email else "No email available"
                if email_result is True:
                    st.success(f"📧 Email sent for Patient {patient_id} → {target_email}")
                else:
                    st.error(f"Email failed for Patient {patient_id}: {email_result}")

                sms_result = send_sms(patient_id, probability, target_phone) if target_phone else "No phone available"
                if sms_result is True:
                    st.success(f"📱 SMS sent for Patient {patient_id} → {target_phone}")
                else:
                    st.error(f"SMS failed for Patient {patient_id}: {sms_result}")

                progress_bar.progress(i / total)

            st.success("✅ All alerts processed")
        else:
            st.success("All patients below risk threshold")

        buffer = BytesIO()
        data.to_csv(buffer, index=False)
        buffer.seek(0)
        st.download_button("Download Predictions as CSV", data=buffer,
                           file_name="patient_predictions.csv", mime="text/csv")

    except Exception as e:
        st.error(f"Error during batch prediction: {e}")





















