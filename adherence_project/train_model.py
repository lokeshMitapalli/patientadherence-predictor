
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

xlsx_path = r"C:\Users\DELL\OneDrive\Desktop\Final Prepared Dataset - Diabetes and Hypertension Data.xlsx"
csv_path = r"C:\Users\DELL\Downloads\patient_adherence_dataset.csv"

df1 = pd.read_excel(xlsx_path)
df2 = pd.read_csv(csv_path)

if list(df1.columns) == list(df2.columns):
    df = pd.concat([df1, df2], ignore_index=True)
else:
    df = df1

df.dropna(thresh=len(df.columns) - 2, inplace=True)
df.fillna({'Gender': 'Unknown', 'Medical_History': 'None', 'App_Usage': 'No'}, inplace=True)
df.fillna(0, inplace=True)

categorical_cols = ['Gender', 'Medical_History', 'Medication_Type', 'App_Usage']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop(columns=['Adherence', 'Patient_ID', 'Last_Visit_Date'], errors='ignore')
y = df['Adherence'].map({'Adherent': 1, 'Non-Adherent': 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

best_model = None
best_acc = 0

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_model = model

joblib.dump(best_model, "model.pkl")
print("Best model saved as model.pkl")
