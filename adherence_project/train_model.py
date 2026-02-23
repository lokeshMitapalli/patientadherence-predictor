import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

xlsx_path = r"C:\Users\DELL\OneDrive\Desktop\Final Prepared Dataset - Diabetes and Hypertension Data.xlsx"
csv_path = r"C:\Users\DELL\Downloads\patient_adherence_dataset.csv"

df1 = pd.read_excel(xlsx_path)
df2 = pd.read_csv(csv_path)

if list(df1.columns) == list(df2.columns):
    df = pd.concat([df1, df2], ignore_index=True)
else:
    df = df1.copy()

df.dropna(thresh=len(df.columns)-2, inplace=True)

df.fillna({
    'Gender': 'Unknown',
    'Medical_History': 'None',
    'Medication_Type': 'None',
    'App_Usage': 'No'
}, inplace=True)

df.fillna(0, inplace=True)

categorical_cols = ['Gender','Medical_History','Medication_Type','App_Usage']

encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

joblib.dump(encoders, "encoders.pkl")

y = df['Adherence'].map({'Adherent':0,'Non-Adherent':1})

X = df.drop(columns=['Adherence','Patient_ID','Last_Visit_Date'], errors='ignore')

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

joblib.dump(scaler,"scaler.pkl")

X_train,X_test,y_train,y_test = train_test_split(
    X_scaled,y,test_size=0.2,random_state=42
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200,random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

best_model=None
best_acc=0

for name,model in models.items():

    model.fit(X_train,y_train)

    preds=model.predict(X_test)

    acc=accuracy_score(y_test,preds)

    print(name,"Accuracy:",acc)

    if acc>best_acc:
        best_acc=acc
        best_model=model

joblib.dump(best_model,"model.pkl")

joblib.dump(list(X.columns),"features.pkl")

print("Best model:",best_model)
print("Accuracy:",best_acc)
print("Saved: model.pkl, encoders.pkl, scaler.pkl, features.pkl")
