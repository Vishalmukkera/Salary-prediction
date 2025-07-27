import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Salary Prediction", layout="centered")
st.title("💼 Salary Prediction App")
st.write("Estimate an employee’s annual salary from their profile features.")


@st.cache_data
def load_data():
    return pd.read_csv("Salary_Data_Based_country_and_race.csv")

@st.cache_resource
def load_model():
    
    return joblib.load("SalaryModel.pkl")

df = load_data()
model = load_model()


cat_feats = ["Gender", "Country", "Education Level", "Job Title", "Race"]
cont_feats = ["Age", "Years of Experience"]

encoders = {
    feat: LabelEncoder().fit(df[feat].astype(str))
    for feat in cat_feats
}
scaler = StandardScaler().fit(df[cont_feats])


st.subheader("Candidate Profile")

gender    = st.selectbox("Gender",          list(encoders["Gender"].classes_))
country   = st.selectbox("Country",         list(encoders["Country"].classes_))
education = st.selectbox("Education Level", list(encoders["Education Level"].classes_))
job_title = st.selectbox("Job Title",       list(encoders["Job Title"].classes_))
race      = st.selectbox("Race",            list(encoders["Race"].classes_))

age = st.number_input(
    "Age",
    min_value=int(df["Age"].min()),
    max_value=int(df["Age"].max()),
    value=int(df["Age"].mean()),
)
experience = st.number_input(
    "Years of Experience",
    min_value=float(df["Years of Experience"].min()),
    max_value=float(df["Years of Experience"].max()),
    value=float(df["Years of Experience"].mean()),
)

salary_scaler = StandardScaler().fit(df[['Salary']])

if st.button("🔮 Predict Salary"):
    
    codes = [
        encoders["Gender"].transform([gender])[0],
        encoders["Country"].transform([country])[0],
        encoders["Education Level"].transform([education])[0],
        encoders["Job Title"].transform([job_title])[0],
        encoders["Race"].transform([race])[0],
    ]
    cont_scaled = scaler.transform([[age, experience]])[0]

    
    X_new = np.array([codes + list(cont_scaled)])

    
    scaled_pred = model.predict(X_new)[0]

    
    pred_actual = salary_scaler.inverse_transform([[scaled_pred]])[0][0]

    
    st.success(f"💰 Estimated Annual Salary: ${pred_actual:,.2f}")

else:
    st.info("Click above to estimate the salary.")

