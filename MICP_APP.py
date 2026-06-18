import streamlit as st
import pandas as pd
import joblib

# Load model + columns
model = joblib.load("motor_insurance_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("🚗 INSURANCE CLAIM PREDICTOR")

# -------- INPUT --------

age = st.number_input("Age", 18, 70, value=40)

gender = st.selectbox("Gender", ["Male", "Female"], index=1)

region = st.selectbox("Region", ["Rural", "Semi-Urban", "Urban"], index=1)

income = st.number_input("Annual Income", value=800000)

experience = st.number_input("Driving Experience (Years)", value=10)

vehicle_age = st.number_input("Vehicle Age", value=2)

vehicle_type = st.selectbox("Vehicle Type", ["Sedan", "SUV", "Truck", "Hatchback"], index=0)

engine = st.number_input("Engine Capacity", value=1200)

vehicle_value = st.number_input("Vehicle Value", value=600000)

policy_type = st.selectbox("Policy Type", ["Third Party", "Comprehensive"], index=1)

tenure = st.number_input("Policy Tenure", value=3)

premium = st.number_input("Premium Amount", value=10000)

deductible = st.number_input("Deductible", value=1000)

bonus = st.number_input("No Claim Bonus", value=40)

accidents = st.number_input("Previous Accidents", value=0)

violations = st.number_input("Traffic Violations", value=0)

credit = st.number_input("Credit Score", value=750)

# -------- DATAFRAME --------

input_df = pd.DataFrame([{
    "Age": age,
    "Gender": 1 if gender == "Male" else 0,
    "Annual_Income": income,
    "Driving_Experience_Years": experience,
    "Vehicle_Age": vehicle_age,
    "Engine_Capacity": engine,
    "Vehicle_Value": vehicle_value,
    "Policy_Tenure": tenure,
    "Premium_Amount": premium,
    "Deductible": deductible,
    "No_Claim_Bonus": bonus,
    "Previous_Accidents": accidents,
    "Traffic_Violations": violations,
    "Credit_Score": credit,
    "Region": region,
    "Vehicle_Type": vehicle_type,
    "Policy_Type": policy_type
}])

# -------- ENCODING --------

input_df = pd.get_dummies(input_df, columns=["Region","Vehicle_Type","Policy_Type"], drop_first=True)
input_df = input_df.reindex(columns=model_columns, fill_value=0)

# -------- PREDICTION --------

prob = model.predict_proba(input_df)[0][1]

st.subheader("📊 Result")

if prob > 0.5:
    st.error(f"⚠️ Claim (Probability: {prob:.2f})")
else:
    st.success(f"✅ No Claim (Probability: {1 - prob:.2f})")