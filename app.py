import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("loan_model.pkl")
trained_features = model.feature_names_in_

st.set_page_config(page_title="Loan Approval Predictor", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
        html, body {
            background-color: #f4e8ff;
            color: #222;
            font-family: 'Segoe UI', sans-serif;
        }
        .main-container {
            max-width: 1000px;
            margin: auto;
            padding: 20px;
        }
        .title {
            font-size: 42px;
            font-weight: 800;
            color: #5e548e;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            font-size: 18px;
            color: #6c63ac;
            text-align: center;
            margin-bottom: 40px;
        }
        .card {
            background-color: #ffffff;
            padding: 25px 30px;
            border-radius: 15px;
            box-shadow: 0px 3px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        .stButton button {
            background-color: #6c63ac;
            color: white;
            font-weight: bold;
            border: none;
            padding: 0.6rem 1.5rem;
            border-radius: 10px;
            font-size: 16px;
        }
        .stButton button:hover {
            background-color: #5a4f9c;
        }
        .result-approved {
            background-color: #e7fbe8;
            color: #2ecc71;
            padding: 18px;
            font-size: 22px;
            font-weight: 600;
            text-align: center;
            border-radius: 10px;
            margin-top: 25px;
        }
        .result-rejected {
            background-color: #fbeaea;
            color: #e74c3c;
            padding: 18px;
            font-size: 22px;
            font-weight: 600;
            text-align: center;
            border-radius: 10px;
            margin-top: 25px;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<div class="title">🏦 Loan Approval Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">📈 Compare your info with averages & predict loan approval</div>', unsafe_allow_html=True)

# --- Combined Card for Input ---
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Information")
    age = st.number_input("Age", min_value=18, max_value=100)
    gender = st.selectbox("Gender", ['Male', 'Female'])
    marital = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced', 'Widowed'])
    education = st.selectbox("Education Level", ['High School', 'Bachelor', 'Master', 'PhD', 'Other'])
    employment = st.selectbox("Employment Status", ['Employed', 'Self-employed', 'Unemployed'])

with col2:
    st.subheader("💰 Financial Information")
    income = st.number_input("Annual Income ($)", min_value=1000)
    loan_amount = st.number_input("Loan Amount Requested ($)", min_value=1000)
    purpose = st.selectbox("Purpose of Loan", ['Personal', 'Home', 'Car', 'Education'])
    existing_loans = st.slider("Existing Loans", 0, 10)
    late_payments = st.slider("Late Payments (Last Year)", 0, 10)
    credit_score = st.slider("Credit Score", 300, 850)

st.markdown('</div>', unsafe_allow_html=True)

# --- Feature Engineering ---
income_per_loan = income / loan_amount if loan_amount > 0 else 0
debt_burden = existing_loans + late_payments

input_dict = {
    'Age': age, 'AnnualIncome': income, 'LoanAmountRequested': loan_amount,
    'CreditScore': credit_score, 'LatePaymentsLastYear': late_payments,
    'IncomePerLoan': income_per_loan, 'DebtBurden': debt_burden,
    'Gender_Male': 1 if gender == 'Male' else 0,
    'MaritalStatus_Married': 1 if marital == 'Married' else 0,
    'MaritalStatus_Divorced': 1 if marital == 'Divorced' else 0,
    'MaritalStatus_Widowed': 1 if marital == 'Widowed' else 0,
    'EducationLevel_Bachelor': 1 if education == 'Bachelor' else 0,
    'EducationLevel_Master': 1 if education == 'Master' else 0,
    'EducationLevel_PhD': 1 if education == 'PhD' else 0,
    'EducationLevel_Other': 1 if education == 'Other' else 0,
    'EmploymentStatus_Self-employed': 1 if employment == 'Self-employed' else 0,
    'EmploymentStatus_Unemployed': 1 if employment == 'Unemployed' else 0,
    'PurposeOfLoan_Car': 1 if purpose == 'Car' else 0,
    'PurposeOfLoan_Education': 1 if purpose == 'Education' else 0,
    'PurposeOfLoan_Home': 1 if purpose == 'Home' else 0,
    'PurposeOfLoan_Personal': 1 if purpose == 'Personal' else 0,
}

# --- DataFrame for prediction ---
input_df = pd.DataFrame([input_dict])
for col in trained_features:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[trained_features]

# --- Predict ---
if st.button("🔍 Predict Loan Approval"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.markdown('<div class="result-approved">✅ Loan Approved</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-rejected">❌ Loan Not Approved</div>', unsafe_allow_html=True)

    # --- Chart Comparison ---
    avg_vals = [75000, 25000, 680, 4]  # Reference values
    user_vals = [income, loan_amount, credit_score, debt_burden]
    labels = ['Income', 'Loan Amount', 'Credit Score', 'Debt Burden']

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs = axs.flatten()

    for i in range(4):
        color = '#e74c3c' if labels[i] == 'Credit Score' and user_vals[i] < 600 else '#6c63ac'
        axs[i].bar(['You', 'Avg'], [user_vals[i], avg_vals[i]], color=[color, 'lightgray'])
        axs[i].set_title(labels[i])
        axs[i].grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    st.pyplot(fig)
