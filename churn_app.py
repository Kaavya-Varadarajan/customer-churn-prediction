import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model, columns, scaler, and threshold
model = joblib.load('churn_prediction_model.pkl')
model_columns = joblib.load('model_columns.pkl')
scaler = joblib.load('scaler.pkl')
try:
    threshold = joblib.load('optimal_threshold.pkl')
except:
    threshold = 0.5

# Define numerical and categorical columns (update as needed)
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'CostPerTenure', 'StabilityScore', 'HighSpenderNew']
categorical_cols = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'TenureGroup'
]

# Custom CSS for dark background and light text
st.markdown("""
    <style>
    body, .main, .stApp {
        background-color: #23272f !important;
        color: #f5f6fa !important;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
    }
    .st-bb, .st-cb {
        background-color: #23272f !important;
        color: #f5f6fa !important;
    }
    .stTextInput>div>div>input, .stSelectbox>div>div>div>input, .stNumberInput>div>div>input {
        background-color: #2c313a !important;
        color: #f5f6fa !important;
    }
    .stRadio>div>label, .stSelectbox>div>div>div, .stSlider>div>div>div {
        color: #f5f6fa !important;
    }
    .stProgress>div>div>div>div {
        background-color: #4CAF50 !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 Customer Churn Prediction")
st.markdown("""
<span style='color:#f5f6fa;'>This app predicts the likelihood of a customer churning based on their details. 
Fill in the information below and click <b>Predict Churn</b> to see the result.</span>
""", unsafe_allow_html=True)

with st.form("churn_form"):
    st.header("Customer Information")
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.number_input('Tenure (months)', min_value=0, max_value=100, value=12)
        monthly_charges = st.number_input('Monthly Charges', min_value=0.0, value=70.0)
        total_charges = st.number_input('Total Charges', min_value=0.0, value=1000.0)
        cost_per_tenure = st.number_input('Cost Per Tenure', min_value=0.0, value=80.0)
        stability_score = st.slider('Stability Score', min_value=0, max_value=10, value=2)
        high_spender_new = st.radio('High Spender & New', [0, 1], format_func=lambda x: 'Yes' if x else 'No')
        senior_citizen = st.radio('Senior Citizen', [0, 1], format_func=lambda x: 'Yes' if x else 'No')
        partner = st.radio('Partner', ['Yes', 'No'])
        dependents = st.radio('Dependents', ['Yes', 'No'])
    with col2:
        gender = st.radio('Gender', ['Female', 'Male'])
        phone_service = st.radio('Phone Service', ['Yes', 'No'])
        multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
        internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
        online_security = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
        online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
        device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
        tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
        streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
        streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
        contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
        paperless_billing = st.radio('Paperless Billing', ['Yes', 'No'])
        payment_method = st.selectbox('Payment Method', [
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
        tenure_group = st.selectbox('Tenure Group', [
            'New(0-1y)', 'Established(1-2y)', 'Senior(2-4y)', 'Veteran(4-6y)', 'Legend(6y+)'])

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    input_data = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'CostPerTenure': cost_per_tenure,
        'StabilityScore': stability_score,
        'HighSpenderNew': 1 if high_spender_new == 1 or high_spender_new == 'Yes' else 0,
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'TenureGroup': tenure_group
    }
    input_df = pd.DataFrame([input_data])
    # One-hot encode categorical variables to match training
    input_df = pd.get_dummies(input_df)
    # Add missing columns
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    # Reorder columns
    input_df = input_df[model_columns]
    # Scale numerical features
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    # Predict
    probability = model.predict_proba(input_df)[0][1]
    prediction = 'Yes' if probability >= threshold else 'No'
    st.markdown("---")
    st.subheader("Prediction Result")
    if prediction == 'Yes':
        st.error(f"Churn Prediction: {prediction}")
    else:
        st.success(f"Churn Prediction: {prediction}")
    st.info(f"Churn Probability: {probability:.2f} (Threshold: {threshold})")
    st.progress(int(probability * 100))
    st.markdown("---")
    st.markdown("<span style='color:#f5f6fa;'><b>Tip:</b> Try changing the values above to see how the prediction changes!</span>", unsafe_allow_html=True)