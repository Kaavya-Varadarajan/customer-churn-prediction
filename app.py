# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model, scaler, and column list
model = joblib.load('churn_prediction_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")
st.title("📞 Customer Churn Prediction Dashboard")
st.markdown("""
This app predicts the likelihood of a customer churning.
Adjust the inputs on the left and see the prediction!
""")

# Create a sidebar for input features
st.sidebar.header("Customer Details")

# Define numerical and categorical columns based on our feature engineering
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'CostPerTenure', 'StabilityScore', 'HighSpenderNew']

# Map shortened names back to the original feature names after one-hot encoding
def user_input_features():
    # Section 1: Basic Information
    st.sidebar.subheader("Basic Information")
    
    tenure = st.sidebar.slider('Tenure (months)', 0, 72, 24)
    MonthlyCharges = st.sidebar.slider('Monthly Charges ($)', 18.0, 120.0, 65.0)
    TotalCharges = st.sidebar.slider('Total Charges ($)', 0.0, 9000.0, 2000.0)
    
    # Calculate derived features
    CostPerTenure = TotalCharges / (tenure + 1)  # +1 to avoid division by zero
    
    # Section 2: Contract & Payment
    st.sidebar.subheader("Contract & Payment")
    
    contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    payment_method = st.sidebar.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    
    # Calculate stability score
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    payment_map = {
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 2
    }
    ContractScore = contract_map[contract]
    PaymentScore = payment_map[payment_method]
    StabilityScore = ContractScore + PaymentScore
    
    # Section 3: Services
    st.sidebar.subheader("Services")
    
    internet_service = st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    phone_service = st.sidebar.selectbox('Phone Service', ('No', 'Yes'))
    multiple_lines = st.sidebar.selectbox('Multiple Lines', ('No', 'No phone service', 'Yes'))
    online_security = st.sidebar.selectbox('Online Security', ('No', 'No internet service', 'Yes'))
    online_backup = st.sidebar.selectbox('Online Backup', ('No', 'No internet service', 'Yes'))
    device_protection = st.sidebar.selectbox('Device Protection', ('No', 'No internet service', 'Yes'))
    tech_support = st.sidebar.selectbox('Tech Support', ('No', 'No internet service', 'Yes'))
    streaming_tv = st.sidebar.selectbox('Streaming TV', ('No', 'No internet service', 'Yes'))
    streaming_movies = st.sidebar.selectbox('Streaming Movies', ('No', 'No internet service', 'Yes'))
    
    # Section 4: Customer Profile
    st.sidebar.subheader("Customer Profile")
    
    gender = st.sidebar.selectbox('Gender', ('Female', 'Male'))
    senior_citizen = st.sidebar.selectbox('Senior Citizen', ('No', 'Yes'))
    partner = st.sidebar.selectbox('Partner', ('No', 'Yes'))
    dependents = st.sidebar.selectbox('Dependents', ('No', 'Yes'))
    paperless_billing = st.sidebar.selectbox('Paperless Billing', ('No', 'Yes'))
    
    # Calculate HighSpenderNew feature
    median_monthly_charge = 70.35  # This should match your training data median
    HighSpenderNew = 1 if (MonthlyCharges > median_monthly_charge) and (tenure < 12) else 0
    
    # Build a dictionary of all features
    data = {
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'CostPerTenure': CostPerTenure,
        'StabilityScore': StabilityScore,
        'HighSpenderNew': HighSpenderNew,
        'gender_Male': 1 if gender == 'Male' else 0,
        'SeniorCitizen_1': 1 if senior_citizen == 'Yes' else 0,
        'Partner_Yes': 1 if partner == 'Yes' else 0,
        'Dependents_Yes': 1 if dependents == 'Yes' else 0,
        'PhoneService_Yes': 1 if phone_service == 'Yes' else 0,
        'MultipleLines_No phone service': 1 if multiple_lines == 'No phone service' else 0,
        'MultipleLines_Yes': 1 if multiple_lines == 'Yes' else 0,
        'InternetService_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
        'InternetService_No': 1 if internet_service == 'No' else 0,
        'OnlineSecurity_No internet service': 1 if online_security == 'No internet service' else 0,
        'OnlineSecurity_Yes': 1 if online_security == 'Yes' else 0,
        'OnlineBackup_No internet service': 1 if online_backup == 'No internet service' else 0,
        'OnlineBackup_Yes': 1 if online_backup == 'Yes' else 0,
        'DeviceProtection_No internet service': 1 if device_protection == 'No internet service' else 0,
        'DeviceProtection_Yes': 1 if device_protection == 'Yes' else 0,
        'TechSupport_No internet service': 1 if tech_support == 'No internet service' else 0,
        'TechSupport_Yes': 1 if tech_support == 'Yes' else 0,
        'StreamingTV_No internet service': 1 if streaming_tv == 'No internet service' else 0,
        'StreamingTV_Yes': 1 if streaming_tv == 'Yes' else 0,
        'StreamingMovies_No internet service': 1 if streaming_movies == 'No internet service' else 0,
        'StreamingMovies_Yes': 1 if streaming_movies == 'Yes' else 0,
        'Contract_One year': 1 if contract == 'One year' else 0,
        'Contract_Two year': 1 if contract == 'Two year' else 0,
        'PaperlessBilling_Yes': 1 if paperless_billing == 'Yes' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == 'Credit card (automatic)' else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == 'Mailed check' else 0,
    }
    
    # For all other one-hot encoded columns not shown, set them to 0.
    features = pd.DataFrame(data, index=[0])
    
    # Ensure we have all columns the model expects, in the right order
    for col in model_columns:
        if col not in features.columns:
            features[col] = 0
            
    features = features[model_columns]  # Reorder columns to match training

    # Scale the numerical features
    features[numerical_cols] = scaler.transform(features[numerical_cols])
    
    return features

input_df = user_input_features()

# Display user inputs
st.subheader('Selected Customer Parameters')

# Create a more user-friendly display of inputs
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Basic Information**")
    st.write(f"Tenure: {input_df['tenure'].iloc[0] * scaler.scale_[0] + scaler.mean_[0]:.0f} months")
    st.write(f"Monthly Charges: ${input_df['MonthlyCharges'].iloc[0] * scaler.scale_[1] + scaler.mean_[1]:.2f}")
    st.write(f"Total Charges: ${input_df['TotalCharges'].iloc[0] * scaler.scale_[2] + scaler.mean_[2]:.2f}")
    
with col2:
    st.markdown("**Customer Profile**")
    st.write(f"Stability Score: {input_df['StabilityScore'].iloc[0] * scaler.scale_[4] + scaler.mean_[4]:.0f}/4")
    st.write(f"High Spender New: {'Yes' if input_df['HighSpenderNew'].iloc[0] > 0 else 'No'}")
    st.write(f"Cost per Tenure: ${input_df['CostPerTenure'].iloc[0] * scaler.scale_[3] + scaler.mean_[3]:.2f}")

# Predict
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

churn_risk = prediction_proba[0][1]  # Probability of class 1 (Churn)

st.subheader('Prediction')
if prediction[0] == 1:
    st.error(f'🚨 High Churn Risk: {churn_risk*100:.2f}%')
else:
    st.success(f'✅ Low Churn Risk: {(1 - churn_risk)*100:.2f}%')

# Add a probability gauge
st.subheader('Churn Probability Gauge')
st.progress(float(churn_risk))
st.write(f"{churn_risk*100:.2f}% probability of churning.")

# Add some explanation
st.subheader('Key Factors')
st.markdown("""
Factors that often contribute to high churn:
*   **Short Tenure:** New customers are more likely to churn.
*   **Month-to-Month Contract:** Less commitment.
*   **High Monthly Charges:** Customers may seek cheaper alternatives.
*   **Electronic Check Payment:** Might indicate a less automated process.
*   **Low Stability Score:** Combination of short contract and less automatic payment methods.
*   **High Spender New:** New customers with high monthly charges are at risk.
""")

# Show feature importance (if available)
try:
    if hasattr(model, 'feature_importances_'):
        st.subheader('Top Influencing Factors')
        feature_importance = pd.DataFrame({
            'feature': model_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        st.bar_chart(feature_importance.set_index('feature'))
except:
    pass

# Add footer
st.markdown("---")
st.markdown("*Built with ❤️ using Streamlit and Scikit-Learn*")