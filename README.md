# Customer Churn Prediction App

A Streamlit web application that predicts customer churn probability using machine learning.

## Features

- Interactive user interface with sliders and dropdowns
- Real-time churn probability prediction
- Feature importance visualization
- Responsive design

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction

Usage
Adjust the customer parameters in the sidebar

View the predicted churn probability in the main panel

See which factors are most influencing the prediction

Model Details
The application uses an XGBoost classifier trained on the Telco Customer Churn dataset with the following features:

Tenure length

Monthly charges

Total charges

Contract type

Payment method

Internet service type

Additional services

File Structure
text
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── churn_prediction_model.pkl  # Trained model (not in repo)
├── scaler.pkl          # Feature scaler (not in repo)
├── model_columns.pkl   # Model columns (not in repo)
└── .gitignore          # Files to exclude from Git
