📊 Project Overview
A comprehensive machine learning solution for predicting customer churn using the Telco Customer Churn dataset. This project implements various techniques to handle class imbalance and optimize prediction performance for identifying customers at risk of leaving.

🎯 Business Problem
Customer churn (customer attrition) is a critical metric for businesses with subscription-based models. Predicting which customers are likely to churn allows companies to take proactive measures to retain them, which is significantly more cost-effective than acquiring new customers.

📁 Dataset
Source: Telco Customer Churn Dataset from Kaggle

Records: 7,043 customers

Features: 21 attributes including demographic info, services subscribed, account information

Target: Binary churn prediction (Yes/No)

⚙️ Technical Implementation
Data Preprocessing & Feature Engineering
Handled missing values in TotalCharges

Created innovative features:

CostPerTenure: Value per month of service

StabilityScore: Combined contract and payment method score

HighSpenderNew: Flag for new customers with high spending

TenureGroup: Binned tenure categories

Handling Class Imbalance
Implemented SMOTE (Synthetic Minority Over-sampling Technique) with proper pipeline integration

Alternative class weighting approach using scale_pos_weight

Threshold optimization to find optimal decision boundary instead of default 0.5

Modeling
XGBoost classifier with hyperparameter tuning

Comprehensive evaluation using multiple metrics:

Precision, Recall, F1-Score

ROC-AUC curves

Precision-Recall curves

Confusion matrices

📈 Key Results
The optimized model achieves:

Improved recall for churn class (identifying more true positives)

Better balance between precision and recall

Custom threshold based on business requirements

Comprehensive evaluation with multiple visualization techniques

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


🛠️ Technologies Used
Python: Primary programming language

Pandas & NumPy: Data manipulation and analysis

Scikit-learn: Machine learning algorithms and metrics

XGBoost: Gradient boosting framework

Imbalanced-learn: Handling class imbalance (SMOTE)

Matplotlib & Seaborn: Data visualization

Streamlit: Web application framework

Joblib: Model serialization

📊 Key Features
Interactive Web Interface: User-friendly Streamlit app for predictions

Multiple Approach Comparison: SMOTE vs. class weighting analysis

Threshold Optimization: Finds optimal decision boundary for business needs

Comprehensive Visualization: Multiple plots for model evaluation

Feature Importance Analysis: Identifies key factors driving churn

🔮 Future Enhancements
Deployment to cloud platform (AWS, GCP, Azure)

Real-time prediction API

Automated retraining pipeline

A/B testing framework for different models

Customer segmentation integration
