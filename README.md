# Customer Churn Prediction Analysis

![Python](https://img.shields.io/badge/Python-3.8%252B-blue)  
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)  
![Web](https://img.shields.io/badge/Web-Streamlit-red)  
![Model](https://img.shields.io/badge/Model-XGBoost-green)  

A comprehensive machine learning solution for predicting customer churn using the **Telco Customer Churn dataset**.  
This project implements advanced techniques to handle class imbalance and optimize prediction performance for identifying at-risk customers.

---

## Project Overview
Customer churn prediction is critical for subscription-based businesses.  
This project helps identify customers likely to cancel their service, enabling **proactive retention strategies** that are more cost-effective than acquiring new customers.

---

## Key Features
- **Advanced Feature Engineering**: Innovative features such as `StabilityScore` and `CostPerTenure`  
- **Class Imbalance Handling**: Implemented **SMOTE** and class weighting techniques  
- **Threshold Optimization**: Custom decision boundary instead of the default 0.5  
- **Interactive Web App**: **Streamlit-based** interface for real-time predictions  
- **Comprehensive Evaluation**: Multiple metrics and visualizations for model assessment  

---

## Results
- Enhanced churn detection with optimized threshold  
- Better balance between **precision** and **recall**  
- Rich evaluation using multiple visualization techniques  

---

##  Quick Start

### Prerequisites
- Python **3.8+**  
- `pip` package manager  

### Installation
Clone the repository:
```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction

📁 Project Structure
customer-churn-prediction/
├── app.py                        # Streamlit web application
├── train_model.py                 # Model training and evaluation
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
├── .gitignore                     # Files to exclude from Git
├── churn_prediction_model.pkl     # Trained model (generated)
├── scaler.pkl                     # Feature scaler (generated)
├── model_columns.pkl              # Model columns (generated)
├── optimal_threshold.pkl          # Optimal threshold (generated)
└── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset (from Kaggle)
