# Customer Churn Prediction Pipeline

An end-to-end supervised machine-learning project that predicts whether a telecom customer is likely to leave the service.

## Features

- Data cleaning and missing-value handling
- One-hot encoding of categorical features
- Numerical feature scaling
- Four engineered features:
  - Average monthly spend
  - Tenure group
  - Number of subscribed services
  - Charges per service
- Logistic Regression, Random Forest and Gradient Boosting
- GridSearchCV with 5-fold cross-validation
- ROC-AUC, PR-AUC, precision, recall, F1, accuracy and confusion matrix
- FastAPI REST API with probability and binary prediction
- Interactive Swagger documentation

## Dataset

Download the IBM Telco Customer Churn CSV from:
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in `data/`.

## Setup (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Train and evaluate

Run from the project root:

```powershell
python -m src.train_model
python -m src.evaluate_model
```

The trained model and evaluation report are saved in `models/`.

## Start the API

```powershell
uvicorn app:app --reload
```

Open:
- API home: http://127.0.0.1:8000/
- Swagger UI: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

## Prediction endpoint

`POST /predict`

Example JSON:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 75.5,
  "TotalCharges": 906.0
}
```

The response contains churn probability, binary prediction, risk level, threshold and selected model.

## Important modelling note

The training script uses a stratified 80/20 train/test split and performs 5-fold cross-validation on the training portion. The model is selected using training cross-validation ROC-AUC; the held-out test set is reserved for reporting performance.

The default 0.50 threshold is a starting point, not a universal business optimum. A telecom business should tune it using retention-offer costs, customer lifetime value and the cost of missed churners.
