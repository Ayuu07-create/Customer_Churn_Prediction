import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from src.feature_engineering import ChurnFeatureEngineer
from src.preprocessing import build_preprocessor

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
PREDICTION_THRESHOLD = 0.50


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place WA_Fn-UseC_-Telco-Customer-Churn.csv inside the data folder."
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    required = {"Churn", "customerID"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # TotalCharges is stored as text in the original IBM dataset; blank strings become NaN.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["Churn"] = df["Churn"].astype(str).str.strip().map({"Yes": 1, "No": 0})

    df = df.dropna(subset=["Churn"]).copy()
    y = df.pop("Churn").astype(int)
    X = df.drop(columns=["customerID"])

    # Ensure all expected model columns exist.
    expected = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges"
    ]
    for col in expected:
        if col not in X.columns:
            X[col] = np.nan
    X = X[expected]

    return X, y


def make_pipeline(model):
    return Pipeline(steps=[
        ("features", ChurnFeatureEngineer()),
        ("preprocessor", build_preprocessor()),
        ("classifier", model)
    ])


def main():
    X, y = load_data()
    print(f"Loaded {len(X):,} records and {X.shape[1]} input columns.")
    print("Target distribution:")
    print(y.value_counts().rename(index={0: "No churn", 1: "Churn"}))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced"),
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__solver": ["liblinear"]
            }
        ),
        "Random Forest": (
            RandomForestClassifier(
                random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
            ),
            {
                "classifier__n_estimators": [200, 400],
                "classifier__max_depth": [None, 10, 20],
                "classifier__min_samples_split": [2, 5]
            }
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__learning_rate": [0.03, 0.1],
                "classifier__max_depth": [2, 3]
            }
        )
    }

    results = {}
    best_name, best_search, best_cv_auc = None, None, -np.inf

    for name, (estimator, grid) in candidates.items():
        print(f"\nTuning {name}...")
        search = GridSearchCV(
            estimator=make_pipeline(estimator),
            param_grid=grid,
            scoring="roc_auc",
            cv=5,
            n_jobs=-1,
            refit=True,
            verbose=0
        )
        search.fit(X_train, y_train)

        probabilities = search.best_estimator_.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= PREDICTION_THRESHOLD).astype(int)

        metrics = {
            "cv_roc_auc": float(search.best_score_),
            "test_roc_auc": float(roc_auc_score(y_test, probabilities)),
            "pr_auc": float(average_precision_score(y_test, probabilities)),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            "best_params": search.best_params_
        }
        results[name] = metrics

        print(f"Best CV ROC-AUC: {metrics['cv_roc_auc']:.4f}")
        print(f"Test ROC-AUC:    {metrics['test_roc_auc']:.4f}")
        print(f"PR-AUC:          {metrics['pr_auc']:.4f}")
        print(f"Precision:       {metrics['precision']:.4f}")
        print(f"Recall:          {metrics['recall']:.4f}")
        print(f"F1-score:        {metrics['f1']:.4f}")

        # Select using cross-validation on training data only. The held-out
        # test set is used for reporting, not for model selection.
        if metrics["cv_roc_auc"] > best_cv_auc:
            best_cv_auc = metrics["cv_roc_auc"]
            best_name, best_search = name, search

    artifact = {
        "model": best_search.best_estimator_,
        "model_name": best_name,
        "threshold": PREDICTION_THRESHOLD,
        "feature_columns": list(X.columns)
    }
    joblib.dump(artifact, MODEL_DIR / "best_churn_model.joblib")

    report = {
        "selected_model": best_name,
        "selection_metric": "5-fold cross-validation ROC-AUC on training data",
        "prediction_threshold": PREDICTION_THRESHOLD,
        "test_records": int(len(X_test)),
        "results": results
    }
    with open(MODEL_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\n" + "=" * 55)
    print(f"Selected model: {best_name}")
    print(f"Saved model: {MODEL_DIR / 'best_churn_model.joblib'}")
    print(f"Saved metrics: {MODEL_DIR / 'evaluation_metrics.json'}")
    print("=" * 55)


if __name__ == "__main__":
    main()
