import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add business-oriented features while preserving the input DataFrame."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.copy()

        tenure = pd.to_numeric(X.get("tenure", 0), errors="coerce").fillna(0)
        monthly = pd.to_numeric(X.get("MonthlyCharges", 0), errors="coerce").fillna(0)
        total = pd.to_numeric(X.get("TotalCharges", 0), errors="coerce").fillna(0)

        # 1. Average billed amount per active month (safe for zero tenure).
        X["avg_monthly_spend"] = np.where(tenure > 0, total / tenure, monthly)

        # 2. Tenure segment.
        X["tenure_group"] = pd.cut(
            tenure,
            bins=[-1, 12, 24, 48, np.inf],
            labels=["0-12 months", "13-24 months", "25-48 months", "49+ months"]
        ).astype("object")

        # 3. Number of subscribed services.
        service_cols = [
            "PhoneService", "MultipleLines", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies"
        ]
        available = [c for c in service_cols if c in X.columns]
        if available:
            X["services_count"] = sum(
                X[c].astype(str).str.lower().isin(["yes"]).astype(int)
                for c in available
            )
        else:
            X["services_count"] = 0

        # 4. Monthly charge relative to the number of subscribed services.
        X["charges_per_service"] = monthly / X["services_count"].replace(0, 1)

        return X
