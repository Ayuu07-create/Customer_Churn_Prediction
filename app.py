from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best_churn_model.joblib"

app = FastAPI(
    title="Customer Churn Prediction",
    description="A simple web interface and REST API for customer churn prediction.",
    version="1.1.0",
)

# The static folder is included with this update package.
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

artifact = None


class CustomerData(BaseModel):
    gender: str = Field(..., examples=["Female"])
    SeniorCitizen: int = Field(..., ge=0, le=1, examples=[0])
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., ge=0, examples=[12])
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No"])
    InternetService: str = Field(..., examples=["Fiber optic"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["Yes"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["Yes"])
    StreamingMovies: str = Field(..., examples=["Yes"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., ge=0, examples=[75.5])
    TotalCharges: Optional[float] = Field(None, ge=0, examples=[906.0])


@app.on_event("startup")
def load_model():
    global artifact
    if MODEL_PATH.exists():
        artifact = joblib.load(MODEL_PATH)


@app.get("/", response_class=HTMLResponse)
def home():
    page = ROOT / "templates" / "index.html"
    if not page.exists():
        return HTMLResponse(
            "<h2>Customer Churn Prediction</h2><p>UI files are missing. "
            "Restore templates/index.html and static/style.css.</p>",
            status_code=500,
        )
    return HTMLResponse(page.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": artifact is not None}


@app.post("/predict")
def predict(customer: CustomerData):
    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Train it first using: python -m src.train_model",
        )

    try:
        customer_df = pd.DataFrame([customer.model_dump()])
        probability = float(artifact["model"].predict_proba(customer_df)[0, 1])
        threshold = float(artifact.get("threshold", 0.5))
        prediction = int(probability >= threshold)

        if probability >= 0.70:
            risk = "High"
        elif probability >= 0.40:
            risk = "Medium"
        else:
            risk = "Low"

        return {
            "churn_probability": round(probability, 4),
            "churn_probability_percent": round(probability * 100, 2),
            "prediction": prediction,
            "prediction_label": "Churn" if prediction == 1 else "No Churn",
            "risk_level": risk,
            "decision_threshold": threshold,
            "model_used": artifact["model_name"],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")
