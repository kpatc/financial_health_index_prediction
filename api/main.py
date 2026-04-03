"""
FastAPI application for serving Financial Health Index predictions.

Endpoints:
- GET / : Health check and API info
- POST /predict : Single prediction
- POST /predict/batch : Batch predictions
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from cleaning.clean import (
    normalize_strings, encode_ordinal_features, encode_binary_features,
    encode_categorical_features, log_transform_financial, add_engineered_features
)

# Initialize FastAPI app
app = FastAPI(
    title="Financial Health Index Predictor",
    description="Predict financial health (Low/Medium/High) for small businesses",
    version="1.0.0"
)

# Global variables for model and preprocessor
model = None
preprocessor = None


def load_model():
    """Load trained model and preprocessor on startup."""
    global model, preprocessor

    model_path = Path(__file__).parent.parent / 'modeling' / 'model.pkl'
    preprocessor_path = Path(__file__).parent.parent / 'modeling' / 'preprocessor.pkl'

    if not model_path.exists() or not preprocessor_path.exists():
        raise FileNotFoundError(
            f"Model artifacts not found. Please run modeling/train.py first.\n"
            f"Expected: {model_path}, {preprocessor_path}"
        )

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    print(f"Loaded model from {model_path}")
    print(f"Loaded preprocessor from {preprocessor_path}")


# ===== Request/Response Models =====

class BusinessFeatures(BaseModel):
    """Pydantic model for single business prediction request."""
    country: Optional[str] = Field(None, description="Country: Eswatini/Lesotho/Malawi/Zimbabwe")
    owner_age: Optional[float] = Field(None, description="Age of owner in years")
    attitude_stable_business_environment: Optional[str] = Field(None, description="Yes/No")
    attitude_worried_shutdown: Optional[str] = Field(None, description="Yes/No")
    compliance_income_tax: Optional[str] = Field(None, description="Yes/No")
    perception_insurance_doesnt_cover_losses: Optional[str] = Field(None, description="Yes/No")
    perception_cannot_afford_insurance: Optional[str] = Field(None, description="Yes/No")
    personal_income: Optional[float] = Field(None, description="Monthly personal income")
    business_expenses: Optional[float] = Field(None, description="Monthly/annual business expenses")
    business_turnover: Optional[float] = Field(None, description="Annual business turnover")
    business_age_years: Optional[float] = Field(None, description="Years running business")
    motor_vehicle_insurance: Optional[str] = Field(None, description="Never had/Used to have/Have now")
    has_mobile_money: Optional[str] = Field(None, description="Never had/Used to have/Have now")
    current_problem_cash_flow: Optional[str] = Field(None, description="Yes/No")
    has_cellphone: Optional[str] = Field(None, description="Yes/No")
    owner_sex: Optional[str] = Field(None, description="Female/Male")
    offers_credit_to_customers: Optional[str] = Field(None, description="Yes/No")
    attitude_satisfied_with_achievement: Optional[str] = Field(None, description="Yes/No")
    has_credit_card: Optional[str] = Field(None, description="Never had/Used to have/Have now")
    keeps_financial_records: Optional[str] = Field(None, description="Yes/No")
    perception_insurance_companies_dont_insure_businesses_like_yours: Optional[str] = Field(None)
    perception_insurance_important: Optional[str] = Field(None)
    has_insurance: Optional[str] = Field(None)
    covid_essential_service: Optional[str] = Field(None)
    attitude_more_successful_next_year: Optional[str] = Field(None)
    problem_sourcing_money: Optional[str] = Field(None)
    marketing_word_of_mouth: Optional[str] = Field(None)
    has_loan_account: Optional[str] = Field(None)
    has_internet_banking: Optional[str] = Field(None)
    has_debit_card: Optional[str] = Field(None)
    future_risk_theft_stock: Optional[str] = Field(None)
    business_age_months: Optional[float] = Field(None)
    medical_insurance: Optional[str] = Field(None)
    funeral_insurance: Optional[str] = Field(None)
    motivation_make_more_money: Optional[str] = Field(None)
    uses_friends_family_savings: Optional[str] = Field(None)
    uses_informal_lender: Optional[str] = Field(None)

    class Config:
        schema_extra = {
            "example": {
                "country": "malawi",
                "owner_age": 35,
                "attitude_stable_business_environment": "Yes",
                "personal_income": 50000,
                "business_expenses": 20000,
                "business_turnover": 100000,
                "business_age_years": 5
            }
        }


class PredictionResponse(BaseModel):
    """Response model for single prediction."""
    prediction: str = Field(description="Predicted class: Low/Medium/High")
    probabilities: dict = Field(description="Probability for each class")
    confidence: float = Field(description="Max probability across classes")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    businesses: List[BusinessFeatures] = Field(description="List of businesses to predict")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[dict] = Field(description="List of predictions")
    count: int = Field(description="Number of predictions made")


# ===== Helper Functions =====

def clean_and_preprocess(data_dict: dict) -> pd.DataFrame:
    """
    Clean and preprocess a single business record.
    Mirrors the cleaning pipeline used in training.
    """
    # Convert dict to DataFrame for processing
    df = pd.DataFrame([data_dict])

    # Apply cleaning steps
    df = normalize_strings(df)
    df = encode_ordinal_features(df)
    df = encode_binary_features(df)
    df = encode_categorical_features(df)
    df = log_transform_financial(df)
    df = add_engineered_features(df)

    return df


def predict_single(features: BusinessFeatures) -> dict:
    """Make a single prediction."""
    if model is None:
        raise RuntimeError("Model not loaded. Application startup failed.")

    try:
        # Convert Pydantic model to dict
        data_dict = features.dict()

        # Clean and preprocess
        X = clean_and_preprocess(data_dict)

        # Drop ID and Target if present
        X = X.drop(columns=['ID', 'Target'], errors='ignore')

        # Make prediction
        prediction_class = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        class_names = ['Low', 'Medium', 'High']
        pred_name = class_names[int(prediction_class)]
        confidence = float(probabilities[int(prediction_class)])

        return {
            'prediction': pred_name,
            'probabilities': {
                'Low': float(probabilities[0]),
                'Medium': float(probabilities[1]),
                'High': float(probabilities[2])
            },
            'confidence': confidence
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


# ===== API Endpoints =====

@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    try:
        load_model()
        print("Application started successfully")
    except FileNotFoundError as e:
        print(f"Startup error: {e}")
        raise


@app.get("/")
async def health_check():
    """
    Health check endpoint - verify API is running and model is loaded.
    """
    return {
        "status": "healthy",
        "service": "Financial Health Index Predictor",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "endpoints": {
            "predict_single": "POST /predict",
            "predict_batch": "POST /predict/batch",
            "docs": "/docs",
            "openapi_schema": "/openapi.json"
        }
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: BusinessFeatures):
    """
    Make a single prediction for a business.

    Request body: Business features (all optional - NaN will be imputed)
    Response: Predicted class + probabilities
    """
    return predict_single(features)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Make batch predictions for multiple businesses.

    Request body: List of business feature dictionaries
    Response: List of predictions
    """
    predictions = []
    for business in request.businesses:
        pred = predict_single(business)
        predictions.append(pred)

    return {
        "predictions": predictions,
        "count": len(predictions)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
