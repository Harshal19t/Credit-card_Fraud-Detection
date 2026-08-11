from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os
import logging
from prometheus_fastapi_instrumentator import Instrumentator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "../models/fraud_model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the file exists.")

model = joblib.load(model_path)
logger.info("Model loaded successfully.")

# Define input schema with validation
class Transaction(BaseModel):
    V1: float = Field(..., gt=-10, lt=10)
    V2: float = Field(..., gt=-10, lt=10)
    V3: float = Field(..., gt=-10, lt=10)
    V4: float = Field(..., gt=-10, lt=10)
    V5: float = Field(..., gt=-10, lt=10)
    V6: float = Field(..., gt=-10, lt=10)
    V7: float = Field(..., gt=-10, lt=10)
    V8: float = Field(..., gt=-10, lt=10)
    V9: float = Field(..., gt=-10, lt=10)
    V10: float = Field(..., gt=-10, lt=10)
    V11: float = Field(..., gt=-10, lt=10)
    V12: float = Field(..., gt=-10, lt=10)
    V13: float = Field(..., gt=-10, lt=10)
    V14: float = Field(..., gt=-10, lt=10)
    V15: float = Field(..., gt=-10, lt=10)
    V16: float = Field(..., gt=-10, lt=10)
    V17: float = Field(..., gt=-10, lt=10)
    V18: float = Field(..., gt=-10, lt=10)
    V19: float = Field(..., gt=-10, lt=10)
    V20: float = Field(..., gt=-10, lt=10)
    V21: float = Field(..., gt=-10, lt=10)
    V22: float = Field(..., gt=-10, lt=10)
    V23: float = Field(..., gt=-10, lt=10)
    V24: float = Field(..., gt=-10, lt=10)
    V25: float = Field(..., gt=-10, lt=10)
    V26: float = Field(..., gt=-10, lt=10)
    V27: float = Field(..., gt=-10, lt=10)
    V28: float = Field(..., gt=-10, lt=10)
    Amount: float = Field(..., gt=0)

# Initialize FastAPI
app = FastAPI()
Instrumentator().instrument(app).expose(app)

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Fraud Detection API is running!"}

# Predict endpoint
@app.post("/predict")
async def predict(request: Request, transaction: Transaction):
    try:
        logger.info(f"Received input: {transaction.dict()}")

        # Convert input to DataFrame
        input_data = pd.DataFrame([transaction.dict()])

        # Predict
        prediction = model.predict(input_data)[0]
        fraud_probability = model.predict_proba(input_data)[0][1]  # Probability of fraud (class 1)

        # Classify as fraud if probability > threshold
        threshold = 0.5
        is_fraud = fraud_probability > threshold

        logger.info(f"Prediction: {prediction}, Fraud Probability: {fraud_probability}")

        return {
            "prediction": int(prediction),
            "fraud_probability": float(fraud_probability),
            "is_fraud": bool(is_fraud),
            "model_version": "1.0",
            "threshold": threshold,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# Model info endpoint
@app.get("/model_info")
async def model_info():
    try:
        feature_names = model.feature_names_in_ if hasattr(model, "feature_names_in_") else ["Unknown"]
        class_names = list(model.classes_) if hasattr(model, "classes_") else ["Unknown"]

        return {
            "model_type": str(type(model).__name__),
            "feature_names": list(feature_names),
            "class_names": class_names,
            "training_date": "2026-08-11",
            "version": "1.0"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}