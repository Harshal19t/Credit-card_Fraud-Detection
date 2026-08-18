from fastapi import FastAPI, HTTPException, Request
import joblib
import pandas as pd
import os
import logging
from prometheus_fastapi_instrumentator import Instrumentator

from app.schema import Transaction

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model
model_path = os.path.join(os.path.dirname(__file__), "../models/fraud_model.pkl")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}. Please ensure the file exists.")

model = joblib.load(model_path)
logger.info("Model loaded successfully.")

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
        logger.info(f"Received input: {transaction.model_dump()}")

        # Convert input to DataFrame (column order matches training feature order: V1-V28, Amount)
        input_data = pd.DataFrame([transaction.model_dump()])

        # Predict. The model was trained on a plain numpy array (no feature names), so we
        # pass .values here too -- otherwise scikit-learn warns on every request that the
        # model was "fitted without feature names".
        prediction = model.predict(input_data.values)[0]
        fraud_probability = model.predict_proba(input_data.values)[0][1]  # Probability of fraud (class 1)

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
        feature_names = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else ["Unknown"]
        class_names = [int(c) for c in model.classes_] if hasattr(model, "classes_") else ["Unknown"]

        return {
            "model_type": str(type(model).__name__),
            "feature_names": [str(f) for f in feature_names],
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