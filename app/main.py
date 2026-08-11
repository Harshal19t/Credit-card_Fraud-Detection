from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from prometheus_fastapi_instrumentator import Instrumentator

model = joblib.load("models/fraud_model.pkl")

class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

app = FastAPI()
Instrumentator().instrument(app).expose(app)

@app.get("/")
async def root():
    return {"message": "Fraud Detection API is running!"}

@app.post("/predict")
async def predict(transaction: Transaction):
    try:
        input_data = pd.DataFrame([transaction.dict()])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        return {
            "prediction": int(prediction), 
            "probability": float(probability),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))