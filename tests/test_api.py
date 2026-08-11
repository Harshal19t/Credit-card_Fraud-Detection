from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Fraud Detection API is running!"}

def test_predict_endpoint():
    test_data = {
        "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782,
        "V5": -0.3383, "V6": 0.4624, "V7": 0.2398, "V8": 0.0987,
        "V9": 0.3264, "V10": 0.1894, "V11": -0.2256, "V12": -0.6384,
        "V13": 0.1012, "V14": -0.3398, "V15": 0.1671, "V16": 0.1258,
        "V17": -0.0064, "V18": 0.0883, "V19": -0.0574, "V20": -0.0693,
        "V21": -0.2256, "V22": 0.0626, "V23": 0.0617, "V24": 0.0094,
        "V25": -0.0314, "V26": -0.0688, "V27": 0.0349, "V28": 0.0274,
        "Amount": 149.62
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    assert "fraud_probability" in response.json()