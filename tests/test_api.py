from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

VALID_TRANSACTION = {
    "V1": -1.3598, "V2": -0.0728, "V3": 2.5363, "V4": 1.3782,
    "V5": -0.3383, "V6": 0.4624, "V7": 0.2398, "V8": 0.0987,
    "V9": 0.3264, "V10": 0.1894, "V11": -0.2256, "V12": -0.6384,
    "V13": 0.1012, "V14": -0.3398, "V15": 0.1671, "V16": 0.1258,
    "V17": -0.0064, "V18": 0.0883, "V19": -0.0574, "V20": -0.0693,
    "V21": -0.2256, "V22": 0.0626, "V23": 0.0617, "V24": 0.0094,
    "V25": -0.0314, "V26": -0.0688, "V27": 0.0349, "V28": 0.0274,
    "Amount": 149.62,
}


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Fraud Detection API is running!"}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_model_info_endpoint():
    response = client.get("/model_info")
    assert response.status_code == 200
    body = response.json()
    assert "model_type" in body
    assert "feature_names" in body
    assert "class_names" in body


def test_predict_endpoint():
    response = client.post("/predict", json=VALID_TRANSACTION)
    assert response.status_code == 200
    body = response.json()
    assert "fraud_probability" in body
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert "is_fraud" in body
    assert "prediction" in body


def test_predict_missing_field_returns_422():
    incomplete = dict(VALID_TRANSACTION)
    del incomplete["V14"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_out_of_range_value_returns_422():
    invalid = dict(VALID_TRANSACTION)
    invalid["V1"] = 999.0  # outside the (-10, 10) bound enforced by the schema
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_negative_amount_returns_422():
    invalid = dict(VALID_TRANSACTION)
    invalid["Amount"] = -10.0  # Amount must be > 0
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422


def test_predict_wrong_type_returns_422():
    invalid = dict(VALID_TRANSACTION)
    invalid["Amount"] = "not-a-number"
    response = client.post("/predict", json=invalid)
    assert response.status_code == 422
