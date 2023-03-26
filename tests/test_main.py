import pytest
from fastapi.testclient import TestClient
from main import app, InputData

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the FastAPI model inference API!"}

def test_predict_0():
    input_data = InputData(age=25, workclass="Private", fnlgt=226802, education="11th", education_num=7, marital_status="Never-married", occupation="Machine-op-inspct", relationship="Own-child", race="White", sex="Male", capital_gain=0, capital_loss=0, hours_per_week=40, native_country="Germany")
    response = client.post("/predict", json=input_data.dict())
    assert response.status_code == 200
    assert response.json() == {"predicted_output": "0"}

def test_predict_1():
    input_data = InputData(age=42, workclass="Self-emp-not-inc", fnlgt=287927, education="HS-grad", education_num=9, marital_status="Married-civ-spouse", occupation="Exec-managerial", relationship="Husband", race="White", sex="Male", capital_gain=15024, capital_loss=0, hours_per_week=60, native_country="United-States")
    response = client.post("/predict", json=input_data.dict())
    assert response.status_code == 200
    assert response.json() == {"predicted_output": "1"}