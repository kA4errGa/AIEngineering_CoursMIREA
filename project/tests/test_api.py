from fastapi.testclient import TestClient

from src.service.main import app

client = TestClient(app)


def test_health():

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict():

    payload = {
        "features": {
            "Age": 56,
            "Income": 85994,
            "LoanAmount": 50587,
            "CreditScore": 520,
            "MonthsEmployed": 80,
            "NumCreditLines": 4,
            "InterestRate": 15.23,
            "LoanTerm": 36,
            "DTIRatio": 0.44,
            "Education_Bachelor's": 1,
            "Education_High School": 0,
            "Education_Master's": 0,
            "Education_PhD": 0,
            "EmploymentType_Full-time": 1,
            "EmploymentType_Part-time": 0,
            "EmploymentType_Self-employed": 0,
            "EmploymentType_Unemployed": 0,
            "MaritalStatus_Divorced": 1,
            "MaritalStatus_Married": 0,
            "MaritalStatus_Single": 0,
            "HasMortgage_No": 0,
            "HasMortgage_Yes": 1,
            "HasDependents_No": 0,
            "HasDependents_Yes": 1,
            "LoanPurpose_Auto": 0,
            "LoanPurpose_Business": 0,
            "LoanPurpose_Education": 0,
            "LoanPurpose_Home": 0,
            "LoanPurpose_Other": 1,
            "HasCoSigner_No": 0,
            "HasCoSigner_Yes": 1
        }
    }

    response = client.post(
        "/predict",
        json=payload
    )
    data = response.json()

    assert response.status_code == 200
    assert "prediction" in data
    assert "probability" in data

def test_predict_missing_feature():

    payload = {
        "features": {
            "Age": 56
        }
    }

    response = client.post(
        "/predict",
        json=payload
    )
    data = response.json()

    assert response.status_code == 400
    assert data["detail"]["error"] == "Missing features"