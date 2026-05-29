from pydantic import BaseModel, Field
from typing import Dict, Union


class PredictRequest(BaseModel):
    features:  Dict[str, Union[int, float]] = Field(
        ...,
        json_schema_extra={
            "example": {
                "Age": 0,
                "Income": 0,
                "LoanAmount": 0,
                "CreditScore": 0,
                "MonthsEmployed": 0,
                "NumCreditLines": 0,
                "InterestRate": 0,
                "LoanTerm": 0,
                "DTIRatio": 0.0,
                "Education_Bachelor's": 0,
                "Education_High School": 0,
                "Education_Master's": 0,
                "Education_PhD": 0,
                "EmploymentType_Full-time": 0,
                "EmploymentType_Part-time": 0,
                "EmploymentType_Self-employed": 0,
                "EmploymentType_Unemployed": 0,
                "MaritalStatus_Divorced": 0,
                "MaritalStatus_Married": 0,
                "MaritalStatus_Single": 0,
                "HasMortgage_No": 0,
                "HasMortgage_Yes": 1,
                "HasDependents_No": 0,
                "HasDependents_Yes": 0,
                "LoanPurpose_Auto": 0,
                "LoanPurpose_Business": 0,
                "LoanPurpose_Education": 0,
                "LoanPurpose_Home": 0,
                "LoanPurpose_Other": 0,
                "HasCoSigner_No": 0,
                "HasCoSigner_Yes": 0
            }
        }
    )


class PredictResponse(BaseModel):
    prediction: int
    probability: float