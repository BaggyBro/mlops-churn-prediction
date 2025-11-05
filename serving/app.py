from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from autogluon.tabular import TabularPredictor

app = FastAPI()

model = TabularPredictor.load('/home/baggybro/skills/machine/churn-platform/notebooks/AutogluonModels/ag-20251105_144149')

class InputData(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float


@app.post("/predict")
def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)
    pred_proba = model.predict_proba(df)
    return {
        "prediction": prediction.tolist(),
        "probabilities": pred_proba.values.tolist()
    }