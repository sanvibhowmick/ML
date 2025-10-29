import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Load the trained pipeline
with open("pipeline_v1.bin", "rb") as f_in:
    pipeline = pickle.load(f_in)

app = FastAPI(title="customer-convert")

# Input data model
class Customer(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# Output data model
class PredictResponse(BaseModel):
    prob: float
    res: bool

def predict_single(customer):
    prob = pipeline.predict_proba([customer])[0, 1]
    return prob




@app.post("/predict", response_model=PredictResponse)
def predict(customer: Customer):
    prob = predict_single(customer.model_dump())
    return PredictResponse(
        prob=prob,
        res=prob >= 0.5
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
