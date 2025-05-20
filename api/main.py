from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc

app = FastAPI(title="Bank Customer Churn Prediction API")

model_name = "bank_churn_model"  
model_stage = "Production"

model = mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{model_stage}")

# Define the input schema
class CustomerData(BaseModel):
    CreditScore: int
    Geography: str
    Gender: str
    Age: int
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Churn Prediction API"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: CustomerData):
    input_df = data.dict()
    input_df = [input_df]  
    prediction = model.predict(input_df)
    return {"prediction": int(prediction[0])}
