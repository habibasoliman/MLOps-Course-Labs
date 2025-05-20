from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import logging
import pandas as pd
import mlflow


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Bank Churn Prediction API")
print("Tracking URI:", mlflow.get_tracking_uri())

# Load the production model from MLflow Model Registry
model = mlflow.pyfunc.load_model("models:/bank_churn_model/Production")

class CustomerData(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float

@app.get("/")
async def home():
    return {"message": "Welcome to the Bank Customer Churn Prediction API!"}

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(data: CustomerData):
    try:
        df = pd.DataFrame([data.dict()])
        prediction = model.predict(df)
        logger.info(f"Prediction made successfully for input: {data}")
        return {"prediction": int(prediction[0])}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {"error": str(e)}
