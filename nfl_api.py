
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("nfl_model.joblib")
scaler = joblib.load("nfl_scaler.joblib")

app = FastAPI()

class PredictionInput(BaseModel):
    YardsPerPlay: float
    RushAttempts: int
    PassAttempts: int
    FirstDowns: int

@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        input_array = np.array([[input_data.YardsPerPlay, input_data.RushAttempts, 
                                 input_data.PassAttempts, input_data.FirstDowns]])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)
        return {"predicted_touchdowns": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
