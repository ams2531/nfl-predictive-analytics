from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Define input model for predictions
class PredictionInput(BaseModel):
    YardsPerPlay: float
    RushAttempts: int
    PassAttempts: int
    FirstDowns: int

# Load pre-trained model
model = joblib.load('nfl_model.joblib')

@app.get("/")
def read_root():
    return {"message": "NFL Predictor API is running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    # Prepare input data for the model
    features = np.array([[
        input_data.YardsPerPlay,
        input_data.RushAttempts,
        input_data.PassAttempts,
        input_data.FirstDowns
    ]])
    
    # Make prediction
    prediction = model.predict(features)
    probability = model.predict_proba(features).max()

    return {
        "prediction": "Win" if prediction[0] == 1 else "Loss",
        "probability": float(probability),  # Convert numpy float to Python float
        "input_received": input_data.dict()
    }
