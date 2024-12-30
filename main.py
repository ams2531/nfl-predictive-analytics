# Step 1: Open the main.py file in your GitHub repository and replace its content with the updated code
# This step assumes you will manually copy and paste the code into GitHub's web interface.

# Updated main.py content with NFL prediction endpoint
updated_main_py_content = """from fastapi import FastAPI
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

# Load pre-trained model (replace 'nfl_model.joblib' with your actual model file)
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
    features = np.array([[input_data.YardsPerPlay, input_data.RushAttempts, input_data.PassAttempts, input_data.FirstDowns]])
    
    # Make prediction
    prediction = model.predict(features)
    probability = model.predict_proba(features).max()

    return {
        "prediction": "Win" if prediction[0] == 1 else "Loss",
        "probability": probability,
        "input_received": input_data.dict()
    }
"""

# Save the updated content to main.py locally (for reference or manual upload)
with open('main.py', 'w') as f:
    f.write(updated_main_py_content)

print("Updated main.py content saved locally. Please copy and paste this into your GitHub repository.")
