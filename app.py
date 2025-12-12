from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# Load trained pipeline
bundle = joblib.load("pipeline.joblib")
pipe = bundle["pipeline"]
FEATURES = bundle["features"]

app = FastAPI(title="Student Pass/Fail API")

# Allow frontend (later) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictInput(BaseModel):
    data: dict

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: PredictInput):
    # Arrange input in correct feature order
    row = {f: payload.data.get(f) for f in FEATURES}
    X = pd.DataFrame([row], columns=FEATURES)

    prediction = int(pipe.predict(X)[0])
    probability = float(pipe.predict_proba(X)[0][1])

    return {
        "prediction": prediction,
        "pass_probability": probability
    }
