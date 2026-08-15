from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Heart Disease Prediction API")


# Load the saved model and scaler
model = joblib.load(BASE_DIR / "W:\\DEPI - Agentic & Generative AI\\Technical\\Assignments\\Heart Disease\\HeartDisease_model.pkl")
scaler = joblib.load(BASE_DIR / "W:\\DEPI - Agentic & Generative AI\\Technical\\Assignments\\Heart Disease\\scaler.pkl")


class PatientInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running"}


@app.post("/predict")
def predict(data: PatientInput):
    features = np.array(
        [[
            data.age,
            data.sex,
            data.cp,
            data.trestbps,
            data.restecg,
            data.thalach,
            data.exang,
            data.oldpeak,
            data.slope,
            data.ca,
            data.thal,
        ]],
        dtype=float,
    )

    scaled_features = scaler.transform(features)
    probabilities = model.predict_proba(scaled_features)[0]
    prediction = int(model.predict(scaled_features)[0])

    result = "Heart Disease Present" if prediction == 0 else "Healthy"

    return {
        "prediction": result,
        "probability_heart_disease": round(float(probabilities[0] * 100), 2),
        "probability_healthy": round(float(probabilities[1] * 100), 2),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
