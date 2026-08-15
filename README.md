# Heart Disease Prediction Project

This project predicts whether a patient is likely to have heart disease using a trained machine learning model. It includes two interfaces:

- A Streamlit web app for interactive use in the browser
- A FastAPI backend service for API-based prediction requests

The model is trained on a heart disease dataset and saved as reusable artifacts for prediction.

---

## Project Overview

The application uses a logistic regression classifier with feature scaling. The project loads a saved model and scaler and makes predictions based on patient health indicators such as:

- age
- sex
- chest pain type
- resting blood pressure
- resting ECG
- maximum heart rate
- exercise-induced angina
- ST depression
- slope of ST segment
- number of major vessels
- thalassemia category

The dataset is stored in:

- `Notebook/heart.csv`

The model files are:

- `HeartDisease_model.pkl`
- `scaler.pkl`

---

## Folder Structure

```text
Heart Disease/
├── app.py                          # Streamlit application
├── requirements.txt               # Main dependencies for the Streamlit app
├── HeartDisease_model.pkl         # Trained ML model
├── scaler.pkl                    # Saved StandardScaler used during training
├── Notebook/
│   ├── heart.csv                 # Dataset used for modeling
│   └── Logistic_regression_on_Heart_Disease_dataset.ipynb
├── API/
│   ├── api.py                    # FastAPI prediction service
│   └── api_requirements.txt      # API-only dependencies
├── README.md                     # Project documentation
└── __pycache__/                  # Python cache files
```

---

## 1. Streamlit App

The Streamlit app is the main user-facing interface and is defined in `app.py`.

### Purpose
- Allows the user to input patient details through a sidebar
- Scales the input values using the saved scaler
- Sends the values to the trained logistic regression model
- Shows prediction result and estimated risk percentage

### Run the Streamlit app
https://heart-disease-depi.streamlit.app/

---
## 2. FastAPI Service

The API is placed in the `API` folder and is defined in `API/api.py`.

### Purpose
- Exposes a REST endpoint for programmatic prediction requests
- Accepts patient features in JSON format
- Returns prediction result and probabilities

### Run the API

```powershell
cd API
pip install -r api_requirements.txt
python api.py
```

The app starts on:

```text
http://127.0.0.1:8000
```

Swagger docs are available at:

```text
http://127.0.0.1:8000/docs
```

### API endpoint

#### GET /
Returns a simple health message.

Example response:

```json
{
  "message": "Heart Disease Prediction API is running"
}
```

#### POST /predict
Accepts JSON patient data and returns the prediction.

Example payload:

```json
{
  "age": 52,
  "sex": 1,
  "cp": 1,
  "trestbps": 130,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 1.5,
  "slope": 1,
  "ca": 0,
  "thal": 2
}
```

Example response:

```json
{
  "prediction": "Healthy",
  "probability_heart_disease": 18.35,
  "probability_healthy": 81.65
}
```

---

## 3. Dependencies

The main project dependencies are listed in `requirements.txt` and include:

- Streamlit
- pandas
- numpy
- joblib
- scikit-learn

The API-specific dependencies are listed in `API/api_requirements.txt`:

- fastapi
- uvicorn
- joblib
- scikit-learn
- numpy

---

## 4. Model Details

This project uses a trained logistic regression model with preprocessing through a StandardScaler.

Important notes:

- The scaler and model are saved and loaded from disk using `joblib`
- The input order must match the training feature order exactly
- The model is trained for binary classification:
  - `0` = Heart Disease Present
  - `1` = Healthy

In the Streamlit app, the code converts the feature values to a NumPy array and applies scaling before prediction.

---

## 5. How the Project Works

1. The dataset is explored in the notebook.
2. A machine learning model is trained and validated.
3. The trained model and scaler are saved as `.pkl` files.
4. The Streamlit app loads those files and allows users to make predictions interactively.
5. The FastAPI service loads the same files and exposes the prediction logic through HTTP.

---

## 6. Recommended Setup

It is best to create a virtual environment before installing dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For the API-only environment:

```powershell
python -m venv .venv-api
.\.venv-api\Scripts\Activate.ps1
pip install -r API/api_requirements.txt
```

---

## 7. Notes

- Keep the same Python version and scikit-learn version used during training to avoid model compatibility warnings.
- The model artifacts were generated from a specific training environment, so version mismatches may lead to warnings or inconsistent predictions.
- The Streamlit app and API use the same model files, so they stay consistent across interfaces.

---

## 8. Future Improvements

Possible next steps for the project include:

- adding model evaluation metrics and confusion matrix
- adding a better UI with patient history and risk interpretation
- deploying the API to a cloud service
- creating a front-end dashboard for clinicians
- adding logging and monitoring for API requests

---

## Summary

This project demonstrates an end-to-end machine learning workflow for heart disease prediction:

- data exploration
- model training
- model saving
- web-based inference via Streamlit
- API-based inference via FastAPI

It is a practical example of combining machine learning, deployment, and interactive visualization in one project.
