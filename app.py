import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(page_title="Heart Disease Classifier", page_icon="❤️")

# Load the saved model and scaler
@st.cache_resource
def load_model_assets():
    model = joblib.load("HeartDisease_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

def main():
    st.title("❤️ Heart Disease Prediction")
    st.markdown("--- ")

    try:
        model, scaler = load_model_assets()

        st.sidebar.header("Patient Input Features")


        age = st.sidebar.number_input("Age", 1, 100, 50)
        sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
        cp = st.sidebar.slider("Chest Pain Type (cp)", 0, 3, 1)
        trestbps = st.sidebar.number_input("Resting Blood Pressure", 80, 200, 120)
        restecg = st.sidebar.slider("Resting Electrocardiographic Results", 0, 2, 0)
        thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
        exang = st.sidebar.selectbox("Exercise Induced Angina", options=[0, 1])
        oldpeak = st.sidebar.number_input("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
        slope = st.sidebar.slider("Slope of Peak Exercise ST Segment", 0, 2, 1)
        ca = st.sidebar.slider("Number of Major Vessels", 0, 4, 0)
        thal = st.sidebar.slider("Thal", 0, 3, 2)

        # Organize input data into a dataframe or array
        # Ensure the order matches exactly with the training features
        input_data = np.array([[age, sex, cp, trestbps, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        # Scale the data
        scaled_input = scaler.transform(input_data)

        if st.button("Predict"):
            # Model probability mapping: index 0 = Disease, index 1 = Healthy
            probabilities = model.predict_proba(scaled_input)[0]
            prob_disease = probabilities[0] * 100
            prob_healthy = probabilities[1] * 100

            prediction = model.predict(scaled_input)[0]

            # In this dataset: 0 = Heart Disease Present, 1 = Healthy
            if prediction == 0:
                st.error(f"### Result: High Risk of Heart Disease ({prob_disease:.1f}% estimated risk)")
            else:
                st.success(f"### Result: Low Risk / Healthy ({prob_healthy:.1f}% confidence)")

    except Exception as e:
        st.error(f"Error loading model or performing prediction: {e}")

if __name__ == '__main__':
    main()
