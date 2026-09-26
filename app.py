import streamlit as st
import numpy as np
import joblib

# Load trained model
artifacts = joblib.load("sonar_model.pkl")

model = artifacts["model"]
scaler = artifacts["scaler"]
encoder = artifacts["encoder"]

# Page configuration
st.set_page_config(
    page_title="Sonar Rock & Mine Prediction",
    page_icon="🌊",
    layout="centered"
)


# Title
st.title("🌊 Sonar Rock & Mine Prediction")

st.write(
    "Enter the 60 sonar signal features to predict whether "
    "the object is a Rock or a Mine."
)

# Input fields
st.subheader("Sonar Features")

input_data = []

for i in range(60):
    value = st.number_input(
        f"Feature {i + 1}",
        value=0.0,
        format="%.4f"
    )
    input_data.append(value)


# Prediction button
if st.button("Predict", type="primary"):

    # Convert input to NumPy array
    input_array = np.asarray(input_data)

    # Reshape into one sample
    input_reshaped = input_array.reshape(1, -1)

    # Standardize using the scaler from training
    input_standardized = scaler.transform(input_reshaped)

    # Make prediction
    prediction = model.predict(input_standardized)

    # Convert numerical prediction back to M/R
    prediction_label = encoder.inverse_transform(prediction)

    result = prediction_label[0]

    st.subheader("Prediction Result")

    if result == "M":
        st.error("💣 The object is a Mine")
    else:
        st.success("🪨 The object is a Rock")