import streamlit as st
import pickle
import numpy as np

# Set the title of the app
st.title('Predict Bo')

# Provide a description of the app
st.write("This app uses 6 inputs to predict the value of Bo using an ANN model. Use the form below to get started!")

# Load the ANN model and scaler
with open('ann_bo_model.pickle', 'rb') as model_pickle:
    model = pickle.load(model_pickle)

with open('scaler.pickle', 'rb') as scaler_pickle:
    scaler = pickle.load(scaler_pickle)

# Optional: Display the loaded model and scaler for verification
st.write("Loaded ANN Model:")
st.write(model)
st.write("Loaded Scaler:")
st.write(scaler)

# Input fields for user to input data in Streamlit
st.header("Bo Prediction")

# Taking user inputs for the features
Rs = st.number_input('Solution Gas Oil Ratio (SCF/STB)', min_value=0.0, format="%.2f")
T = st.number_input('Temperature (°F)', min_value=0.0, format="%.2f")
yg = st.number_input('Gas Specific Gravity (-)', min_value=0.0, format="%.2f")
API = st.number_input('API (-)', min_value=0.0, format="%.2f")
yo = st.number_input('Oil Specific Gravity (-)', min_value=0.0, format="%.2f")
Pb = st.number_input('Bubble Point Pressure (Psi)', min_value=0.0, format="%.2f")

# Convert inputs into a NumPy array
input_data = np.array([[Rs, T, yg, API, yo, Pb]])

# Preprocess the input data (scale the features)
scaled_data = scaler.transform(input_data)

# Make prediction using the loaded model
prediction = model.predict(scaled_data)

# Display the prediction result
st.write(f'The predicted Bo value is: {prediction[0][0]:.4f}')
