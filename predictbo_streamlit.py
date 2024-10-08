import streamlit as st
import pickle
import numpy as np
import pandas as pd
from file_uploader import upload_files

# Call the file uploader function (limit of 5 files)
dataframes = upload_files()

# Set the title of the app
st.title('Predict Bo')

# Provide a description of the app
st.write("This app uses 6 inputs to predict the value of Bo using an ANN model. You can either input single values or select an uploaded file for batch predictions.")

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

# Decision point for the user to either make a single prediction or proceed with CSV
prediction_mode = st.radio(
    "How would you like to proceed?",
    ("Single Prediction", "Batch Prediction (Upload CSV)")
)

# Single prediction flow
if prediction_mode == "Single Prediction":
    st.header("Single Prediction")

    # Manually input values
    Rs = st.number_input('Solution Gas Oil Ratio (SCF/STB)', min_value=0.0, format="%.2f")
    T = st.number_input('Temperature (°F)', min_value=0.0, format="%.2f")
    yg = st.number_input('Gas Specific Gravity (-)', min_value=0.0, format="%.2f")
    API = st.number_input('API (-)', min_value=0.0, format="%.2f")
    yo = st.number_input('Oil Specific Gravity (-)', min_value=0.0, format="%.2f")
    Pb = st.number_input('Bubble Point Pressure (Psi)', min_value=0.0, format="%.2f")

    # Prepare the input for prediction
    input_data = np.array([[Rs, T, yg, API, yo, Pb]])

    # Preprocess the input data (scale the features)
    scaled_data = scaler.transform(input_data)

    # Make prediction using the loaded model
    prediction = model.predict(scaled_data)

    # Display the prediction result
    st.write(f'The predicted Bo value is: {prediction[0][0]:.4f}')

# Batch prediction flow
elif prediction_mode == "Batch Prediction (Upload CSV)" and dataframes:
    st.header("Batch Prediction with Uploaded CSV")

    # Let the user select which uploaded CSV file they want to use
    selected_file = st.selectbox("Select the CSV file to use", options=[f"File {i+1}" for i in range(len(dataframes))])

    # Map the selected file to the dataframe
    df = dataframes[int(selected_file.split()[1]) - 1]

    # Display a preview of the selected CSV file
    st.write("Selected File Data Preview:")
    st.write(df.head())

    # Dropdowns to select columns from the uploaded file
    col_Rs = st.selectbox('Select column for Solution Gas Oil Ratio Rs', df.columns)
    col_T = st.selectbox('Select column for Temperature T', df.columns)
    col_yg = st.selectbox('Select column for Gas Specific Gravity yg', df.columns)
    col_API = st.selectbox('Select column for API', df.columns)
    col_yo = st.selectbox('Select column for Oil Specific Gravity yo', df.columns)
    col_Pb = st.selectbox('Select column for Bubble Point Pressure Pb', df.columns)

    # Extract selected columns for prediction
    Rs = df[col_Rs].values
    T = df[col_T].values
    yg = df[col_yg].values
    API = df[col_API].values
    yo = df[col_yo].values
    Pb = df[col_Pb].values

    # Prepare the input for prediction
    input_data = np.column_stack((Rs, T, yg, API, yo, Pb))

    # Preprocess the input data (scale the features)
    scaled_data = scaler.transform(input_data)

    # Make batch predictions using the loaded model
    predictions = model.predict(scaled_data)

    # Display the batch predictions
    st.write("Batch Predictions:")
    st.write(predictions)

else:
    st.info("Please upload a CSV file to proceed with batch predictions.")

