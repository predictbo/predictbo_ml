import streamlit as st
import pickle
import numpy as np
import pandas as pd
from prediction_mode_selection import handle_prediction_mode
from plot_module import generate_plots

# Set the title of the app
st.title('Predict Bo')

# Provide a description of the app
st.write("This app uses 6 inputs to predict the value of Bo using an ANN model. You can either input single values or select an uploaded file for batch predictions.")

# Call the file uploader function 
uploaded_files = st.file_uploader('Choose a CSV file', type='csv', accept_multiple_files=True)

# Load the ANN model and scaler
with open('ann_bo_model.pickle', 'rb') as model_pickle:
    model = pickle.load(model_pickle)

with open('scaler.pickle', 'rb') as scaler_pickle:
    scaler = pickle.load(scaler_pickle)

# Call the function to generate plots
generate_plots(uploaded_files)

# Call the function to handle the prediction mode (either single or batch prediction)
handle_prediction_mode(scaler, model, uploaded_files)
