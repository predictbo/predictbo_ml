import streamlit as st
import joblib
from tensorflow.keras.models import load_model
from prediction_mode_selection import handle_prediction_mode


# Set the title of the app
st.title('Predict Bo')

# Provide a description of the app
st.write("Estimate Bo values with our ANN-powered tool. Choose between entering individual inputs or uploading a dataset for comprehensive predictions.")

# Call the file uploader function 
uploaded_files = st.file_uploader('Choose a CSV file', type='csv', accept_multiple_files=True)

# Load the preprocessing pipeline and ANN model
pipeline = joblib.load('preprocessing_pipeline.pkl')
model = load_model('trained_model.h5')

# Call the modularized prediction mode handler
handle_prediction_mode(pipeline, model, uploaded_files)