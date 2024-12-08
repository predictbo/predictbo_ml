import streamlit as st
import joblib
from tensorflow.keras.models import load_model
from transformers import Discretizer
from prediction_mode_selection import handle_prediction_mode
from file_uploader import upload_files

# Set the title of the app
st.title('Predict Bo')

# Provide a description of the app
st.write("Estimate Bo values with our ANN-powered tool. Choose between entering individual inputs or uploading a dataset for comprehensive predictions.")

# Call the file uploader function and get uploaded files
uploaded_files = upload_files()

# Load the preprocessing pipeline and ANN model
pipeline = joblib.load('preprocessing_pipeline.pkl')
model = load_model('trained_model.h5')

# Call the modularized prediction mode handler
handle_prediction_mode(pipeline, model, uploaded_files)