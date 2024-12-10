import streamlit as st
import joblib
from tensorflow.keras.models import load_model
from transformers import Discretizer
from prediction_mode_selection import handle_prediction_mode
from file_uploader import upload_files

# Set the title of the app
st.title('Predict Bo')

# Add a markdown with smaller font size and clickable link
st.markdown(
    """
    <p style="font-size:14px;">
        <a href="https://www.linkedin.com/in/baaba-yakuub-abdul-muumin-4b181729a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=ios_app" target="_blank" style="text-decoration:none; color:blue;">
            App by Yakuub Abdul-Muumin
        </a>
    </p>
    """,
    unsafe_allow_html=True
)

# Add a markdown description of Bo
st.markdown("""
**Oil formation volume factor (Bo)** is a petrophysical parameter used in numerous oil and gas reservoir calculations.
""")
# Provide a description of the app
st.write("Estimate Bo values with an ANN-powered tool. Choose between entering individual inputs or uploading a dataset for comprehensive predictions.")

# Call the file uploader function and get uploaded files
uploaded_files = upload_files()

# Load the preprocessing pipeline and ANN model
pipeline = joblib.load('preprocessing_pipeline.pkl')
model = load_model('trained_model.h5')

# Call the modularized prediction mode handler
handle_prediction_mode(pipeline, model, uploaded_files)