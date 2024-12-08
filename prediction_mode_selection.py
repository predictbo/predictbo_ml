from missing_values_handler import handle_missing_values
from file_downloader import download_predictions
import streamlit as st
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

def handle_prediction_mode(pipeline, model, uploaded_files):
    prediction_mode = st.radio('How would you like to proceed?', 
                               ('Single Prediction', 'Dataset Prediction'))

    if prediction_mode == 'Single Prediction':
        st.header('Single Prediction')

        # Manually input values
        Rs = st.number_input('Rs', min_value=0.0, format='%.3f')
        T = st.number_input('T', min_value=0.0, format='%.3f')
        API = st.number_input('API', min_value=0.0, format='%.3f')
        Pb = st.number_input('Pb', min_value=0.0, format='%.3f')

        # Prepare the input for prediction
        input_data = np.array([[Rs, T, API, Pb]])

        # Preprocess the input data
        preprocessed_data = pipeline.transform(input_data)

        # Predict button for single prediction
        if st.button("Predict"):
            # Make prediction using the loaded model
            prediction = model.predict(preprocessed_data)

            # Display the prediction result
            st.write(f'The predicted Bo value is: {prediction[0][0]:.3f}')

            # Option to download the prediction
            download_predictions(pd.DataFrame(input_data, columns=['Rs', 'T', 'API', 'Pb']),
                                 prediction, file_type='csv', append=True)

    elif prediction_mode == 'Dataset Prediction':
        st.header('Dataset Prediction')

        if uploaded_files:
            filenames = [file_data[0] for file_data in uploaded_files]
            selected_file_name = st.selectbox("Select the file to use", filenames)

            selected_file_data = next(file_data[1] for file_data in uploaded_files if file_data[0] == selected_file_name)

            st.write("Selected File Data Preview:")
            st.write(selected_file_data.head())

            col_Rs = st.selectbox('Select column for Rs', selected_file_data.columns)
            col_T = st.selectbox('Select column for T', selected_file_data.columns)
            col_API = st.selectbox('Select column for API', selected_file_data.columns)
            col_Pb = st.selectbox('Select column for Pb', selected_file_data.columns)

            # Check for missing values in the selected columns
            if selected_file_data[[col_Rs, col_T, col_API, col_Pb]].isnull().any().any():
                # Show missing value handling option if there are missing values
                df_cleaned = handle_missing_values(selected_file_data)
            else:
                # If no missing values, use the original dataframe
                df_cleaned = selected_file_data

            # Prepare the input for prediction
            input_data = df_cleaned[[col_Rs, col_T, col_API, col_Pb]].values

            # Preprocess the cleaned data
            preprocessed_data = pipeline.transform(input_data)

            if st.button("Predict"):
                predictions = model.predict(preprocessed_data)
                
                st.write("Batch Predictions :")
                    
                # Convert predictions to a DataFrame for better display
                predictions_df = pd.DataFrame(predictions, columns=['Predicted Bo']).round(3)
                
                # Combine the cleaned data with predictions
                results_df = pd.concat([df_cleaned.reset_index(drop=True), predictions_df], axis=1)  
                
                # Display predictions in a better format (table with scroll)
                st.dataframe(results_df)
                
                # Option to download the predictions
                download_predictions(results_df, predictions, file_type='csv', append=True)

        else:
            st.info("Please upload a file to proceed with batch predictions.")
