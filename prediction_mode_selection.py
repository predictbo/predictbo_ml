# prediction_mode_selection.py
from missing_values_handler import handle_missing_values
from file_downloader import download_predictions
import streamlit as st
import numpy as np
import pandas as pd

def handle_prediction_mode(scaler, model, uploaded_files):
    prediction_mode = st.radio('How would you like to proceed?',
                               ('Single Prediction', 'Batch Prediction'))

    if prediction_mode == 'Single Prediction':
        st.header('Single Prediction')

        # Manually input values
        Rs = st.number_input('Rs', min_value=0.0, format='%.3f')
        T = st.number_input('T', min_value=0.0, format='%.3f')
        yo = st.number_input('yo', min_value=0.0, format='%.3f')
        Pb = st.number_input('Pb', min_value=0.0, format='%.3f')
        API = st.number_input('API', min_value=0.0, format='%.3f')
        yg = st.number_input('yg', min_value=0.0, format='%.3f')

        # Prepare the input for prediction
        input_data = np.array([[Rs, T, yo, Pb, API, yg]])

        # Preprocess the input data (scale the features)
        scaled_data = scaler.transform(input_data)

        # Predict button for single prediction
        if st.button("Predict"):
            # Make prediction using the loaded model
            prediction = model.predict(scaled_data)

            # Display the prediction result
            st.write(f'The predicted Bo value is: {prediction[0][0]:.4f}')

            # Option to download the prediction
            download_predictions(pd.DataFrame(input_data, columns=['Rs', 'T', 'yo', 'Pb', 'API', 'yg']),
                                 prediction, file_type='csv', append=True)

    elif prediction_mode == 'Batch Prediction':
        st.header('Batch Prediction')

        if uploaded_files:
            filenames = [uploaded_file.name for uploaded_file in uploaded_files]
            selected_file_name = st.selectbox("Select the CSV file to use", filenames)

            selected_file = next(file for file in uploaded_files if file.name == selected_file_name)

            # Reset the buffer to avoid EmptyDataError on re-use
            selected_file.seek(0)
            
            df = pd.read_csv(selected_file)

            st.write("Selected File Data Preview:")
            st.write(df.head())

            col_Rs = st.selectbox('Select column for Rs', df.columns)
            col_T = st.selectbox('Select column for T', df.columns)
            col_yo = st.selectbox('Select column for yo', df.columns)
            col_Pb = st.selectbox('Select column for Pb', df.columns)
            col_API = st.selectbox('Select column for API', df.columns)
            col_yg = st.selectbox('Select column for yg', df.columns)

            # Check for missing values in the selected columns and only show missing value handling if missing values exist
            if df[[col_Rs, col_T, col_yo, col_Pb, col_API, col_yg]].isnull().any().any():
                # Show missing value handling option if there are missing values
                df_cleaned = handle_missing_values(df)
            else:
                # If no missing values, use the original dataframe
                df_cleaned = df

            # Drop the target column 'Bo' if it's present
            df_cleaned = df_cleaned.drop(columns=['Bo'], errors='ignore')

            # Prepare the input for prediction
            input_data = df_cleaned[[col_Rs, col_T, col_yo, col_Pb, col_API, col_yg]].values

            # Preprocess the cleaned data (scale the features)
            scaled_data = scaler.transform(input_data)

            if st.button("Predict"):
                predictions = model.predict(scaled_data)
                
                st.write("Batch Predictions :")
                    
                # Convert predictions to a DataFrame for better display
                predictions_df = pd.DataFrame(predictions, columns=['Predicted Bo'])
                    
                # Display predictions in a better format (table with scroll)
                st.dataframe(predictions_df)
                
                # Option to download the predictions
                download_predictions(df_cleaned, predictions, file_type='csv', append=True)

        else:
            st.info("Please upload a CSV file to proceed with batch predictions.")
