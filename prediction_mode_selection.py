import streamlit as st
import numpy as np
import pandas as pd

def handle_prediction_mode(scaler, model, uploaded_files):
    prediction_mode = st.radio('How would you like to proceed?',
                               ('Single Prediction', 'Batch Prediction'))

    if prediction_mode == 'Single Prediction':
        st.header('Single Prediction')

        # manually input values
        Rs = st.number_input('Rs', min_value=0.0, format='%.3f')
        T = st.number_input('T', min_value=0.0, format='%.3f')
        yo = st.number_input('yo', min_value=0.0, format='%.3f')
        Pb = st.number_input('P', min_value=0.0, format='%.3f')
        API = st.number_input('API', min_value=0.0, format='%.3f')
        yg = st.number_input('yg', min_value=0.0, format='%.3f')

        # Prepare the input for prediction
        input_data = np.array([[Rs, T, yo, Pb, API, yg]])

        # Preprocess the input data (scale the features)
        scaled_data = scaler.transform(input_data)

        # Make prediction using the loaded model
        prediction = model.predict(scaled_data)

        # Display the prediction result
        st.write(f'The predicted Bo value is: {prediction[0][0]:.4f}')

    elif prediction_mode == 'Batch Prediction' and uploaded_files:
        st.header('Batch Prediction')

        # Display the file names of uploaded files
        filenames = [uploaded_file.name for uploaded_file in uploaded_files]

        # Let the user select which uploaded CSV file they want to use
        selected_file_name = st.selectbox("Select the CSV file to use", filenames)

        # Find the selected file's corresponding data
        selected_file = next(file for file in uploaded_files if file.name == selected_file_name)

        # Read the selected CSV file into a DataFrame
        df = pd.read_csv(selected_file)

        # Display a preview of the selected CSV file
        st.write("Selected File Data Preview:")
        st.write(df.head())

        # Dropdowns to select columns from the uploaded file
        col_Rs = st.selectbox('Select column for Rs', df.columns)
        col_T = st.selectbox('Select column for T', df.columns)
        col_yo = st.selectbox('Select column for yo', df.columns)
        col_Pb = st.selectbox('Select column for Pb', df.columns)
        col_API = st.selectbox('Select column for API', df.columns)
        col_yg = st.selectbox('Select column for yg', df.columns)

        # Extract selected columns for prediction
        Rs = df[col_Rs].values
        T = df[col_T].values
        yo = df[col_yo].values
        Pb = df[col_Pb].values
        API = df[col_API].values
        yg = df[col_yg].values

        # Prepare the input for prediction
        input_data = np.column_stack((Rs, T, yo, Pb, API, yg))

        # Preprocess the input data (scale the features)
        scaled_data = scaler.transform(input_data)

        # Make batch predictions using the loaded model
        predictions = model.predict(scaled_data)

        # Display the batch predictions
        st.write("Batch Predictions:")
        st.write(predictions)

    else:
        st.info("Please upload a CSV file to proceed with batch predictions.")
