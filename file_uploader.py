import streamlit as st
import pandas as pd

def upload_files():
    uploaded_files = st.file_uploader('Choose a CSV file', type='csv', accept_multiple_files=True)
    
    if uploaded_files:
        if len(uploaded_files) > 5:
            st.warning('You cannot upload more than 5 files')
            uploaded_files = uploaded_files[:5]

        # Check if there is at least one file uploaded
        if len(uploaded_files) == 1:
            dataframes = [pd.read_csv(uploaded_files[0])]
        else:
            dataframes = [pd.read_csv(file) for file in uploaded_files]
        
        return dataframes
    
    else:
        st.info('No files uploaded')
        return None
