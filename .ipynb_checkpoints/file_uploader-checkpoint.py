import streamlit as st
import pandas as pd

def upload_files():
    uploaded_files = st.file_uploader(
    'Choose a file (CSV or Excel)', 
    type=['csv', 'xlsx', 'xls'],  
    accept_multiple_files=True
)

    if uploaded_files:
        if len(uploaded_files) > 5:
            st.warning('You cannot upload more than 5 files')
            uploaded_files = uploaded_files[:5]

        # Load dataframes along with file names
        files_data = []
        for file in uploaded_files:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file)
            else:
                st.warning(f"Unsupported file type: {file.name}")
                continue
            files_data.append((file.name, df))

        return files_data
    
    else:
        st.info('No files uploaded')
        return None
