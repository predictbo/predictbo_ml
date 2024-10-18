import pandas as pd
import streamlit as st
import numpy as np

def handle_missing_values(df):
    # Separate numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    missing_option = st.selectbox(
        'How would you like to handle missing values?',
        ('Delete rows with missing values', 'Impute with mean', 'Impute with median')
    )

    if missing_option == 'Delete rows with missing values':
        df_cleaned = df.dropna()
        st.write("Rows with missing values have been removed.")
    elif missing_option == 'Impute with mean':
        df_cleaned = df.fillna(numeric_df.mean())
        st.write("Missing values have been imputed with the mean.")
    elif missing_option == 'Impute with median':
        df_cleaned = df.fillna(numeric_df.median())
        st.write("Missing values have been imputed with the median.")
    
    return df_cleaned

