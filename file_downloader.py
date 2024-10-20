# file_downloader.py
import streamlit as st
import pandas as pd
from io import StringIO

def download_predictions(df, predictions, file_type='csv', append=True):
    """
    Append predictions to the original dataset or create a new file with only predictions.
    
    Args:
    df (pd.DataFrame): Original dataframe.
    predictions (np.ndarray): Model predictions.
    file_type (str): File type to download ('csv' or 'txt').
    append (bool): If True, append predictions to original dataset.
    
    Returns:
    Streamlit download button.
    """
    # Convert predictions to a DataFrame
    predictions_df = pd.DataFrame(predictions, columns=['Predicted_Bo'])

    if append:
        # Append predictions to the original dataframe
        result_df = pd.concat([df.reset_index(drop=True), predictions_df], axis=1)
    else:
        # Only download the predictions
        result_df = predictions_df

    # Convert DataFrame to CSV or TXT format based on user choice
    if file_type == 'csv':
        csv_data = result_df.to_csv(index=False)
        st.download_button(label="Download CSV", data=csv_data, file_name="predictions.csv", mime="text/csv")
    elif file_type == 'txt':
        txt_data = result_df.to_string(index=False)
        st.download_button(label="Download TXT", data=txt_data, file_name="predictions.txt", mime="text/plain")
