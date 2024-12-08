import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt

@st.cache_data
def combine_data_with_predictions(df, predictions, feature_columns):
    """
    Combine features and predictions into a single DataFrame.

    Parameters:
    df: Original DataFrame containing uploaded file data
    predictions: Predicted Bo values (Array or DataFrame)
    feature_columns: List of columns selected as features for the prediction
    """
    # Extract selected feature columns from DataFrame
    features = df[feature_columns]

    if isinstance(predictions, pd.DataFrame):
        predictions_df = predictions.rename(columns={0: 'Predicted Bo'})
    else:
        predictions_df = pd.DataFrame(predictions, columns=['Predicted Bo'])

    return pd.concat([features.reset_index(drop=True), predictions_df.reset_index(drop=True)], axis=1)


def generate_post_prediction_plots_with_files(uploaded_files, predictions, feature_columns):
    """
    Generate plots using uploaded files and predictions.

    Parameters:
    uploaded_files: List of uploaded CSV files
    predictions: Array or DataFrame containing predicted Bo values
    feature_columns: List of selected feature columns
    """
    st.title("Post-Prediction Analysis")

    if uploaded_files and predictions is not None:
        filenames = [uploaded_file.name for uploaded_file in uploaded_files]
        selected_file_name = st.selectbox("Select the CSV file to use", filenames, key="post_plot_file_select")

        # Locate and reset the selected file pointer
        selected_file = next(file for file in uploaded_files if file.name == selected_file_name)
        selected_file.seek(0)

        # Read the CSV into a DataFrame
        df = pd.read_csv(selected_file)

        st.write("Selected File Data Preview:")
        st.write(df.head())

        # Combine the uploaded data and predictions for visualization
        combined_df = combine_data_with_predictions(df, predictions, feature_columns)

        st.write("Combined Data with Predictions:")
        st.write(combined_df.head())

        # Select a feature for plotting against Predicted Bo
        selected_feature = st.selectbox("Select a feature for the scatter plot", feature_columns, key="feature_prediction_plot")

        # Generate scatter plot
        if st.button("Generate Scatter Plot with Predictions"):
            scatter_title = f"Scatter plot of Predicted Bo vs {selected_feature}"
            scatter_plot = (
                alt.Chart(combined_df, title=scatter_title)
                .mark_circle(size=60)
                .encode(
                    x=alt.X(selected_feature, title=f"{selected_feature}"),
                    y=alt.Y('Predicted Bo', title="Predicted Bo"),
                    tooltip=[selected_feature, 'Predicted Bo']
                )
                .interactive()
            )
            st.altair_chart(scatter_plot, use_container_width=True)

        # Generate box plot
        if st.button("Generate Box Plot for Predictions"):
            fig, ax = plt.subplots()
            sns.boxplot(x=combined_df['Predicted Bo'], ax=ax)
            ax.set_title("Box Plot of Predicted Bo")
            st.pyplot(fig)

    else:
        st.info("Please upload a CSV file and ensure predictions are made before generating plots.")
