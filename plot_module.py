import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt  # Import matplotlib for plotting

def generate_plots(uploaded_files):
    """
    Function to handle the selection of an uploaded file and generate scatter or box plots.
    Uses Altair for scatter plot and seaborn for box plot.

    Parameters:
    uploaded_files: List of uploaded CSV files
    """

    # Ask user if they want to generate a plot
    generate_plot = st.radio("Do you want to generate a plot?", ("No", "Yes"), key="generate_plot")

    if generate_plot == "Yes":
        st.title("Generate Plot")  # Section title

        if uploaded_files:
            filenames = [uploaded_file.name for uploaded_file in uploaded_files]
            selected_file_name = st.selectbox("Select the CSV file to use", filenames, key="plot_file_select")

            selected_file = next(file for file in uploaded_files if file.name == selected_file_name)

            # Reset the buffer to avoid EmptyDataError on re-use
            selected_file.seek(0)

            df = pd.read_csv(selected_file)

            st.write("Selected File Data Preview:")
            st.write(df.head())

            # Choose plot type
            plot_type = st.selectbox("Select plot type", ["Scatter Plot (Altair)", "Box Plot"], key="plot_type_select")

            if plot_type == "Scatter Plot (Altair)":
                st.write("Select features for scatter plot:")

                selected_x_feature = st.selectbox('What do you want the x feature to be?', df.columns, key="x_feature_scatter")
                selected_y_feature = st.selectbox('What do you want the y feature to be?', df.columns, key="y_feature_scatter")

                if st.button("Generate Scatter Plot"):
                    # Dynamic title for the scatter plot
                    scatter_title = f"Scatter plot of {selected_y_feature} vs {selected_x_feature}"
                    alt_chart = (
                        alt.Chart(df, title=scatter_title)
                        .mark_circle()
                        .encode(
                            x=selected_x_feature,
                            y=selected_y_feature,
                            color="species" if 'species' in df.columns else alt.value("blue"),
                        )
                        .interactive()
                    )
                    st.altair_chart(alt_chart, use_container_width=True)

            elif plot_type == "Box Plot":
                st.write("Select feature for box plot:")
                selected_feature = st.selectbox('Feature', df.columns, key="box_feature")

                if st.button("Generate Box Plot"):
                    # Dynamic title for the box plot
                    box_title = f"Box plot of {selected_feature}"
                    fig, ax = plt.subplots()
                    ax.set_title(box_title)  # Set the dynamic title
                    sns.boxplot(x=df[selected_feature], ax=ax)
                    st.pyplot(fig)

        else:
            st.info("Please upload a CSV file to proceed with plot generation.")

