{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dd31bad9",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import pickle\n",
    "import numpy as np\n",
    "\n",
    "# Set the title of the app\n",
    "st.title('Predict Bo')\n",
    "\n",
    "# Provide a description of the app\n",
    "st.write(\"This app uses 6 inputs to predict the value of Bo using an ANN model. Use the form below to get started!\")\n",
    "\n",
    "# Load the ANN model and scaler\n",
    "with open('ann_bo_model.pickle', 'rb') as model_pickle:\n",
    "    model = pickle.load(model_pickle)\n",
    "\n",
    "with open('scaler.pickle', 'rb') as scaler_pickle:\n",
    "    scaler = pickle.load(scaler_pickle)\n",
    "\n",
    "# Optional: Display the loaded model and scaler for verification\n",
    "st.write(\"Loaded ANN Model:\")\n",
    "st.write(model)\n",
    "st.write(\"Loaded Scaler:\")\n",
    "st.write(scaler)\n",
    "\n",
    "# Input fields for user to input data in Streamlit\n",
    "st.header(\"Bo Prediction\")\n",
    "\n",
    "# Taking user inputs for the features\n",
    "Rs = st.number_input('Solution Gas Oil Ratio (SCF/STB)', min_value=0.0, format=\"%.2f\")\n",
    "T = st.number_input('Temperature (°F)', min_value=0.0, format=\"%.2f\")\n",
    "yg = st.number_input('Gas Specific Gravity (-)', min_value=0.0, format=\"%.2f\")\n",
    "API = st.number_input('API (-)', min_value=0.0, format=\"%.2f\")\n",
    "yo = st.number_input('Oil Specific Gravity (-)', min_value=0.0, format=\"%.2f\")\n",
    "Pb = st.number_input('Bubble Point Pressure (Psi)', min_value=0.0, format=\"%.2f\")\n",
    "\n",
    "# Convert inputs into a NumPy array\n",
    "input_data = np.array([[Rs, T, yg, API, yo, Pb]])\n",
    "\n",
    "# Preprocess the input data (scale the features)\n",
    "scaled_data = scaler.transform(input_data)\n",
    "\n",
    "# Make prediction using the loaded model\n",
    "prediction = model.predict(scaled_data)\n",
    "\n",
    "# Display the prediction result\n",
    "st.write(f'The predicted Bo value is: {prediction[0][0]:.4f}')\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
