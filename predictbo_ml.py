import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers
import pickle

# Load dataset
dataset = pd.read_csv('Rashidi.csv')

# Separate features and target
X = dataset.drop(columns=['Bo'])  # Replace 'Bo' with your target column name
y = dataset['Bo']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Building the ANN model
model = tf.keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),  # Adjust input shape to match the number of features
    layers.Dense(32, activation='relu'),      # Hidden layer with 32 neurons and ReLU activation
    layers.Dense(1)                           # Output layer with a single neuron (for regression)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_scaled, y_train, epochs=100, batch_size=5, validation_split=0.20, verbose=0)

# Predict on test set
y_pred = model.predict(X_test_scaled)

# Calculate R2 and MSE
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f'R² score: {r2}')
print(f'Mean Squared Error: {mse}')

# Save the model using pickle
model_pickle = open('ann_bo_model.pickle', 'wb')
pickle.dump(model, model_pickle)
model_pickle.close()

# Save the scaler (important for preprocessing during predictions)
scaler_pickle = open('scaler.pickle', 'wb')
pickle.dump(scaler, scaler_pickle)
scaler_pickle.close()
