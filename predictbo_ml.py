import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras import Sequential
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load datasets
experimental_data = pd.read_csv("EXPERIMENTAL_DATA_RENAMED.csv")
rashidi_data = pd.read_csv("Rashidi.csv")

# Define features and target
features = [
    "Solution Gas Oil Ratio, (SCF/STB)", 
    "Temperature, (F)", 
    "API, (-)", 
    "Bubble Point Pressure, (Psi)"
]
target = "Oil Formation Volume Factor, (bbl/STB)"

# Prepare datasets
experimental_data = experimental_data[features + ["Bo"]].rename(columns={"Bo": target}).dropna()
rashidi_data = rashidi_data[features + [target]].dropna()

# Combine datasets
combined_data = pd.concat([experimental_data, rashidi_data], ignore_index=True)

# Split features and target
X = combined_data[features].values
y = combined_data[target].values

# Custom transformer for discretization
class Discretizer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_indices, n_bins=100):
        self.feature_indices = feature_indices
        self.discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    
    def fit(self, X, y=None):
        for idx in self.feature_indices:
            self.discretizer.fit(X[:, idx].reshape(-1, 1))
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for idx in self.feature_indices:
            X_transformed[:, idx] = self.discretizer.transform(X[:, idx].reshape(-1, 1)).flatten()
        return X_transformed

# Define the pipeline
pipeline = Pipeline([
    ('discretizer', Discretizer(feature_indices=[0, 1, 3])),
    ('scaler', StandardScaler()),
])

# K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Prepare a function to create the model
def create_model():
    model = Sequential([
        layers.Input(shape=(X.shape[1],)),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-6)),
        layers.Dense(137, activation='relu', kernel_regularizer=regularizers.l2(1e-6)),
        layers.Dense(163, activation='relu', kernel_regularizer=regularizers.l2(1e-6)),
        layers.Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])
    return model

# Train the model using K-Fold
for train_index, val_index in kf.split(X):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Preprocess the data
    X_train_transformed = pipeline.fit_transform(X_train, y_train)
    X_val_transformed = pipeline.transform(X_val)

    # Create and train the model
    model = create_model()
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

    history = model.fit(
        X_train_transformed, y_train,
        validation_data=(X_val_transformed, y_val),
        epochs=1000,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

# Save the preprocessing pipeline
joblib.dump(pipeline, 'preprocessing_pipeline.pkl')

# Save the trained Keras model
model.save('trained_model.h5')

print("Pipeline and model saved successfully!")