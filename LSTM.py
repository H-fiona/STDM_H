import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import pickle
import os
import gc

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create the LSTM_Results directory if it doesn't exist
if not os.path.exists('LSTM_Results'):
    os.makedirs('LSTM_Results')

# Load the hourly combined data from a CSV file, setting 'time' as the index with parsed dates
combined_data = pd.read_csv('Data/combined_data_hourly.csv', index_col='time', parse_dates=True)

# Select features for the model
features = ['flow', 'hour', 'day_of_week', 'is_holiday', 'neighbor_flow_1', 'sin_hour', 'cos_hour', 'sin_day_of_week', 'cos_day_of_week']
data = combined_data[features]

# Standardize the data using MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data).astype(np.float32)  # Fit and transform data to [0, 1] range
scaled_df = pd.DataFrame(scaled_data, columns=features, index=data.index)  # Create DataFrame with scaled data

# Save the scaler to a pickle file for later inverse transformation
with open('LSTM_Results/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Define a function to create sequences (past 24 hours to predict the next hour)
def create_sequences(data, seq_length=24, pred_length=1, train_ratio=0.8):
    X, y = [], []  # Initialize empty lists for input sequences (X) and target values (y)
    for i in range(len(data) - seq_length - pred_length):
        X.append(data[i:i+seq_length])  # Add sequence of length 'seq_length' to X
        y.append(data[i+seq_length, 0])  # Target is the flow of sensor 0 at the next time step
    X = np.array(X, dtype=np.float32)  # Convert to numpy array with float32 dtype
    y = np.array(y, dtype=np.float32)  # Convert to numpy array with float32 dtype
    train_size = int(len(X) * train_ratio)  # Split data based on train_ratio
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test

# Generate training and testing sequences
X_train, X_test, y_train, y_test = create_sequences(scaled_data)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")  # Print shapes for verification

# Free up memory by deleting temporary variables
del scaled_data
gc.collect()

# Build a simple LSTM model
model = Sequential([
    LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])),  # LSTM layer with 50 units
    Dropout(0.05),  # Dropout layer with 5% dropout rate to prevent overfitting
    Dense(1)  # Output layer with 1 unit for single value prediction
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')  # Compile model with Adam optimizer and MSE loss

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # Stop if no improvement after 10 epochs
history = model.fit(X_train, y_train, epochs=100, batch_size=32,  # Train for up to 100 epochs
                    validation_split=0.2, callbacks=[early_stopping])  # Use 20% of training data for validation

# Generate predictions on the test set
y_pred = model.predict(X_test)

# Save the test data and predictions to a pickle file
with open('LSTM_Results/lstm_results_hourly.pkl', 'wb') as f:
    pickle.dump({'y_test': y_test, 'y_pred': y_pred}, f)

# Print confirmation message
print("LSTM model results saved to 'LSTM_Results/lstm_results_hourly.pkl'")