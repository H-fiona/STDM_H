import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import pickle
import os
import gc

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create the ST_LSTM_Results directory if it doesn't exist
if not os.path.exists('ST_LSTM_Results'):
    os.makedirs('ST_LSTM_Results')

# Load the hourly combined data from a CSV file, setting 'time' as the index with parsed dates
combined_data = pd.read_csv('Data/combined_data_hourly.csv', index_col='time', parse_dates=True)

# Select features (including flow for sensors 0 and 117, and temporal features)
features = ['flow', 'neighbor_flow_1', 'hour', 'day_of_week', 'is_holiday', 'sin_hour', 'cos_hour', 'sin_day_of_week', 'cos_day_of_week']
data = combined_data[features]

# Standardize the data using MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data).astype(np.float32)  # Fit and transform data to [0, 1] range
scaled_df = pd.DataFrame(scaled_data, columns=features, index=data.index)  # Create DataFrame with scaled data

# Save the scaler to a pickle file for later inverse transformation
with open('ST_LSTM_Results/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Define sequence parameters (past 24 hours to predict the next hour)
seq_length = 24  # 24 hours of historical data
pred_length = 1  # Predict 1 hour ahead
train_ratio = 0.8  # 80% of data for training

# Define a function to create sequences for time series data
def create_sequences(data, seq_length=24, pred_length=1, train_ratio=0.8):
    X, y_0, y_1 = [], [], []  # X: input sequences, y_0: future flow for sensor 0, y_1: future flow for sensor 117
    for i in range(len(data) - seq_length - pred_length):
        X.append(data[i:i+seq_length])  # Past 24 hours
        y_0.append(data[i+seq_length, 0])  # Future 1-hour flow for sensor 0
        y_1.append(data[i+seq_length, 1])  # Future 1-hour flow for sensor 117
    X = np.array(X, dtype=np.float32)  # Convert to numpy array with float32 dtype
    y_0 = np.array(y_0, dtype=np.float32)  # Convert sensor 0 targets to numpy array
    y_1 = np.array(y_1, dtype=np.float32)  # Convert sensor 117 targets to numpy array
    train_size = int(len(X) * train_ratio)  # Split data based on train_ratio
    X_train, X_test = X[:train_size], X[train_size:]
    y_0_train, y_0_test = y_0[:train_size], y_0[train_size:]
    y_1_train, y_1_test = y_1[:train_size], y_1[train_size:]
    return X_train, X_test, y_0_train, y_0_test, y_1_train, y_1_test

# Generate training and testing sequences
X_train, X_test, y_0_train, y_0_test, y_1_train, y_1_test = create_sequences(scaled_data, seq_length=seq_length, pred_length=pred_length, train_ratio=train_ratio)
print(f"X_train shape: {X_train.shape}, y_0_train shape: {y_0_train.shape}, y_1_train shape: {y_1_train.shape}")  # Print shapes for verification

# Free up memory by deleting temporary variables
del scaled_data
gc.collect()

# Build the spatial weight matrix based on PEMS04.csv
adj_data = pd.read_csv('data/PEMS04.csv')
num_sensors = 307  # PEMS04 has 307 sensors
W = np.zeros((num_sensors, num_sensors), dtype=np.float32)  # Initialize adjacency matrix
for _, row in adj_data.iterrows():
    W[int(row['from']), int(row['to'])] = 1 / row['cost']  # Use inverse distance as weight
    W[int(row['to']), int(row['from'])] = 1 / row['cost']  # Ensure symmetry
W = W / (W.sum(axis=1, keepdims=True) + 1e-6)  # Normalize rows (add small epsilon to avoid division by zero)

# Select sensors 0 and 117 for spatial weighting
sensor_ids = [0, 117]
W_subset = W[sensor_ids][:, sensor_ids]  # Extract submatrix for selected sensors

# Build ST-LSTM models for each sensor
lstm_models = []
for i in range(len(sensor_ids)):
    model = Sequential([
        LSTM(250, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),  # First LSTM layer with 250 units
        Dropout(0.05),  # Dropout layer with 5% dropout rate
        LSTM(200, return_sequences=True),  # Second LSTM layer with 200 units
        Dropout(0.05),  # Dropout layer with 5% dropout rate
        LSTM(150),  # Third LSTM layer with 150 units
        Dropout(0.05),  # Dropout layer with 5% dropout rate
        Dense(1)  # Output layer with 1 unit for single value prediction
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')  # Compile model with Adam optimizer and MSE loss
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)  # Early stopping callback
    # Select the correct target values for training
    y_train_sensor = y_0_train if i == 0 else y_1_train  # Use sensor 0 or 117 targets
    y_test_sensor = y_0_test if i == 0 else y_1_test
    history = model.fit(X_train, y_train_sensor, epochs=100, batch_size=32,  # Train for up to 100 epochs
                        validation_split=0.2, callbacks=[early_stopping])  # Use 20% of training data for validation
    lstm_models.append(model)

# Generate predictions for each sensor
predictions = []
for i in range(len(sensor_ids)):
    y_pred_sensor = lstm_models[i].predict(X_test)  # Predict using the trained model
    predictions.append(y_pred_sensor)
predictions = np.array(predictions)  # Shape: (num_sensors, len(test_data), 1)

# Use spatial weight matrix to combine predictions (adjust weighting)
st_lstm_pred = np.zeros(len(y_0_test))  # Initialize prediction array for sensor 0
for t in range(len(y_0_test)):
    weighted_pred = np.dot(W_subset[0, :], predictions[:, t, 0])  # Weighted prediction for sensor 0
    sarima_pred_0 = predictions[0, t, 0]  # Direct LSTM prediction for sensor 0
    # Combine LSTM prediction (95%) and spatially weighted prediction (5%)
    st_lstm_pred[t] = 0.95 * sarima_pred_0 + 0.05 * weighted_pred

# Save the test data and ST-LSTM predictions to a pickle file
with open('ST_LSTM_Results/st_lstm_corrected_results.pkl', 'wb') as f:
    pickle.dump({'y_test': y_0_test, 'y_pred_st_lstm': st_lstm_pred}, f)

# Print confirmation message
print("Corrected ST-LSTM (hourly) model results saved to 'ST_LSTM_Results/st_lstm_corrected_results.pkl'")