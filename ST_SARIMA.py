import numpy as np
import pandas as pd
import pickle
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Create the ST_SARIMA_Results directory if it doesn't exist
if not os.path.exists('ST_SARIMA_Results'):
    os.makedirs('ST_SARIMA_Results')

# Load the hourly combined data from a CSV file, setting 'time' as the index with parsed dates
combined_data = pd.read_csv('Data/combined_data_hourly.csv', index_col='time', parse_dates=True)

# Load PEMS04 data to include additional neighboring sensors
pems_data = np.load('data/pems04.npz')['data']  # Shape: (16992, 307, 3)
# Create a 5-minute interval time index starting from 2018-01-01
start_date = pd.to_datetime('2018-01-01 00:00:00')
time_index_5min = pd.date_range(start=start_date, periods=len(pems_data), freq='5min')
pems_df = pd.DataFrame(index=time_index_5min)

# Select sensor 0 and additional neighboring sensors (e.g., first 5 neighbors)
sensor_ids = [0, 117, 92, 1, 2]  # Expanded to 5 sensors
for sid in sensor_ids:
    pems_df[f'flow_sensor_{sid}'] = pems_data[:, sid, 0]  # Extract flow data for each sensor

# Resample the 5-minute data to hourly intervals by taking the mean
pems_df_hourly = pems_df.resample('H').mean()

# Align the time index with combined_data
pems_df_hourly = pems_df_hourly.loc[combined_data.index]  # Match the index of combined_data

# Construct the dataset (use combined_data's flow as sensor 0's flow)
data = pd.DataFrame(index=combined_data.index)
data['flow_sensor_0'] = combined_data['flow']  # Use flow from combined_data for sensor 0
for sid in sensor_ids[1:]:  # Load other sensors from pems_df_hourly
    data[f'flow_sensor_{sid}'] = pems_df_hourly[f'flow_sensor_{sid}']

# Build the spatial weight matrix based on PEMS04.csv
adj_data = pd.read_csv('data/PEMS04.csv')
num_sensors = 307  # PEMS04 has 307 sensors
W = np.zeros((num_sensors, num_sensors), dtype=np.float32)  # Initialize adjacency matrix
for _, row in adj_data.iterrows():
    W[int(row['from']), int(row['to'])] = 1 / row['cost']  # Use inverse distance as weight
    W[int(row['to']), int(row['from'])] = 1 / row['cost']  # Ensure symmetry
W = W / (W.sum(axis=1, keepdims=True) + 1e-6)  # Normalize rows (add small epsilon to avoid division by zero)

# Extract the submatrix for selected sensors
W_subset = W[sensor_ids][:, sensor_ids]  # Subset for sensor_ids

# Define a function to create sequences for time series data (consistent with LSTM setup)
def create_sequences(data, seq_length=24, pred_length=1, train_ratio=0.8):
    X, y = [], []  # Initialize empty lists for input sequences (X) and target values (y)
    for i in range(len(data) - seq_length - pred_length):
        X.append(data[i:i+seq_length])  # Add sequence of length 'seq_length' to X
        y.append(data[i+seq_length, 0])  # Target is the flow of sensor 0
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    train_size = int(len(X) * train_ratio)  # Split data based on train_ratio
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test

# Generate sequences for training and testing
X_train, X_test, y_train, y_test = create_sequences(data.values)

# Prepare ST-SARIMA data
train_data = data.values[:len(y_train)]  # Training data (includes sensor 0 and neighbors)
test_data = data.values[len(y_train):len(y_train)+len(y_test)]  # Test data

# Fit SARIMA models separately for each sensor
sarima_models = []
for i in range(len(sensor_ids)):
    sensor_data = train_data[:, i]  # Flow data for sensor i
    sarima_model = SARIMAX(sensor_data, order=(2, 1, 2), seasonal_order=(1, 1, 1, 24))  # Define SARIMA model
    sarima_fit = sarima_model.fit(disp=False)  # Fit model, suppress convergence output
    sarima_models.append(sarima_fit)

# Generate predictions for each sensor
predictions = []
for i in range(len(sensor_ids)):
    sarima_pred = sarima_models[i].forecast(steps=len(test_data))  # Forecast for test period
    predictions.append(sarima_pred)
predictions = np.array(predictions)  # Shape: (num_sensors, len(test_data))

# Use spatial weight matrix to combine historical values for ST-SARIMA prediction
st_sarima_pred = np.zeros(len(test_data))  # Initialize prediction array
for t in range(len(test_data)):
    # Get historical values of neighboring sensors at current time step
    historical_values = test_data[t, :]  # Flow values for all sensors at time t
    weighted_historical = np.dot(W_subset[0, :], historical_values)  # Weighted historical value for sensor 0
    # Get SARIMA prediction for sensor 0
    sarima_pred_0 = predictions[0, t]  # SARIMA prediction for sensor 0 at time t
    # Combine SARIMA prediction (80%) and spatially weighted historical value (20%)
    st_sarima_pred[t] = 0.8 * sarima_pred_0 + 0.2 * weighted_historical

# Save the test data and ST-SARIMA predictions to a pickle file
with open('ST_SARIMA_Results/st_sarima_results.pkl', 'wb') as f:
    pickle.dump({'y_test': y_test, 'y_pred_st_sarima': st_sarima_pred}, f)

# Print confirmation message (note: directory name in message has a typo, should be ST_SARIMA_Results)
print("ST-SARIMA model results saved to 'ST_ARIMA_Results/st_sarima_results.pkl'")