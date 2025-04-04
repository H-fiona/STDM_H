import numpy as np
import pandas as pd
import pickle
import os
from pmdarima import auto_arima

# Create the AUTO_SARIMA_Results directory if it doesn't exist
if not os.path.exists('AUTO_SARIMA_Results'):
    os.makedirs('AUTO_SARIMA_Results')

# Load the hourly combined data from a CSV file, setting 'time' as the index with parsed dates
combined_data = pd.read_csv('Data/combined_data_hourly.csv', index_col='time', parse_dates=True)

# Select features (only using 'flow' data in this case)
features = ['flow']
data = combined_data[features]

# Define a function to create sequences for time series data
def create_sequences(data, seq_length=24, pred_length=1, train_ratio=0.8):
    X, y = [], []  # Initialize empty lists for input sequences (X) and target values (y)
    # Loop through the data to create sequences of specified length
    for i in range(len(data) - seq_length - pred_length):
        X.append(data[i:i+seq_length])  # Add sequence of length 'seq_length' to X
        y.append(data[i+seq_length, 0])  # Add the next value (flow) as the target
    # Convert lists to numpy arrays with float32 dtype
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    # Split data into training and testing sets based on the train_ratio
    train_size = int(len(X) * train_ratio)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test

# Generate sequences, but only use y_train and y_test for SARIMA (X_train and X_test not used here)
_, _, y_train, y_test = create_sequences(data.values)

# Prepare data for SARIMA: split raw 'flow' data into training and testing sets
train_data = data['flow'].values[:len(y_train)]  # Training data aligned with y_train length
test_data = data['flow'].values[len(y_train):len(y_train)+len(y_test)]  # Test data aligned with y_test

# Use auto_arima to automatically select the best SARIMA model
sarima_model = auto_arima(train_data,
                          seasonal=True, m=24,  # Enable seasonality with a 24-hour cycle
                          start_p=0, start_q=0, max_p=3, max_q=3,  # Range for non-seasonal AR and MA orders
                          start_P=0, start_Q=0, max_P=2, max_Q=2,  # Range for seasonal AR and MA orders
                          d=1, D=1,  # Non-seasonal and seasonal differencing orders
                          trace=True,  # Print model selection process
                          error_action='ignore',  # Ignore errors during model fitting
                          suppress_warnings=True)  # Suppress warnings for cleaner output

# Fit the selected SARIMA model to the training data
sarima_fit = sarima_model.fit(train_data)

# Generate predictions for the test period
sarima_pred = sarima_fit.predict(n_periods=len(test_data))

# Save the test data and predictions to a pickle file
with open('AUTO_SARIMA_Results/sarima_auto_results.pkl', 'wb') as f:
    pickle.dump({'y_test': test_data, 'y_pred_sarima_auto': sarima_pred}, f)

# Print confirmation message
print("S-ARIMA(AUTO.ARIMA) model results saved to 'AUTO_SARIMA_Results/sarima_auto_results.pkl'")