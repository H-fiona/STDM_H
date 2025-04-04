import numpy as np
import pandas as pd
import pickle
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Create the SARIMA_Results directory if it doesn't exist
if not os.path.exists('SARIMA_Results'):
    os.makedirs('SARIMA_Results')

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

# Fit the SARIMA model with specified order and seasonal order (24-hour seasonality)
sarima_model = SARIMAX(train_data, order=(2, 1, 2), seasonal_order=(1, 1, 1, 24))
sarima_fit = sarima_model.fit(disp=False)  # Fit the model, suppressing convergence output

# Generate predictions for the test period
sarima_pred = sarima_fit.forecast(steps=len(test_data))

# Save the test data and predictions to a pickle file
with open('SARIMA_Results/sarima_results.pkl', 'wb') as f:
    pickle.dump({'y_test': test_data, 'y_pred_arima': sarima_pred}, f)

# Print confirmation message
print("SARIMA model results saved to 'SARIMA_Results/sarima_results.pkl'")