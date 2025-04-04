import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf

# Load the hourly combined data from a CSV file, setting 'time' as the index with parsed dates
combined_data = pd.read_csv('Data/combined_data_hourly.csv', index_col='time', parse_dates=True)

# Select features (only using 'flow' data in this case)
features = ['flow']
data = combined_data[features]

# Define a function to create sequences for time series data (consistent with LSTM setup)
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

# Generate sequences, but only use y_train and y_test (X_train and X_test not used here)
_, _, y_train, y_test = create_sequences(data.values)

# Prepare ARIMA data: use raw 'flow' data for training
train_data = data['flow'].values[:len(y_train)]

# Perform first-order differencing (d=1)
diff_data = np.diff(train_data, n=1)

# Perform seasonal differencing (D=1, s=24) to account for 24-hour seasonality
seasonal_diff_data = diff_data[24:] - diff_data[:-24]

# Calculate ACF values for 24-hour and 48-hour lags
lag_24h = 24  # 24-hour lag
lag_48h = 48  # 48-hour lag
acf_24h = acf(seasonal_diff_data, nlags=lag_24h, fft=True)  # ACF for 24 hours
acf_48h = acf(seasonal_diff_data, nlags=lag_48h, fft=True)  # ACF for 48 hours

# Manually calculate the 95% confidence interval (fixed value)
n = len(seasonal_diff_data)  # Sample size after differencing
ci = 1.96 * np.sqrt(1 / n)  # 95% confidence interval based on standard normal distribution

# Plot the 24-hour ACF
plt.figure(figsize=(8, 6))  # Set figure size to 8x6 inches
markerline, stemlines, baseline = plt.stem(range(len(acf_24h)), acf_24h, linefmt='-', markerfmt='o', basefmt='r-')  # Create stem plot
plt.setp(markerline, color='#1f77b4')  # Set marker color to blue
plt.setp(stemlines, color='#1f77b4')   # Set stem line color to blue
plt.setp(baseline, color='red')        # Set baseline color to red
plt.axhline(y=ci, color='black', linestyle='--', label='95% Confidence Interval')  # Upper CI line
plt.axhline(y=-ci, color='black', linestyle='--')  # Lower CI line
plt.axhline(y=0, color='black', linestyle='--')  # Zero line
plt.title('Autocorrelation of Flow (24 Hours, Differenced)')  # Set plot title
plt.xlabel('Lag (Hourly intervals)')  # Label x-axis
plt.ylabel('ACF')  # Label y-axis
plt.legend()  # Add legend
plt.grid(True, linestyle='--', alpha=0.7)  # Add dashed grid with 70% opacity
plt.tight_layout()  # Adjust layout to prevent clipping
plt.savefig('EDA_Results/acf_24h_differenced_plot.png', dpi=300, bbox_inches='tight')  # Save figure as PNG
plt.show()  # Display the plot

# Plot the 48-hour ACF
plt.figure(figsize=(8, 6))  # Set figure size to 8x6 inches
markerline, stemlines, baseline = plt.stem(range(len(acf_48h)), acf_48h, linefmt='-', markerfmt='o', basefmt='r-')  # Create stem plot
plt.setp(markerline, color='#1f77b4')  # Set marker color to blue
plt.setp(stemlines, color='#1f77b4')   # Set stem line color to blue
plt.setp(baseline, color='red')        # Set baseline color to red
plt.axhline(y=ci, color='black', linestyle='--', label='95% Confidence Interval')  # Upper CI line
plt.axhline(y=-ci, color='black', linestyle='--')  # Lower CI line
plt.axhline(y=0, color='black', linestyle='--')  # Zero line
plt.gca().set_xticks(range(0, lag_48h + 1, 6))  # Set x-axis ticks every 6 hours
plt.title('Autocorrelation of Flow (48 Hours, Differenced)')  # Set plot title
plt.xlabel('Lag (Hourly intervals)')  # Label x-axis
plt.ylabel('ACF')  # Label y-axis
plt.legend()  # Add legend
plt.grid(True, linestyle='--', alpha=0.7)  # Add dashed grid with 70% opacity
plt.tight_layout()  # Adjust layout to prevent clipping
plt.savefig('EDA_Results/acf_48h_differenced_plot.png', dpi=300, bbox_inches='tight')  # Save figure as PNG
plt.show()  # Display the plot

# Calculate PACF values for the first 50 lags
lags = 50
pacf_values = pacf(seasonal_diff_data, nlags=lags, method='ols')  # Use OLS method for PACF

# Manually calculate the 95% confidence interval (fixed value)
n = len(seasonal_diff_data)  # Sample size after differencing
ci = 1.96 * np.sqrt(1 / n)  # 95% confidence interval

# Plot the PACF with significant lags highlighted
plt.figure(figsize=(8, 6))  # Set figure size to 8x6 inches
markerline, stemlines, baseline = plt.stem(range(len(pacf_values)), pacf_values, linefmt='-', markerfmt='o', basefmt='r-')  # Create stem plot
plt.setp(markerline, color='#1f77b4')  # Set marker color to blue
plt.setp(stemlines, color='#1f77b4')   # Set stem line color to blue
plt.setp(baseline, color='red')        # Set baseline color to red
# Highlight significant lags (outside the confidence interval) with red markers
for i in range(len(pacf_values)):
    if pacf_values[i] > ci or pacf_values[i] < -ci:
        plt.plot(i, pacf_values[i], 'ro')  # Red circle for significant values
plt.axhline(y=ci, color='black', linestyle='--', label='95% Confidence Interval')  # Upper CI line
plt.axhline(y=-ci, color='black', linestyle='--')  # Lower CI line
plt.axhline(y=0, color='black', linestyle='--')  # Zero line
plt.title('Partial Autocorrelation of Flow (First 50 Lags, Differenced)')  # Set plot title
plt.xlabel('Lag (Hourly intervals)')  # Label x-axis
plt.ylabel('PACF')  # Label y-axis
plt.legend()  # Add legend
plt.grid(True, linestyle='--', alpha=0.7)  # Add dashed grid with 70% opacity
plt.tight_layout()  # Adjust layout to prevent clipping
plt.savefig('EDA_Results/pacf_differenced_plot.png', dpi=300, bbox_inches='tight')  # Save figure as PNG
plt.show()  # Display the plot