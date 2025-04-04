import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import seaborn as sns
from datetime import datetime, timedelta
import os

# Create the EDA_Results directory if it doesn't exist
if not os.path.exists('EDA_Results'):
    os.makedirs('EDA_Results')

# Load PEMS04 data from a .npz file
pems_data = np.load('data/pems04.npz')['data']  # Shape: (16992, 307, 3)
sensor_id = 0  # Select the first sensor (sensor 0)
flow_data = pems_data[:, sensor_id, 0]  # Extract flow data for sensor 0

# Generate timestamps (every 5 minutes)
start_time = datetime(2018, 1, 1, 0, 0)  # Starting time: January 1, 2018, 00:00
time_steps = len(flow_data)  # Total number of time steps: 16992
time_interval = timedelta(minutes=5)  # 5-minute intervals
timestamps = [start_time + i * time_interval for i in range(time_steps)]  # Create list of timestamps
time_index = pd.DatetimeIndex(timestamps)  # Convert to pandas DatetimeIndex

# Plot the time series for the first day (zoomed in)
plt.figure(figsize=(8, 6))  # Set figure size to 8x6 inches
plt.plot(time_index[:288], flow_data[:288], label='Traffic Flow (Sensor 0, First Day)', color='#1f77b4')  # Plot first 288 points (1 day)
plt.title('Traffic Flow - First Day (2018-01-01)')  # Set plot title
plt.xlabel('Time')  # Label x-axis
plt.ylabel('Flow')  # Label y-axis
plt.legend()  # Add legend
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=6))  # Set x-axis ticks every 6 hours
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Format x-axis as HH:MM
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.grid(True, linestyle='--', alpha=0.7)  # Add dashed grid with 70% opacity
plt.tight_layout()  # Adjust layout to prevent clipping
plt.savefig('EDA_Results/traffic_flow_first_day.png', dpi=300, bbox_inches='tight')  # Save figure as PNG
plt.show()  # Display the plot

# Plot the average hourly traffic flow for the first week
flow_df = pd.DataFrame(flow_data, index=time_index, columns=['flow'])  # Create DataFrame with flow data
flow_weekly = flow_df[:288*7].resample('H').mean()  # Resample to hourly data for the first 7 days (288*7 = 2016 points)
plt.figure(figsize=(12, 6))  # Set figure size to 12x6 inches
plt.plot(flow_weekly.index, flow_weekly['flow'], label='Hourly Average Traffic Flow (First Week)', color='#1f77b4')  # Plot hourly flow
plt.title('Hourly Average Traffic Flow - First Week (2018-01-01 to 2018-01-07)')  # Set plot title
plt.xlabel('Time')  # Label x-axis
plt.ylabel('Flow')  # Label y-axis
plt.legend()  # Add legend
plt.xticks(rotation=45)  # Rotate x-axis labels for readability
plt.grid(True, linestyle='--', alpha=0.7)  # Add dashed grid with 70% opacity
plt.tight_layout()  # Adjust layout to prevent clipping
plt.savefig('EDA_Results/traffic_flow_weekly.png', dpi=300, bbox_inches='tight')  # Save figure as PNG
plt.show()  # Display the plot

# Perform time series decomposition (7 days of hourly data)
flow_df = pd.DataFrame(flow_data[:288*7], index=time_index[:288*7], columns=['flow'])  # DataFrame for first 7 days
flow_hourly = flow_df.resample('H').mean()  # Resample to hourly intervals
decomposition = seasonal_decompose(flow_hourly['flow'], model='additive', period=24)  # Decompose with 24-hour seasonality

# Manually plot decomposition components with adjusted line width
fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)  # Create 4 subplots with shared x-axis
axes[0].plot(decomposition.observed, color='#1f77b4', linewidth=1)  # Plot observed data
axes[0].set_title('Observed')  # Set subplot title
axes[0].grid(True, linestyle='--', alpha=0.7)  # Add grid
axes[1].plot(decomposition.trend, color='#1f77b4', linewidth=1)  # Plot trend component
axes[1].set_title('Trend')  # Set subplot title
axes[1].grid(True, linestyle='--', alpha=0.7)  # Add grid
axes[2].plot(decomposition.seasonal, color='#1f77b4', linewidth=1)  # Plot seasonal component
axes[2].set_title('Seasonal')  # Set subplot title
axes[2].grid(True, linestyle='--', alpha=0.7)  # Add grid
axes[3].plot(decomposition.resid, color='#1f77b4', linewidth=1)  # Plot residual component
axes[3].set_title('Residual')  # Set subplot title
axes[3].grid(True, linestyle='--', alpha=0.7)  # Add grid
axes[3].xaxis.set_major_locator(mdates.DayLocator(interval=1))  # Set x-axis ticks every day
axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Format x-axis as YYYY-MM-DD
plt.xticks(rotation=45)  # Rotate x-axis labels
plt.tight_layout()  # Adjust layout
plt.savefig('EDA_Results/decomposition_plot.png', dpi=300, bbox_inches='tight')  # Save figure as PNG
plt.show()  # Display the plot

# Plot a heatmap of traffic flow for the first week (first 10 sensors)
flow_data_week1 = pems_data[:288*7, :10, 0]  # First week data for the first 10 sensors (2016 time steps, 10 sensors)
plt.figure(figsize=(12, 8))  # Set figure size to 12x8 inches
sns.heatmap(flow_data_week1.T, cmap='viridis', cbar_kws={'label': 'Flow'})  # Create heatmap with viridis colormap
plt.title('Traffic Flow Heatmap (First Week, First 10 Sensors)')  # Set plot title
plt.xlabel('Time (5-min intervals)')  # Label x-axis
plt.ylabel('Sensor ID')  # Label y-axis
plt.savefig('EDA_Results/flow_heatmap.png', dpi=300, bbox_inches='tight')  # Save figure as PNG
plt.show()  # Display the plot

# Check stationarity of the data using the Augmented Dickey-Fuller test
result = adfuller(flow_data)
print('ADF Statistic:', result[0])  # Print ADF statistic
print('p-value:', result[1])  # Print p-value
if result[1] > 0.05:  # If p-value > 0.05, data may not be stationary
    print("Data may not be stationary, applying differencing")
    flow_diff = np.diff(flow_data)  # Apply first-order differencing
else:
    print("Data is stationary, no differencing needed")
    flow_diff = flow_data  # Use original data if stationary

# Set lags for 24 hours and 48 hours (in 5-minute intervals)
lag_24h = 288  # 24 hours = 288 * 5 minutes
lag_48h = 576  # 48 hours = 576 * 5 minutes

# Calculate ACF values for 24-hour and 48-hour lags
acf_24h = acf(flow_diff, nlags=lag_24h, fft=True)  # ACF for 24 hours
acf_48h = acf(flow_diff, nlags=lag_48h, fft=True)  # ACF for 48 hours

# Manually calculate the 95% confidence interval (fixed value)
n = len(flow_diff)  # Sample size: 16992 (16991 after differencing)
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
plt.title('Autocorrelation of Flow (24 Hours)')  # Set plot title
plt.xlabel('Lag (5-min intervals)')  # Label x-axis
plt.ylabel('ACF')  # Label y-axis
plt.legend()  # Add legend
plt.grid(True, linestyle='--', alpha=0.7)  # Add dashed grid with 70% opacity
plt.tight_layout()  # Adjust layout
plt.savefig('EDA_Results/acf_24h_plot.png', dpi=300, bbox_inches='tight')  # Save figure as PNG
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
plt.gca().set_xticks(range(0, lag_48h + 1, 72))  # Set x-axis ticks every 6 hours (72 * 5 min = 6 hours)
plt.title('Autocorrelation of Flow (48 Hours)')  # Set plot title
plt.xlabel('Lag (5-min intervals)')  # Label x-axis
plt.ylabel('ACF')  # Label y-axis
plt.legend()  # Add legend
plt.grid(True, linestyle='--', alpha=0.7)  # Add dashed grid with 70% opacity
plt.tight_layout()  # Adjust layout
plt.savefig('EDA_Results/acf_48h_plot.png', dpi=300, bbox_inches='tight')  # Save figure as PNG
plt.show()  # Display the plot

# Calculate PACF values for the first 50 lags
lags = 50
pacf_values = pacf(flow_diff, nlags=lags, method='ols')  # Compute PACF using OLS method

# Manually calculate the 95% confidence interval (fixed value)
n = len(flow_diff)  # Sample size: 16992 (16991 after differencing)
ci = 1.96 * np.sqrt(1 / n)  # 95% confidence interval

# Plot the PACF with significant lags highlighted
plt.figure(figsize=(8, 6))  # Set figure size to 8x6 inches
markerline, stemlines, baseline = plt.stem(range(len(pacf_values)), pacf_values, linefmt='-', markerfmt='o', basefmt='r-')  # Create stem plot
plt.setp(markerline, color='#1f77b4')  # Set marker color to blue
plt.setp(stemlines, color='#1f77b4')   # Set stem line color to blue
plt.setp(baseline, color='red')        # Set baseline color to red
for i in range(len(pacf_values)):  # Highlight significant lags
    if pacf_values[i] > ci or pacf_values[i] < -ci:
        plt.plot(i, pacf_values[i], 'ro')  # Red circle for significant values
plt.axhline(y=ci, color='black', linestyle='--', label='95% Confidence Interval')  # Upper CI line
plt.axhline(y=-ci, color='black', linestyle='--')  # Lower CI line
plt.axhline(y=0, color='black', linestyle='--')  # Zero line
plt.title('Partial Autocorrelation of Flow (First 50 Lags)')  # Set plot title
plt.xlabel('Lag (5-min intervals)')  # Label x-axis
plt.ylabel('PACF')  # Label y-axis
plt.legend()  # Add legend
plt.grid(True, linestyle='--', alpha=0.7)  # Add dashed grid with 70% opacity
plt.tight_layout()  # Adjust layout
plt.savefig('EDA_Results/pacf_plot.png', dpi=300, bbox_inches='tight')  # Save figure as PNG
plt.show()  # Display the plot

# Print the manually calculated confidence interval
print(f"Manual Confidence Interval (fixed): ±{ci:.3f}")

# Calculate ACF values and confidence intervals for 24-hour lag with statistical method
acf_values_24h, confint_24h = acf(flow_diff, nlags=lag_24h, alpha=0.05, fft=True)  # ACF with 95% CI
print(f"ACF at lag 288 (24 hours): {acf_values_24h[288]:.4f}")  # Print ACF value at 24-hour lag
print(f"Confidence interval at lag 288: {confint_24h[288]}")  # Print CI at 24-hour lag

# Manually set a threshold for significant lags
threshold = 0.8
significant_lags_manual = np.where(acf_values_24h > threshold)[0]  # Find lags exceeding threshold
print(f"Manual significant lags (threshold {threshold}): {significant_lags_manual}")

# Save results to a text file
with open('EDA_Results/acf_results.txt', 'w') as f:
    f.write(f"ADF Statistic: {result[0]}\n")  # Write ADF statistic
    f.write(f"p-value: {result[1]}\n")  # Write p-value
    f.write("Data is stationary, no differencing needed\n")  # Write stationarity conclusion
    f.write(f"ACF at lag 288 (24 hours): {acf_values_24h[288]:.4f}\n")  # Write ACF at 24-hour lag
    f.write(f"Confidence interval at lag 288: {confint_24h[288]}\n")  # Write CI at 24-hour lag
    f.write(f"Manual significant lags (threshold {threshold}): {significant_lags_manual}\n")  # Write significant lags