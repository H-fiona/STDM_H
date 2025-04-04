import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timedelta
import os

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

# Create a DataFrame with flow data and timestamps
flow_df = pd.DataFrame(flow_data, index=time_index, columns=['flow'])

# Add neighboring sensor flow data (Sensor 117)
neighbor_id_1 = 117  # First neighboring sensor
neighbor_flow_1 = pems_data[:, neighbor_id_1, 0]  # Extract flow data for sensor 117
flow_df['neighbor_flow_1'] = neighbor_flow_1  # Add to DataFrame

# Uncomment to add a second neighboring sensor (Sensor 92)
# neighbor_id_2 = 92  # Second neighboring sensor
# neighbor_flow_2 = pems_data[:, neighbor_id_2, 0]  # Extract flow data for sensor 92
# flow_df['neighbor_flow_2'] = neighbor_flow_2  # Add to DataFrame

# Resample to hourly intervals by taking the mean
flow_hourly = flow_df.resample('H').mean()

# Add temporal features based on hourly data
flow_hourly['hour'] = flow_hourly.index.hour  # Hour of the day (0-23)
flow_hourly['day_of_week'] = flow_hourly.index.dayofweek  # Day of the week (0-6, Monday=0)
flow_hourly['is_holiday'] = 0  # Initialize holiday flag as 0 (non-holiday)
# Mark January 1, 2018, as a holiday
flow_hourly.loc[flow_hourly.index.date == pd.to_datetime('2018-01-01').date(), 'is_holiday'] = 1

# Add periodic encoding for cyclical features (optional, to be analyzed later)
flow_hourly['sin_hour'] = np.sin(2 * np.pi * flow_hourly['hour'] / 24)  # Sine of hour for 24-hour cycle
flow_hourly['cos_hour'] = np.cos(2 * np.pi * flow_hourly['hour'] / 24)  # Cosine of hour for 24-hour cycle
flow_hourly['sin_day_of_week'] = np.sin(2 * np.pi * flow_hourly['day_of_week'] / 7)  # Sine of day for 7-day cycle
flow_hourly['cos_day_of_week'] = np.cos(2 * np.pi * flow_hourly['day_of_week'] / 7)  # Cosine of day for 7-day cycle

# Save the aligned hourly data to a CSV file
flow_hourly.to_csv('Data/combined_data_hourly.csv', index_label='time')

# Print confirmation message and display the first few rows of the DataFrame
print("Hourly combined data saved to 'Data/combined_data_hourly.csv'")
print(flow_hourly.head())


combined_data = pd.read_csv('Data/combined_data_hourly.csv', index_col='time', parse_dates=True)
mean_flow = combined_data['flow'].mean()
std_flow = combined_data['flow'].std()
print(f"Mean flow: {mean_flow:.2f} vehicles/hour")
print(f"Standard deviation: {std_flow:.2f} vehicles/hour")