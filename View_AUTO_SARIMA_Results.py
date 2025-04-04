import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the hourly combined data from a CSV file, setting 'time' as the index with parsed dates
combined_data = pd.read_csv('Data/combined_data_hourly.csv', index_col='time', parse_dates=True)

# Load S-ARIMA (AUTO.ARIMA) results from a pickle file
with open('AUTO_SARIMA_Results/sarima_auto_results.pkl', 'rb') as f:
    sarima_auto_results = pickle.load(f)

# Extract true test values and predicted values from the loaded results
y_test = sarima_auto_results['y_test']
y_pred_sarima_auto = sarima_auto_results['y_pred_sarima_auto']

# Use original values directly (ARIMA predictions are not standardized)
y_test_inv = y_test
y_pred_sarima_auto_inv = y_pred_sarima_auto

# Calculate the mean of the test set
test_mean = y_test_inv.mean()
print(f"Test set mean (vehicles/hour): {test_mean:.2f}")

# Print the first 10 true and predicted values for inspection
print("First 10 true values (vehicles/hour):", y_test_inv[:10])
print("First 10 predicted values (vehicles/hour):", y_pred_sarima_auto_inv[:10])

# Calculate error metrics: Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE)
rmse_sarima_auto = np.sqrt(mean_squared_error(y_test_inv, y_pred_sarima_auto_inv))
mae_sarima_auto = mean_absolute_error(y_test_inv, y_pred_sarima_auto_inv)
print(f"S-ARIMA(AUTO.ARIMA) - RMSE: {rmse_sarima_auto:.2f}, MAE: {mae_sarima_auto:.2f}")

# Calculate error ratios as percentages of the test mean
rmse_ratio = (rmse_sarima_auto / test_mean) * 100
mae_ratio = (mae_sarima_auto / test_mean) * 100
print(f"RMSE Ratio: {rmse_ratio:.2f}%")
print(f"MAE Ratio: {mae_ratio:.2f}%")

# Visualize the prediction results for the first 200 test points
plt.figure(figsize=(12, 6))  # Set figure size to 12x6 inches
plt.plot(y_test_inv[:200], label='True', color='black')  # Plot true values in black
plt.plot(y_pred_sarima_auto_inv[:200], label='Predicted (S-ARIMA(AUTO.ARIMA))', color='#1f77b4', linestyle='--')  # Plot predictions in blue dashed line
plt.title('S-ARIMA(AUTO.ARIMA) Traffic Flow Prediction (First 200 Test Points)')  # Set plot title
plt.xlabel('Time Step (Hourly)')  # Label x-axis
plt.ylabel('Traffic Flow (vehicles/hour)')  # Label y-axis
plt.legend()  # Add legend
plt.grid(True)  # Add grid
plt.savefig('AUTO_SARIMA_Results/sarima_auto_prediction_hourly.png', dpi=300, bbox_inches='tight')  # Save figure as PNG
plt.show()  # Display the plot

# Plot the error distribution
error = y_test_inv - y_pred_sarima_auto_inv  # Calculate prediction errors
plt.figure(figsize=(8, 6))  # Set figure size to 8x6 inches
plt.hist(error, bins=50, color='#1f77b4', alpha=0.7)  # Create histogram with 50 bins, blue color, and 70% opacity
plt.title('Error Distribution (S-ARIMA(AUTO.ARIMA))')  # Set plot title
plt.xlabel('Prediction Error (vehicles/hour)')  # Label x-axis
plt.ylabel('Frequency')  # Label y-axis
plt.grid(True)  # Add grid
plt.savefig('AUTO_SARIMA_Results/sarima_auto_error_distribution_hourly.png', dpi=300, bbox_inches='tight')  # Save figure as PNG
plt.show()  # Display the plot