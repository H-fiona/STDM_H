import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os

if not os.path.exists('Comparison_Results'):
    os.makedirs('Comparison_Results')

# Load prediction results from all models
with open('AUTO_SARIMA_Results/sarima_auto_results.pkl', 'rb') as f:
    sarima_auto_results = pickle.load(f)  # Load S-ARIMA(AUTO.ARIMA) results
with open('SARIMA_Results/sarima_results.pkl', 'rb') as f:
    sarima_results = pickle.load(f)  # Load S-ARIMA(CUSTOMIZE) results
with open('ST_SARIMA_Results/st_sarima_results.pkl', 'rb') as f:
    st_sarima_results = pickle.load(f)  # Load ST-SARIMA results
with open('LSTM_Results/lstm_results_hourly.pkl', 'rb') as f:
    lstm_results = pickle.load(f)  # Load LSTM results
with open('ST_LSTM_Results/st_lstm_corrected_results.pkl', 'rb') as f:
    st_lstm_results = pickle.load(f)  # Load ST-LSTM results

# Extract true values and predicted values for the first 200 test points
y_test = sarima_auto_results['y_test'][:200]  # True values (first 200 test points)
y_pred_sarima_auto = sarima_auto_results['y_pred_sarima_auto'][:200]  # S-ARIMA(AUTO.ARIMA) predictions
y_pred_sarima = sarima_results['y_pred_arima'][:200]  # S-ARIMA(CUSTOMIZE) predictions
y_pred_st_sarima = st_sarima_results['y_pred_st_sarima'][:200]  # ST-SARIMA predictions
y_pred_lstm = lstm_results['y_pred'][:200]  # LSTM predictions
y_pred_st_lstm = st_lstm_results['y_pred_st_lstm'][:200]  # ST-LSTM predictions

# Plot a comparison of all models
plt.figure(figsize=(12, 6))  # Set figure size to 12x6 inches
plt.plot(y_test, label='True', color='black')  # Plot true values in black
plt.plot(y_pred_sarima_auto, label='S-ARIMA(AUTO.ARIMA)', linestyle='--')  # Plot S-ARIMA(AUTO.ARIMA) predictions as dashed line
plt.plot(y_pred_sarima, label='S-ARIMA(CUSTOMIZE)', linestyle='--')  # Plot S-ARIMA(CUSTOMIZE) predictions as dashed line
plt.plot(y_pred_st_sarima, label='ST-SARIMA', linestyle='--')  # Plot ST-SARIMA predictions as dashed line
plt.plot(y_pred_lstm, label='LSTM', linestyle='--')  # Plot LSTM predictions as dashed line
plt.plot(y_pred_st_lstm, label='ST-LSTM', linestyle='--')  # Plot ST-LSTM predictions as dashed line
plt.xlabel('Time Step (Hourly)')  # Label x-axis
plt.ylabel('Traffic Flow (vehicles/hour)')  # Label y-axis
plt.title('Model Comparison (First 200 Test Points)')  # Set plot title
plt.legend()  # Add legend
plt.grid(True)  # Add grid
plt.savefig('Comparison_Results/model_comparison_hourly.png', dpi=300, bbox_inches='tight')  # Save figure as PNG
plt.show()  # Display the plot