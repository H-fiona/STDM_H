Traffic Flow Prediction with PeMS04: A Comparative Study of Spatio-Temporal Models

Overview
This project focuses on short-term traffic flow prediction using the PeMS04 dataset, covering hourly traffic flow data from January 1, 2018, to February 28, 2018. The study compares five models: S-ARIMA(AUTO.ARIMA), S-ARIMA(CUSTOMIZE), ST-SARIMA, LSTM, and ST-LSTM, to predict the hourly traffic flow of sensor 0 over a one-hour horizon. The project includes exploratory data analysis (EDA), model implementation, and performance comparison, with a particular emphasis on manual tuning to optimize S-ARIMA performance and the application of spatio-temporal models (ST-SARIMA and ST-LSTM) to capture spatial dependencies.

Environment Setup
Prerequisites
- Python 3.8 or higher
- Git (for cloning the repository)

Dependencies
Install the required packages using the following command in your terminal:
pip install -r requirements.txt

The requirements.txt file includes:
pandas
numpy
pmdarima
statsmodels
tensorflow==2.19.0
matplotlib
seaborn

Data
The dataset used in this project is the PeMS04 dataset, sourced from Kaggle (Elmahy, 2022). It can be downloaded from:
https://www.kaggle.com/datasets/elmahy/pems-dataset

Preprocessing
1. Download the PeMS04 dataset from the above link.
2. Place the PEMS04.csv file in the data/ directory.
3. Run the preprocessing script (included in pems04_process.py) to generate Data/combined_data_hourly.csv, which contains hourly traffic flow data for sensor 0 and sensor 117, along with added time features (hour, day_of_week, etc.).

Running Instructions
1. Exploratory Data Analysis (EDA)
Run the EDA script to generate ACF, PACF, decomposition, and other plots:
python pems04_spatial.py
python pems04_temporal.py
- Outputs are saved in EDA_Results/.

2. S-ARIMA(AUTO.ARIMA)
Train and evaluate the S-ARIMA(AUTO.ARIMA) model:
python AUTO_SARIMA.py
- Results are saved in AUTO_SARIMA_Results/.

3. S-ARIMA(CUSTOMIZE)
Train and evaluate the S-ARIMA(CUSTOMIZE) model:
python SARIMA.py
- Results are saved in SARIMA_Results/.

4. ST-SARIMA, LSTM, and ST-LSTM
Train and evaluate ST-SARIMA, LSTM, and ST-LSTM models:
python ST_LSTM.py
python LSTM.py
python ST_ARIMA.py
- Results are saved in ST_SARIMA_Results/, LSTM_Results/, and ST_LSTM_Results/.

5. Model Comparison
Generate comparison plots and tables:
python Compare_Results.py
- Outputs are saved in Comparison_Results/.

Results
Model Performance
Model              | RMSE  | MAE   | RMSE Ratio (%) | MAE Ratio (%)
------------------|-------|-------|----------------|---------------
S-ARIMA(AUTO.ARIMA)| 65.95 | 40.13 | 26.20          | 15.94
S-ARIMA(CUSTOMIZE) | 55.79 | 38.70 | 22.16          | 15.37
ST-SARIMA          | 53.27 | 38.05 | 21.07          | 15.05
LSTM               | 24.87 | 18.68 | 9.84           | 7.39
ST-LSTM            | 28.15 | 19.21 | 11.13          | 7.60


The generated results are in their respective Results folders.