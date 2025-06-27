# Forecasting_LSTM

This program performs time series forecasting using an LSTM (Long Short-Term Memory) model on a simulated univariate dataset. 
It applies a supervised learning approach by transforming the time series into input-output pairs based on a specified window size. 
The model is trained to predict the next value in the sequence, making it a one-step-ahead forecast. This setup is particularly 
useful for capturing temporal dependencies and trends within the data. The simulation allows for controlled experimentation and 
evaluation of the LSTM’s ability to learn and generalize from sequential patterns in univariate time series forecasting tasks.

# Synthetic Time Series Data Generator
The file generar_series_temporales.py defines a Python class for generating synthetic time series data for meteorological variables: 
atmospheric pressure, temperature, and wind speed. 

The generator creates realistic data by mimicking daily and yearly cycles, adding trends, and incorporating random noise.
