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

# LSTM model

The file modelo_LSTM.py defines the complete workflow for building, training, and evaluating an LSTM (Long Short-Term Memory) neural network for time series forecasting, specifically to predict wind based on pressure, temperature, and wind history. Here’s a breakdown of what each main part does:

DatasetLSTM (Lines 8–53):

Prepares the input data for the LSTM model.
Takes a DataFrame with columns ['presion', 'temperatura', 'viento'] and creates sequences of lagged features for supervised learning.
Normalizes the features and targets using sklearn’s StandardScaler.
Implements the necessary methods for use with PyTorch’s DataLoader, including the ability to reverse the normalization of predictions.
ModeloLSTM (Lines 54–97):

Defines an LSTM neural network using PyTorch.
The model has two LSTM layers (stacked), each followed by dropout for regularization.
The output of the second LSTM layer is passed through a fully connected (linear) layer to generate the final prediction (wind at the next time step).
The forward method ensures only the last time step’s output is used for the prediction.
EntrenadorLSTM (Lines 98–186):

Handles training and evaluation of the LSTM model.
Uses Mean Squared Error (MSE) as the loss function and Adam as the optimizer.
Implements methods to train for one epoch, evaluate on a validation set, and a full training loop with early stopping (stops if validation loss doesn’t improve for several epochs).
Saves the best model based on validation loss
