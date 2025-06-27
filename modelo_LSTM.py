import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class DatasetLSTM(Dataset):
    def __init__(self, df, n_lags=3):
        """
        Prepara el dataset para el modelo LSTM.
        
        Args:
            df (pd.DataFrame): DataFrame con las columnas ['presion', 'temperatura', 'viento']
            n_lags (int): Número de rezagos a utilizar
        """
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Preparar los datos
        data = df[['presion', 'temperatura', 'viento']].values
        X, y = [], []
        
        # Para cada punto en el tiempo t, tomamos los n_lags anteriores como features
        for i in range(n_lags, len(data)):
            # Tomamos los n_lags anteriores de presión, temperatura y viento
            X.append(data[i-n_lags:i, :])
            # La salida es el viento en t+1
            y.append(data[i, 2])  # índice 2 corresponde a 'viento'
            
        # Convertir a arrays de numpy
        X = np.array(X)
        y = np.array(y)
        
        # Reshape X para el escalado (2D)
        X_2d = X.reshape(-1, X.shape[-1])
        self.X = self.scaler_X.fit_transform(X_2d).reshape(X.shape)
        self.y = self.scaler_y.fit_transform(y.reshape(-1, 1)).reshape(-1)
        
        # Convertir a tensores de PyTorch
        self.X = torch.FloatTensor(self.X)
        self.y = torch.FloatTensor(self.y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    def inverse_transform_y(self, y):
        """Revierte la normalización de y."""
        return self.scaler_y.inverse_transform(y.reshape(-1, 1)).reshape(-1)

class ModeloLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size1=64, hidden_size2=32, num_layers=2):
        """
        Modelo LSTM con dos capas ocultas para predicción de viento.
        
        Args:
            input_size (int): Número de variables de entrada (presión, temperatura, viento)
            hidden_size1 (int): Número de unidades en la primera capa oculta
            hidden_size2 (int): Número de unidades en la segunda capa oculta
            num_layers (int): Número de capas LSTM apiladas
        """
        super(ModeloLSTM, self).__init__()
        
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.num_layers = num_layers
        
        # Primera capa LSTM
        self.lstm1 = nn.LSTM(input_size, hidden_size1, 
                           num_layers=1, batch_first=True)
        
        # Segunda capa LSTM
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, 
                           num_layers=1, batch_first=True)
        
        # Capa fully connected para la salida
        self.fc = nn.Linear(hidden_size2, 1)
        
        # Dropout para regularización
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Primera capa LSTM
        out1, _ = self.lstm1(x)
        out1 = self.dropout(out1)
        
        # Segunda capa LSTM
        out2, _ = self.lstm2(out1)
        out2 = self.dropout(out2)
        
        # Tomamos solo el último output para la predicción
        out = self.fc(out2[:, -1, :])
        return out.squeeze()

class EntrenadorLSTM:
    def __init__(self, model, learning_rate=0.001):
        """
        Manejador para entrenar el modelo LSTM.
        
        Args:
            model (ModeloLSTM): Instancia del modelo LSTM
            learning_rate (float): Tasa de aprendizaje para el optimizador
        """
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
    def train_epoch(self, train_loader):
        """Entrena el modelo por una época."""
        self.model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader):
        """Evalúa el modelo en el conjunto de validación."""
        self.model.eval()
        total_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()
                predictions.extend(outputs.numpy())
                actuals.extend(batch_y.numpy())
                
        return (total_loss / len(val_loader), 
                np.array(predictions), 
                np.array(actuals))
    
    def train(self, train_loader, val_loader, n_epochs, patience=5):
        """
        Entrena el modelo con early stopping.
        
        Args:
            train_loader: DataLoader con datos de entrenamiento
            val_loader: DataLoader con datos de validación
            n_epochs (int): Número máximo de épocas
            patience (int): Número de épocas a esperar antes de early stopping
        
        Returns:
            list: Lista con las pérdidas de entrenamiento
            list: Lista con las pérdidas de validación
        """
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, _, _ = self.evaluate(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f'Época {epoch+1}/{n_epochs}:')
            print(f'  Loss Entrenamiento: {train_loss:.4f}')
            print(f'  Loss Validación: {val_loss:.4f}')
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Guardar el mejor modelo
                torch.save(self.model.state_dict(), 'mejor_modelo_lstm.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print('Early stopping activado')
                    break
        
        return train_losses, val_losses
