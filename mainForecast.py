import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from generar_series_temporales import GeneradorSeriesTemporales
from modelo_LSTM import DatasetLSTM, ModeloLSTM, EntrenadorLSTM
from torch.utils.data import random_split, DataLoader, Subset


# 1. Generar series temporales
print("1. Generando series temporales...")
generador = GeneradorSeriesTemporales()
df = generador.generar_series()

# 2. Preparar datos para LSTM
print("\n2. Preparando datos para LSTM...")
dataset = DatasetLSTM(df, n_lags=3)

# Dividir en conjuntos de entrenamiento y validación (80-20) manteniendo la secuencia temporal
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

# Crear índices secuenciales para mantener el orden temporal
indices = np.arange(len(dataset))
train_indices = indices[:train_size]
val_indices = indices[train_size:]


# Crear subconjuntos usando los índices

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
print(train_dataset)

# Crear dataloaders (sin shuffle en validación para mantener orden temporal)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # shuffle=False para mantener secuencia temporal
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Tamaño del conjunto de entrenamiento: {train_size}")
print(f"Tamaño del conjunto de validación: {val_size}")

# 3. Crear y entrenar el modelo LSTM
print("\n3. Creando y entrenando el modelo LSTM...")
modelo = ModeloLSTM(input_size=3, hidden_size1=64, hidden_size2=32)
entrenador = EntrenadorLSTM(modelo, learning_rate=0.001)

# Entrenar el modelo
n_epochs = 50
train_losses, val_losses = entrenador.train(train_loader, val_loader, n_epochs, patience=5)

# 4. Visualizar las pérdidas
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Entrenamiento')
plt.plot(val_losses, label='Validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida (MSE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Evaluar el modelo en el conjunto de validación
print("\n5. Evaluando el modelo...")
_, predicciones, reales = entrenador.evaluate(val_loader)

# Convertir predicciones y valores reales a la escala original
predicciones = dataset.inverse_transform_y(predicciones)
reales = dataset.inverse_transform_y(reales)

# Calcular métricas de error
mse = np.mean((predicciones - reales) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(predicciones - reales))
mape = np.mean(np.abs((reales - predicciones) / reales)) * 100
r2 = r2_score(reales, predicciones)
adj_r2 = 1 - (1 - r2) * (len(reales) - 1) / (len(reales) - 3 - 1)   # Ajuste de R2

print(f"\nMétricas de error en el conjunto de validación:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"MAPE: {mape:.4f}%")
print(f"R2: {r2:.4f}")
print(f"R2 ajustado: {adj_r2:.4f}")

# 6. Visualizar predicciones vs valores reales y correlación
# Obtener las fechas correspondientes al conjunto de validación
tiempo_validacion = df['tiempo'].values[-(len(predicciones)):]  # Tomamos las últimas fechas correspondientes a la validación

# Calcular correlación
corr = np.corrcoef(reales, predicciones)[0, 1]
print(f"\nCorrelación entre valores reales y predicciones: {corr:.4f}")

# Crear subplots para series temporales y dispersión
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# Gráfico de series temporales
ax1.plot(tiempo_validacion[:100], reales[:100], label='Real', alpha=0.7)
ax1.plot(tiempo_validacion[:100], predicciones[:100], label='Predicción', alpha=0.7)
ax1.set_title('Predicciones vs Valores Reales (Primeras 100 muestras)')
ax1.set_xlabel('Tiempo')
ax1.set_ylabel('Velocidad del Viento (m/s)')
ax1.legend()
ax1.grid(True)
ax1.tick_params(axis='x', rotation=45)

# Gráfico de dispersión
ax2.scatter(reales, predicciones, alpha=0.5)
ax2.plot([min(reales), max(reales)], [min(reales), max(reales)], 'r--', label='Ideal (y=x)')
ax2.set_title(f'Dispersión Real vs Predicción (R={corr:.4f})')
ax2.set_xlabel('Valores Reales (m/s)')
ax2.set_ylabel('Predicciones (m/s)')
ax2.legend()
ax2.grid(True)

# Ajustar layout
plt.tight_layout()
plt.show()
