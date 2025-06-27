import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class GeneradorSeriesTemporales:
    def __init__(self, start_date=datetime(2024, 1, 1), n_samples=10000, interval_minutes=10):
        """
        Inicializa el generador de series temporales.
        
        Args:
            start_date (datetime): Fecha de inicio de las series
            n_samples (int): Número de muestras a generar
            interval_minutes (int): Intervalo en minutos entre muestras
        """
        self.start_date = start_date
        self.n_samples = n_samples
        self.interval_minutes = interval_minutes
        np.random.seed(42)
        
    def _generar_indices_tiempo(self):
        """Genera el rango de tiempo para las series."""
        time_range = [self.start_date + timedelta(minutes=self.interval_minutes*i) 
                     for i in range(self.n_samples)]
        return pd.DatetimeIndex(time_range)
    
    def _calcular_componentes_periodicos(self, time_index):
        """Calcula los componentes periódicos para las series."""
        hora_del_dia = np.sin(2 * np.pi * time_index.hour / 24)
        dia_del_anio = np.sin(2 * np.pi * time_index.dayofyear / 365)
        return hora_del_dia, dia_del_anio
    
    def generar_series(self):
        """
        Genera las series temporales de presión, temperatura y viento.
        
        Returns:
            pd.DataFrame: DataFrame con las columnas ['tiempo', 'presion', 'temperatura', 'viento']
        """
        # Generar índices de tiempo
        time_index = self._generar_indices_tiempo()
        hora_del_dia, dia_del_anio = self._calcular_componentes_periodicos(time_index)
        
        # Generar presión
        presion = 1013 + \
                 5 * hora_del_dia + \
                 3 * dia_del_anio + \
                 0.1 * np.arange(len(time_index)) + \
                 np.random.normal(0, 1, len(time_index))
        
        # Generar temperatura
        temperatura = 25 + \
                     10 * hora_del_dia + \
                     7 * dia_del_anio + \
                     0.05 * np.arange(len(time_index)) + \
                     np.random.normal(0, 2, len(time_index))
        
        # Generar viento
        viento = 10 + \
                8 * hora_del_dia + \
                5 * dia_del_anio + \
                0.3 * (presion - np.mean(presion)) / np.std(presion) + \
                0.2 * (temperatura - np.mean(temperatura)) / np.std(temperatura) + \
                np.random.normal(0, 2, len(time_index))
        
        # Crear y retornar DataFrame
        return pd.DataFrame({
            'tiempo': time_index,
            'presion': presion,
            'temperatura': temperatura,
            'viento': viento
        })
    

        """
        Guarda las series temporales en un archivo CSV.
        
        Args:
            df (pd.DataFrame): DataFrame con las series temporales
            nombre_archivo (str): Nombre del archivo donde guardar las series
        """
        df.to_csv(nombre_archivo, index=False)
        
    def graficar_series(self, df):
        """
        Grafica las series temporales.
        
        Args:
            df (pd.DataFrame): DataFrame con las series temporales
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 12))
        
        plt.subplot(3, 1, 1)
        plt.plot(df['tiempo'], df['presion'])
        plt.title('Presión Atmosférica')
        plt.ylabel('hPa')
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(df['tiempo'], df['temperatura'])
        plt.title('Temperatura')
        plt.ylabel('°C')
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(df['tiempo'], df['viento'])
        plt.title('Velocidad del Viento')
        plt.ylabel('m/s')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
