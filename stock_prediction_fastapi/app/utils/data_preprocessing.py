import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data_from_csv(file_path: str) -> pd.DataFrame:
    """
    Load stock data from a CSV file.
    
    Args:
        file_path (str): The path to the CSV file.
    
    Returns:
        pd.DataFrame: A DataFrame containing the stock data.
    """
    try:
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'], format='%d%m%Y')  # Ensure the date format is correctly parsed
        df = df.sort_values('Date')
        return df
    except Exception as e:
        raise ValueError(f"Error loading data from {file_path}: {e}")

def preprocess_data(df: pd.DataFrame):
    """
    Preprocess the stock data by scaling the features.
    
    Args:
        df (pd.DataFrame): DataFrame containing stock data.
    
    Returns:
        tuple: Scaled data and the fitted scaler.
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])
    return scaled_data, scaler

def create_dataset(data: np.ndarray, time_step: int = 60) -> tuple:
    """
    Create datasets for the LSTM model.
    
    Args:
        data (np.ndarray): Scaled data.
        time_step (int): Number of time steps to look back for predictions.
    
    Returns:
        tuple: Features and labels for the model.
    """
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step, 3])  # Predicting the 'Close' price
    return np.array(X), np.array(y)

