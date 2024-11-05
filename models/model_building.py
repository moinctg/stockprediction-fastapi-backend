# train_model.py

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import joblib  # To save the scaler

# Load and preprocess your dataset
def load_data(file_path):
    # Load your dataset (make sure to adjust this according to your data)
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Example preprocessing (adjust based on your needs)
    prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    return scaled_prices, scaler

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

def build_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Main function to train the model
if __name__ == "__main__":
    file_path = 'DSE_STOCKS.csv'  # Update with your file path
    data = load_data(file_path)
    scaled_data, scaler = preprocess_data(data)
    
    # Set time step
    time_step = 5
    X, y = create_dataset(scaled_data, time_step)
    
    # Reshape input for LSTM [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    # Build and train the model
    model = build_model((X.shape[1], 1))
    model.fit(X, y, epochs=100, batch_size=32)
    
    # Save the model
    model.save('lstm_model.keras')  # Save in Keras format
    
    # Save the scaler
    joblib.dump(scaler, 'scaler.pkl')  # Save the scaler
