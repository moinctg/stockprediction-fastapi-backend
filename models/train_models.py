# train_models.py

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import statsmodels.api as sm
import os

# Define metrics calculation function
def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r_squared = r2_score(y_true, y_pred)
    return {"mse": mse, "mae": mae, "r_squared": r_squared}

# Load and preprocess data
def load_data(filename="DSE_STOCKS.csv"):
    data = pd.read_csv(filename)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    close_prices = data['Close'].values  # Assuming 'Close' column for stock prices
    return close_prices

# Train Linear Regression model
def train_linear_regression(X_train, y_train, X_test, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    pickle.dump(model, open("models/linear_regression.pkl", "wb"))
    return metrics

# Train Random Forest model
def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    pickle.dump(model, open("models/random_forest.pkl", "wb"))
    return metrics

# Train XGBoost model
def train_xgboost(X_train, y_train, X_test, y_test):
    model = XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred)
    pickle.dump(model, open("models/xgboost.pkl", "wb"))
    return metrics

# Train ARIMA model
def train_arima(close_prices):
    model = sm.tsa.ARIMA(close_prices, order=(1, 1, 1))
    model_fit = model.fit()
    future_predictions = model_fit.predict(start=len(close_prices), end=len(close_prices) + 5)
    model_fit.save("models/arima.pkl")
    return {"future_predictions": future_predictions.tolist()}

# Train LSTM model
def train_lstm(X_train, y_train, X_test, y_test):
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)
    y_pred = model.predict(X_test)
    metrics = calculate_metrics(y_test.flatten(), y_pred.flatten())
    model.save("models/lstm_model.h5")
    return metrics

# Main function to train all models
def train_all_models():
    # Load and preprocess the data
    close_prices = load_data()
    X = np.array(range(len(close_prices))).reshape(-1, 1)
    y = close_prices

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Directory to save models
    if not os.path.exists("models"):
        os.makedirs("models")

    # Train each model and store metrics
    metrics = {}

    # Train Linear Regression
    metrics["linear_regression"] = train_linear_regression(X_train, y_train, X_test, y_test)

    # Train Random Forest
    metrics["random_forest"] = train_random_forest(X_train, y_train, X_test, y_test)

    # Train XGBoost
    metrics["xgboost"] = train_xgboost(X_train, y_train, X_test, y_test)

    # Train ARIMA (doesn't require split data)
    metrics["arima"] = train_arima(close_prices)

    # Reshape data for LSTM model
    X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    metrics["lstm"] = train_lstm(X_train_lstm, y_train, X_test_lstm, y_test)

    # Save metrics
    with open("models/metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    print("Training complete. Models and metrics saved.")
    print("Model metrics:", metrics)

# Run the training script
if __name__ == "__main__":
    train_all_models()
