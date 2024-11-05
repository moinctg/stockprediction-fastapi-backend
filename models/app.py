# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from statsmodels.tsa.arima.model import ARIMAResults
import pandas as pd
import os

# Initialize FastAPI app
app = FastAPI()

# Load all models and metrics
with open("models/metrics.pkl", "rb") as f:
    metrics_data = pickle.load(f)

linear_regression_model = pickle.load(open("models/linear_regression.pkl", "rb"))
random_forest_model = pickle.load(open("models/random_forest.pkl", "rb"))
xgboost_model = pickle.load(open("models/xgboost.pkl", "rb"))
lstm_model = load_model("models/lstm_model.h5")
arima_model = ARIMAResults.load("models/arima.pkl")

# Define the request and response formats
class PredictionRequest(BaseModel):
    scrip: str
    historical_data: List[float]
    model_type: str  # "linear_regression", "random_forest", "xgboost", "lstm", or "arima"

class PredictionResponse(BaseModel):
    scrip: str
    current_price: float
    future_predictions: List[float]
    metrics: dict

# Helper functions for each model's prediction logic
def predict_linear_regression(data):
    data = np.array(data).reshape(-1, 1)
    predictions = linear_regression_model.predict(data)
    return predictions[-5:].tolist()  # Returning last 5 predictions as future predictions

def predict_random_forest(data):
    data = np.array(data).reshape(-1, 1)
    predictions = random_forest_model.predict(data)
    return predictions[-5:].tolist()

def predict_xgboost(data):
    data = np.array(data).reshape(-1, 1)
    predictions = xgboost_model.predict(data)
    return predictions[-5:].tolist()

def predict_lstm(data):
    data = np.array(data).reshape((1, len(data), 1))
    predictions = lstm_model.predict(data)
    return predictions.flatten()[-5:].tolist()

def predict_arima(data):
    predictions = arima_model.predict(start=len(data), end=len(data) + 4)  # Predicting 5 steps ahead
    return predictions.tolist()

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_stock(request: PredictionRequest):
    model_type = request.model_type.lower()
    historical_data = request.historical_data
    if len(historical_data) < 5:
        raise HTTPException(status_code=400, detail="Insufficient data. Provide at least 5 data points.")

    # Get predictions based on model type
    if model_type == "linear_regression":
        predictions = predict_linear_regression(historical_data)
        metrics = metrics_data.get("linear_regression")
    elif model_type == "random_forest":
        predictions = predict_random_forest(historical_data)
        metrics = metrics_data.get("random_forest")
    elif model_type == "xgboost":
        predictions = predict_xgboost(historical_data)
        metrics = metrics_data.get("xgboost")
    elif model_type == "lstm":
        predictions = predict_lstm(historical_data)
        metrics = metrics_data.get("lstm")
    elif model_type == "arima":
        predictions = predict_arima(historical_data)
        metrics = metrics_data.get("arima")
    else:
        raise HTTPException(status_code=400, detail="Invalid model_type. Choose from linear_regression, random_forest, xgboost, lstm, or arima.")

    response = {
        "scrip": request.scrip,
        "current_price": historical_data[-1],
        "future_predictions": predictions,
        "metrics": metrics
    }
    return response

# Health check endpoint
@app.get("/")
async def root():
    return {"message": "Stock Prediction API is running"}
