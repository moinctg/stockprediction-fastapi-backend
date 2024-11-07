# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import numpy as np
import pandas as pd
import tensorflow as tf
import uvicorn
import threading
import traceback
import logging
import statistics
import joblib 
from typing import List, Optional
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from statsmodels.tsa.arima.model import ARIMAResults
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

import os

app = FastAPI()


# # # # Load model and scaler
# model = load_model("lstm_model.keras")
# scaler = joblib.load("scaler.pkl")

# # Request body structure
# class PredictionRequestLSTM(BaseModel):
#     scrip: str
#     data: list[float]
#     days: list[int]
#     actual_prices: list[float] = None  # Optional for evaluation

# # Prediction endpoint
# @app.post("/predict_api")
# def predict_stock(pr: PredictionRequestLSTM):
#     data_scaled = scaler.transform(np.array(pr.data).reshape(-1, 1))
#     input_sequence = data_scaled[-3:].reshape(1, 3, 1)

#     predicted_prices = []
#     for day in pr.days:
#         pred = model.predict(input_sequence)
#         predicted_prices.append(pred[0, 0])
#         input_sequence = np.append(input_sequence[:, 1:, :], [[pred]], axis=1)

#     # Inverse scale predictions
#     predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1)).flatten()

#     # Calculate metrics if actual_prices provided
#     evaluation = {}
#     if pr.actual_prices:
#         if len(pr.actual_prices) == len(predicted_prices):
#             mse = np.mean((predicted_prices - pr.actual_prices) ** 2)
#             mae = np.mean(np.abs(predicted_prices - pr.actual_prices))
#             evaluation = {"mse": mse, "mae": mae}
#         else:
#             raise HTTPException(status_code=400, detail="The length of predicted prices and actual prices must match.")

#     # Calculate mean, max, standard deviation
#     metrics = {
#         "mean": float(np.mean(predicted_prices)),
#         "max": float(np.max(predicted_prices)),
#         "standard_deviation": float(np.std(predicted_prices))
#     }

#     return {
#         "scrip": pr.scrip,
#         "current_price": pr.data[-1],
#         "future_predictions": {
#             "days": pr.days,
#             "predicted_prices": predicted_prices.tolist()
#         },
#         "metrics": metrics,
#         "evaluation": evaluation
#     }

# Define the origins that should be allowed to access your API
# origins = [
#     "http://localhost:3000",  # For local development
#     "https://stockinsightai.vercel.app",  # Replace with your actual frontend domain
# ]

# # Add CORS middleware to your app
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,  # List of allowed origins
#     allow_credentials=True,  # Allow cookies and headers to be sent with requests
#     allow_methods=["*"],     # Allow all HTTP methods
#     allow_headers=["*"],     # Allow all headers
# )


# logging.basicConfig(level=logging.DEBUG)















def dummy_model_predict(input_data, days):
    if not input_data:
        raise ValueError("Empty input data")

    current_price = input_data[-1]  # Last known price as the current price
    future_predictions = [(current_price + day * 0.5) for day in days]  # Dummy logic for prediction

    return current_price, future_predictions

# Example prediction request model
class PredictRequest(BaseModel):
    scrip: str
    data: list[float]
    days: list[int]
    actual_prices: list[float]  # Include actual prices for evaluation

# Setup FastAPI and CORS Middleware
# app = FastAPI()

origins = [
    "http://localhost:3000",  # For local development
    "https://stockinsightai.vercel.app",  # Replace with your actual frontend domain
]

# Add CORS middleware to your app
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,  # Allow cookies and headers to be sent with requests
    allow_methods=["*"],     # Allow all HTTP methods
    allow_headers=["*"],     # Allow all headers
)

logging.basicConfig(level=logging.DEBUG)

@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        # Log incoming request data for debugging
        logging.debug(f"Incoming request data: {request.scrip}, {request.data}, {request.days}, {request.actual_prices}")

        # Dummy prediction logic
        current_price, predicted_prices = dummy_model_predict(request.data, request.days)

        # Compute metrics for predicted prices
        mean_price = statistics.mean(predicted_prices)
        max_price = max(predicted_prices)

        # Check if there are enough data points to calculate standard deviation
        if len(predicted_prices) > 1:
            std_dev = statistics.stdev(predicted_prices)
        else:
            std_dev = 0.0  # If there is only one prediction, set stdev to 0

        # Calculate evaluation metrics (MSE, MAE) using actual prices
        if len(request.actual_prices) != len(predicted_prices):
            raise ValueError("Length of actual prices and predicted prices must be the same")

        mse = mean_squared_error(request.actual_prices, predicted_prices)
        mae = mean_absolute_error(request.actual_prices, predicted_prices)

        # Construct response
        response = {
            "scrip": request.scrip,
            "current_price": current_price,
            "future_predictions": {
                "days": request.days,
                "predicted_prices": predicted_prices
            },
            "metrics": {
                "mean": round(mean_price, 2),
                "max": round(max_price, 2),
                "standard_deviation": round(std_dev, 2)
            },
            "evaluation": {
                "mse": round(mse, 4),
                "mae": round(mae, 4)
            }
        }

        # Log response data for debugging
        logging.debug(f"Response data: {response}")
        return response

    except ValueError as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Internal server error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error. Please check your input or try again later.")
#dummy model 
# Example prediction request model
# class PredictRequest(BaseModel):
#     scrip: str
#     data: list[float]
#     days: list[int]
#     actual_prices: list[float]  # Include actual prices for evaluation

# # Dummy model for predictions
# def dummy_model_predict(input_data, days):
#     if not input_data:
#         raise ValueError("Empty input data")

#     current_price = input_data[-1]  # Last known price as the current price
#     future_predictions = [(current_price + day * 0.5) for day in days]  # Dummy logic for prediction

#     return current_price, future_predictions

# # Endpoint for predictions
# @app.post("/predict")
# async def predict(request: PredictRequest):
#     try:
#         # Log incoming request data for debugging
#         logging.debug(f"Incoming request data: {request.scrip}, {request.data}, {request.days}, {request.actual_prices}")

#         # Dummy prediction logic
#         current_price, predicted_prices = dummy_model_predict(request.data, request.days)

#         # Compute metrics for predicted prices
#         mean_price = statistics.mean(predicted_prices)
#         max_price = max(predicted_prices)
#         std_dev = statistics.stdev(predicted_prices)

#         # Calculate evaluation metrics (MSE, MAE) using actual prices
#         if len(request.actual_prices) != len(predicted_prices):
#             raise ValueError("Length of actual prices and predicted prices must be the same")

#         mse = mean_squared_error(request.actual_prices, predicted_prices)
#         mae = mean_absolute_error(request.actual_prices, predicted_prices)

#         # Construct response
#         response = {
#             "scrip": request.scrip,
#             "current_price": current_price,
#             "future_predictions": {
#                 "days": request.days,
#                 "predicted_prices": predicted_prices
#             },
#             "metrics": {
#                 "mean": round(mean_price, 2),
#                 "max": round(max_price, 2),
#                 "standard_deviation": round(std_dev, 2)
#             },
#             "evaluation": {
#                 "mse": round(mse, 4),
#                 "mae": round(mae, 4)
#             }
#         }

#         # Log response data for debugging
#         logging.debug(f"Response data: {response}")
#         return response

#     except ValueError as e:
#         logging.error(f"Prediction error: {e}")
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         logging.error(f"Internal server error: {e}")
#         raise HTTPException(status_code=500, detail="Internal Server Error. Please check your input or try again later.")




# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import numpy as np
# import joblib
# import statsmodels.api as sm
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# app = FastAPI()



# # Load saved models at the start
# linear_regression_model = joblib.load("saved_models/linear_regression.joblib")
# random_forest_model = joblib.load("saved_models/random_forest.joblib")
# xgboost_model = joblib.load("saved_models/xgboost.joblib")
# # arima_model = joblib.load("saved_models/arima.joblib")

# class StockRequest(BaseModel):
#     scrip: str
#     historical_data: list
#     model_type: str  # Options: 'linear_regression', 'random_forest', 'xgboost', 'arima'

# class PredictionResponse(BaseModel):
#     scrip: str
#     current_price: float
#     future_predictions: list
#     metrics: dict

# def calculate_metrics(y_true, y_pred):
#     mse = mean_squared_error(y_true, y_pred)
#     mae = mean_absolute_error(y_true, y_pred)
#     r_squared = r2_score(y_true, y_pred) if len(y_true) > 1 else None
#     mean_pred = np.mean(y_pred)
#     max_pred = np.max(y_pred)
#     min_pred = np.min(y_pred)
#     return {
#         "mean": mean_pred,
#         "max": max_pred,
#         "min": min_pred,
#         "mse": mse,
#         "mae": mae,
#         "r_squared": r_squared
#     }

# @app.post("/predict_multiple", response_model=PredictionResponse)
# async def predict_stock(data: StockRequest):
#     historical_data = np.array(data.historical_data).reshape(-1, 1)
#     last_known_price = historical_data[-1, 0]

#     # Initialize the response
#     response = {
#         "scrip": data.scrip,
#         "current_price": last_known_price,
#         "future_predictions": [],
#         "metrics": {}
#     }

#     X = np.array(range(len(historical_data))).reshape(-1, 1)
#     y = historical_data.flatten()

#     # Define how many future points you want to predict
#     num_future_predictions = 6
#     future_indices = np.array(range(len(historical_data), len(historical_data) + num_future_predictions)).reshape(-1, 1)

#     if data.model_type == "linear_regression":
#         # Use the loaded model for prediction
#         future_predictions = linear_regression_model.predict(future_indices)
#         response["future_predictions"] = future_predictions.tolist()
#         response["metrics"] = calculate_metrics(y, linear_regression_model.predict(X))

#     elif data.model_type == "random_forest":
#         # Use the loaded model for prediction
#         future_predictions = random_forest_model.predict(future_indices)
#         response["future_predictions"] = future_predictions.tolist()
#         response["metrics"] = calculate_metrics(y, random_forest_model.predict(X))

#     elif data.model_type == "xgboost":
#         # Use the loaded model for prediction
#         future_predictions = xgboost_model.predict(future_indices)
#         response["future_predictions"] = future_predictions.tolist()
#         response["metrics"] = calculate_metrics(y, xgboost_model.predict(X))

#     elif data.model_type == "arima":
#         # ARIMA model needs to be fit each time due to its time-series dependency
#         model = sm.tsa.ARIMA(y, order=(1, 1, 1))
#         model_fit = model.fit()
#         future_predictions = model_fit.forecast(steps=num_future_predictions)
#         response["future_predictions"] = future_predictions.tolist()
#         response["metrics"] = calculate_metrics(y[-num_future_predictions:], future_predictions)

#     else:
#         raise HTTPException(status_code=400, detail="Invalid model type specified.")

#     return response

# Load all models and metrics
with open("models/models/metrics.pkl", "rb") as f:
    metrics_data = pickle.load(f)

linear_regression_model = pickle.load(open("models/models/linear_regression.pkl", "rb"))
random_forest_model = pickle.load(open("models/models/random_forest.pkl", "rb"))
xgboost_model = pickle.load(open("models/models/xgboost.pkl", "rb"))
lstm_model = load_model("models/models/lstm_model.h5")
arima_model = ARIMAResults.load("models/models/arima.pkl")

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
@app.post("/predict_multiple", response_model=PredictionResponse)
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




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)












































# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List, Dict
# import numpy as np
# from tensorflow.keras.models import load_model
# import joblib
# import json
# from joblib import load
# import pyngrok
# from pyngrok import ngrok
# import uvicorn
# import threading
# # ! ngrok config add-authtoken 2nnT3Y1XBgCbrPzj9CKoHm3yd7P_39x3kazk5NM6qxop2RLGg

# app = FastAPI()

# # Load the trained model and scaler
# model = load_model('lstm_model.h5')
# scaler = joblib.load('scaler.joblib')

# class PredictRequest(BaseModel):
#     scrip: str
#     data: List[float]  # Historical stock prices for model input
#     days: List[int]  # Days to predict

# @app.post("/predict-lstm")
# async def predict(request: PredictRequest):
#     try:
#         # Scale input data
#         input_data = np.array(request.data).reshape(-1, 1)
#         scaled_data = scaler.transform(input_data)

#         # Prepare data for LSTM (considering time steps)
#         time_step = 3  # Ensure this matches the model training
#         X_input = scaled_data[-time_step:].reshape(1, time_step, 1)

#         # Make predictions for each day
#         predictions = []
#         for day in request.days:
#             predicted_price = model.predict(X_input)
#             predictions.append(predicted_price[0][0])
#             # Update X_input with the new predicted price
#             X_input = np.append(X_input[:, 1:, :], [[predicted_price]], axis=1)

#         # Inverse scale predictions
#         predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten().tolist()

#         # Calculate metrics (example metrics based on predictions)
#         mean_price = np.mean(predicted_prices)
#         max_price = np.max(predicted_prices)
#         stddev_price = np.std(predicted_prices)

#         # Evaluation (for illustration, use actual_prices if available)
#         actual_prices = [153.00, 155.00, 157.50, 164.50, 172.00, 330.00]  # Example actuals
#         mse = mean_squared_error(actual_prices[:len(predicted_prices)], predicted_prices)
#         mae = mean_absolute_error(actual_prices[:len(predicted_prices)], predicted_prices)

#         # Construct response
#         response = {
#             "scrip": request.scrip,
#             "current_price": request.data[-1],
#             "future_predictions": {
#                 "days": request.days,
#                 "predicted_prices": predicted_prices
#             },
#             "metrics": {
#                 "mean": mean_price,
#                 "max": max_price,
#                 "standard_deviation": stddev_price
#             },
#             "evaluation": {
#                 "mse": mse,
#                 "mae": mae
#             }
#         }
#         return response

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
# # # Expose API with ngrok
# if __name__ == "__main__":
#     # Set up ngrok tunnel to expose the FastAPI app
#     # public_url = ngrok.connect(8000)
#     # print(f"ngrok tunnel \"{public_url}\" -> \"http://127.0.0.1:8000\"")

#     # Run the FastAPI app with uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# # def run():
# #     uvicorn.run(app, host="0.0.0.0", port=8000)

# # # Start Uvicorn in a separate thread
# # thread = threading.Thread(target=run)
# # thread.start()
