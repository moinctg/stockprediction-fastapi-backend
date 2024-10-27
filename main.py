# main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import numpy as np
import tensorflow as tf
import uvicorn
import threading
import traceback
import logging
import statistics
import joblib 
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error

app = FastAPI()


# # Load model and scaler
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

origins = [
    "http://localhost:3000",
   # Your Next.js app's URL
    "https://681c-34-148-26-126.ngrok-free.app"  # Your ngrok URL
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.DEBUG)


#dummy model 
# Example prediction request model
class PredictRequest(BaseModel):
    scrip: str
    data: list[float]
    days: list[int]
    actual_prices: list[float]  # Include actual prices for evaluation

# Dummy model for predictions
def dummy_model_predict(input_data, days):
    if not input_data:
        raise ValueError("Empty input data")

    current_price = input_data[-1]  # Last known price as the current price
    future_predictions = [(current_price + day * 0.5) for day in days]  # Dummy logic for prediction

    return current_price, future_predictions

# Endpoint for predictions
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
        std_dev = statistics.stdev(predicted_prices)

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