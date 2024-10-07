


from fastapi import FastAPI
from app.api.endpoints import stock_data, train_model

app = FastAPI()

app.include_router(stock_data.router)
app.include_router(train_model.router)

# Run the app using: uvicorn app.main:app --reload

























# import os
# import numpy as np
# import pandas as pd
# from fastapi import FastAPI, HTTPException, UploadFile, File
# from pydantic import BaseModel
# from motor.motor_asyncio import AsyncIOMotorClient
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from datetime import datetime

# # MongoDB configuration
# MONGO_DETAILS = "mongodb+srv://cpimoinuddin:moin123456@cluster0.wh6jr.mongodb.net/Stock-Db?retryWrites=true&w=majority&appName=Cluster0"  # Change this for your MongoDB connection
# client = AsyncIOMotorClient(MONGO_DETAILS)
# database = client.stock_Db
# stock_collection = database.get_collection("stocks")

# app = FastAPI()

# # Define Stock Data schema
# class StockData(BaseModel):
#     Date: datetime
#     Scrip: str
#     Open: float
#     High: float
#     Low: float
#     Close: float
#     Volume: int

# class TrainModelRequest(BaseModel):
#     batch_size: int = 1
#     epochs: int = 10
#     time_step: int = 60

# # Data preprocessing functions
# def preprocess_data(data):
#     scaler = MinMaxScaler()
#     scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume']])
#     return scaled_data, scaler

# def create_dataset(data, time_step=60):
#     X, y = [], []
#     for i in range(len(data) - time_step - 1):
#         X.append(data[i:(i + time_step), :])
#         y.append(data[i + time_step, 3])  # Predicting 'Close' price
#     return np.array(X), np.array(y)

# def create_lstm_model(input_shape):
#     model = Sequential()
#     model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
#     model.add(Dropout(0.2))
#     model.add(LSTM(50, return_sequences=False))
#     model.add(Dropout(0.2))
#     model.add(Dense(25))
#     model.add(Dense(1))  # Output layer for predicting 'Close' price
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model

# # Route to add stock data
# @app.post("/add-stock-data/")
# async def add_stock_data(stock_data: StockData):
#     stock_data_dict = stock_data.dict()
#     await stock_collection.insert_one(stock_data_dict)
#     return {"message": "Stock data added successfully"}

# # Route to train the model
# 




# # Route to train the model with CSV file upload
# @app.post("/train-with-csv/")
# async def train_model_with_csv(file: UploadFile = File(...), train_request: TrainModelRequest):
#     try:
#         # Read CSV file
#         contents = await file.read()
#         df = pd.read_csv(pd.compat.StringIO(contents.decode('utf-8')))
        
#         # Ensure date is in correct format
#         df['Date'] = pd.to_datetime(df['Date'], format='%d%m%Y', errors='coerce')  # Adjust format as needed
#         df.dropna(subset=['Date'], inplace=True)  # Drop rows where date conversion failed
#         df = df.sort_values('Date')

#         # Check if DataFrame is empty after conversion
#         if df.empty:
#             raise HTTPException(status_code=400, detail="CSV file is empty or invalid.")

#         # Preprocess data
#         scaled_data, scaler = preprocess_data(df)
#         X, y = create_dataset(scaled_data, time_step=train_request.time_step)

#         # Check if we have sufficient data
#         if X.shape[0] == 0 or y.shape[0] == 0:
#             raise HTTPException(status_code=400, detail="Insufficient data for training. Check your dataset.")

#         # Create train/test split
#         train_size = int(len(X) * 0.8)
#         if train_size == 0:
#             raise HTTPException(status_code=400, detail="Insufficient data to create train/test split.")

#         X_train, X_test = X[:train_size], X[train_size:]
#         y_train, y_test = y[:train_size], y[train_size:]

#         # Create and train the LSTM model
#         model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
#         model.fit(X_train, y_train, batch_size=train_request.batch_size, epochs=train_request.epochs)

#         return {"message": "Model trained successfully"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Run the app using: uvicorn main:app --reload