from fastapi import APIRouter, UploadFile, File ,HTTPException,Body
import os
import numpy as np
import pandas as pd
from app.utils.data_preprocessing import load_data_from_csv, preprocess_data, create_dataset
from app.models.stock import TrainModelRequest 
from app.db.mongo_DB import stock_collection
from app.services.prediction_service import create_lstm_model

router = APIRouter()

@router.post("/train")
# # Route to train the model

# Route to train the model
@router.post("/train")
async def train_model(train_request: TrainModelRequest):
    try:
        # Fetch stock data from MongoDB
        stock_data_cursor = stock_collection.find({})
        stock_data = await stock_data_cursor.to_list(length=1000)
        if not stock_data:
            raise HTTPException(status_code=404, detail="No stock data found")

        # Convert stock data to DataFrame
        df = pd.DataFrame(stock_data)

        # Ensure date is in correct format
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%d%m%Y')  # Assuming DDMMYYYY format
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error converting dates: {str(e)}")

        df = df.sort_values('Date')

        # Check if DataFrame is empty after conversion
        if df.empty:
            raise HTTPException(status_code=400, detail="Stock data DataFrame is empty")

        # Preprocess data
        scaled_data, scaler = preprocess_data(df)
        X, y = create_dataset(scaled_data, time_step=train_request.time_step)

        # Check the shapes
        if X.shape[0] == 0 or y.shape[0] == 0:
            raise HTTPException(status_code=400, detail="Insufficient data for training. Check your dataset.")

        # Create train/test split
        train_size = int(len(X) * 0.8)
        if train_size == 0:
            raise HTTPException(status_code=400, detail="Insufficient data to create train/test split.")

        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Check input shapes
        if X_train.shape[0] == 0 or y_train.shape[0] == 0:
            raise HTTPException(status_code=400, detail="No training data available after split.")

        # Create and train the LSTM model
        model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, batch_size=train_request.batch_size, epochs=train_request.epochs)

        return {"message": "Model trained successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train-with-csv/")
async def train_model_with_csv(
    file: UploadFile = File(...),
    train_request: TrainModelRequest = Body(...)
):
    # Save the uploaded file temporarily
    file_location = f"temp/{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())
    
    # Load and preprocess the data
    df = load_data_from_csv(file_location)
    scaled_data, scaler = preprocess_data(df)
    X, y = create_dataset(scaled_data, time_step=train_request.time_step)

    # Continue with model training...
    return {"message": "Data processed and model training initiated"}