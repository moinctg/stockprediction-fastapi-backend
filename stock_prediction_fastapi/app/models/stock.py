from datetime import datetime
from pydantic import BaseModel

class StockData(BaseModel):
    Date: datetime
    Scrip: str
    Open: float
    High: float
    Low: float
    Close: float
    Volume: int

class TrainModelRequest(BaseModel):
    time_step: int = 60
    batch_size: int = 1
    epochs: int = 10
