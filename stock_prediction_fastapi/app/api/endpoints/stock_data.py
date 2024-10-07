from fastapi import APIRouter
from app.models.stock  import StockData

router = APIRouter()

@router.post("/add-stock-data/")
async def add_stock_data(stock_data: StockData):
    # Logic to add stock data
    pass
