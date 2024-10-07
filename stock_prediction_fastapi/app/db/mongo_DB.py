# MongoDB configuration
from motor.motor_asyncio import AsyncIOMotorClient

MONGO_DETAILS = "mongodb+srv://cpimoinuddin:moin123456@cluster0.wh6jr.mongodb.net/Stock-Db?retryWrites=true&w=majority&appName=Cluster0"  # Change this for your MongoDB connection
client = AsyncIOMotorClient(MONGO_DETAILS)
database = client.stock_Db
stock_collection = database.get_collection("stocks")