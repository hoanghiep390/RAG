# backend/config.py
"""
MongoDB Configuration & Connection Manager
"""
import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

class MongoDBConfig:
    """MongoDB configuration and connection manager"""
    
    def __init__(self):
        self.uri = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
        self.db_name = os.getenv('MONGODB_DATABASE', 'lightrag_db')
        self.client = None
        self.db = None
        
    def connect(self):
        """Establish MongoDB connection"""
        try:
            self.client = MongoClient(self.uri)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            logger.info(f"✅ Connected to MongoDB: {self.db_name}")
            return True
        except ConnectionFailure as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            return False
    
    def get_database(self):
        """Get database instance"""
        if self.db is None:
            self.connect()
        return self.db
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("🔒 MongoDB connection closed")

# Global instance
_mongo_config = None

def get_mongodb():
    """Get MongoDB database instance"""
    global _mongo_config
    if _mongo_config is None:
        _mongo_config = MongoDBConfig()
    return _mongo_config.get_database()

def close_mongodb():
    """Close MongoDB connection"""
    global _mongo_config
    if _mongo_config:
        _mongo_config.close()
        _mongo_config = None