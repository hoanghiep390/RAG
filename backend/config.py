# backend/config.py - ENHANCED VERSION
"""
MongoDB Configuration & Connection Manager + Centralized Config
"""
import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

# ================= Centralized Configuration =================
class Config:
    """Centralized configuration for all components"""
    
    # ========== MongoDB ==========
    MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
    MONGODB_DATABASE = os.getenv('MONGODB_DATABASE', 'lightrag_db')
    
    # ========== Embeddings ==========
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
    EMBEDDING_DIM = int(os.getenv('EMBEDDING_DIM', '384'))
    
    # ========== LLM ==========
    LLM_PROVIDER = os.getenv('LLM_PROVIDER', 'groq')
    LLM_MODEL = os.getenv('LLM_MODEL', 'llama-3.1-70b-versatile')
    MAX_CONCURRENT_LLM_CALLS = int(os.getenv('MAX_CONCURRENT_LLM_CALLS', '16'))
    
    # ========== Processing ==========
    EXTRACTION_BATCH_SIZE = int(os.getenv('EXTRACTION_BATCH_SIZE', '20'))
    EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', '128'))
    MAX_GLEANING = int(os.getenv('MAX_GLEANING', '1'))  # Continue extraction attempts
    USE_LLM_ENTITY_MERGE = os.getenv('USE_LLM_ENTITY_MERGE', 'true').lower() == 'true'
    MIN_DESCRIPTIONS_FOR_LLM_MERGE = int(os.getenv('MIN_DESCRIPTIONS_FOR_LLM_MERGE', '3'))
    USE_LLM_RELATIONSHIP_MERGE = os.getenv('USE_LLM_RELATIONSHIP_MERGE', 'true').lower() == 'true'

    
    # ========== Chunking ==========
    DEFAULT_CHUNK_SIZE = int(os.getenv('DEFAULT_CHUNK_SIZE', '300'))
    DEFAULT_CHUNK_OVERLAP = int(os.getenv('DEFAULT_CHUNK_OVERLAP', '50'))
    
    # ========== FAISS ==========
    USE_HNSW = os.getenv('USE_HNSW', 'true').lower() == 'true'
    HNSW_M = int(os.getenv('HNSW_M', '32'))
    HNSW_EF_CONSTRUCTION = int(os.getenv('HNSW_EF_CONSTRUCTION', '200'))
    HNSW_EF_SEARCH = int(os.getenv('HNSW_EF_SEARCH', '50'))
    
    # ========== Performance ==========
    MAX_FILE_SIZE_MB = int(os.getenv('MAX_FILE_SIZE_MB', '50'))
    AUTO_REBUILD_THRESHOLD = float(os.getenv('AUTO_REBUILD_THRESHOLD', '0.2'))
    
    # ========== Retrieval Performance ==========
    ENABLE_QUERY_EXPANSION = os.getenv('ENABLE_QUERY_EXPANSION', 'false').lower() == 'true'
    MAX_CONTEXT_CHUNKS = int(os.getenv('MAX_CONTEXT_CHUNKS', '3'))
    MAX_ENTITY_CACHE = int(os.getenv('MAX_ENTITY_CACHE', '300'))
    
    # ========== Storage Paths ==========
    DATA_DIR = os.getenv('DATA_DIR', 'backend/data')
    
    @classmethod
    def get_user_upload_dir(cls, user_id: str):
        """Get user upload directory"""
        from pathlib import Path
        path = Path(cls.DATA_DIR) / user_id / "uploads"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @classmethod
    def get_user_vector_dir(cls, user_id: str):
        """Get user vector directory"""
        from pathlib import Path
        path = Path(cls.DATA_DIR) / user_id / "vectors"
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        errors = []
        
        # Check API keys
        if cls.LLM_PROVIDER == 'openai' and not os.getenv('OPENAI_API_KEY'):
            errors.append("OPENAI_API_KEY not set but LLM_PROVIDER is 'openai'")
        
        if cls.LLM_PROVIDER == 'groq' and not os.getenv('GROQ_API_KEY'):
            errors.append("GROQ_API_KEY not set but LLM_PROVIDER is 'groq'")
        
        # Check embedding dimension
        if cls.EMBEDDING_MODEL == 'all-MiniLM-L6-v2' and cls.EMBEDDING_DIM != 384:
            logger.warning(f"⚠️ EMBEDDING_DIM là {cls.EMBEDDING_DIM} nhưng all-MiniLM-L6-v2 tạo vectors 384-chiều")
        
        # Check HNSW params
        if cls.USE_HNSW and cls.HNSW_M < 4:
            errors.append(f"HNSW_M ({cls.HNSW_M}) is too low, should be >= 4")
        
        if errors:
            raise ValueError(f"Configuration errors:\n" + "\n".join(f"- {e}" for e in errors))
        
        logger.info("✅ Đã xác thực cấu hình thành công")
# ================= MongoDB Connection Manager =================
class MongoDBConfig:
    """MongoDB configuration and connection manager"""
    
    def __init__(self):
        self.uri = Config.MONGODB_URI
        self.db_name = Config.MONGODB_DATABASE
        self.client = None
        self.db = None
        
    def connect(self):
        """Establish MongoDB connection"""
        try:
            self.client = MongoClient(self.uri)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.db_name]
            logger.info(f"✅ Đã kết nối MongoDB: {self.db_name}")
            return True
        except ConnectionFailure as e:
            logger.error(f"❌ Kết nối MongoDB thất bại: {e}")
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
            logger.info("✅ Đã đóng kết nối MongoDB")
    
    def health_check(self):
        """Check MongoDB health"""
        try:
            if self.client:
                self.client.admin.command('ping')
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Kiểm tra sức khỏe MongoDB thất bại: {e}")
            return False

# Global instance
_mongo_config = None

def get_mongodb():
    """Get MongoDB database instance (Singleton)"""
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

def get_mongodb_client():
    """Get MongoDB client instance"""
    global _mongo_config
    if _mongo_config is None:
        _mongo_config = MongoDBConfig()
    if _mongo_config.client is None:
        _mongo_config.connect()
    return _mongo_config.client

# ================= Initialization =================
def initialize_config():
    """Initialize and validate configuration"""
    try:
        Config.validate()
        Config.print_config()
        
        db = get_mongodb()
        if db is not None:
            logger.info("✅ Đã khởi tạo cấu hình thành công")
            return True
        else:
            logger.error("❌ Không thể kết nối MongoDB")
            return False
    except Exception as e:
        logger.error(f"❌ Khởi tạo cấu hình thất bại: {e}")
        return False

# ================= Export =================
__all__ = [
    'Config',
    'MongoDBConfig',
    'get_mongodb',
    'close_mongodb',
    'get_mongodb_client',
    'initialize_config'
]