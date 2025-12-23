from pathlib import Path
from backend.config import get_mongodb
import logging
import hashlib
import uuid
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class UserManager:
    def __init__(self):
        self.db = get_mongodb()
        self.users = None
        self.use_json_fallback = False
        self.json_file = Path("backend/data/users.json")
        
        if self.db is None:
            logger.warning("⚠️ Cảnh báo: MongoDB không khả dụng. Sử dụng JSON file dự phòng.")
            self.use_json_fallback = True
            self._ensure_json_file()
            self._create_default_admin_json()
            return
        
        try:
            self.users = self.db['users']
            self.users.create_index('username', unique=True)
            
            # Auto-create admin
            if not self.users.find_one({'username': 'admin'}):
                self.users.insert_one({
                    'username': 'admin',
                    'password': hashlib.sha256('admin123'.encode()).hexdigest(),
                    'user_id': 'admin_00000000',
                    'role': 'admin',
                    'created_at': datetime.now().isoformat()
                })
        except Exception as e:
            logger.warning(f"⚠️ Lỗi khởi tạo UserManager: {e}. Chuyển sang JSON.")
            self.use_json_fallback = True
            self.users = None
            self._ensure_json_file()
            self._create_default_admin_json()
    
    def _ensure_json_file(self):
        """Ensure JSON file exists"""
        self.json_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.json_file.exists():
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=2)
    
    def _create_default_admin_json(self):
        """Create default admin in JSON file"""
        users = self._load_from_json()
        if 'admin' not in users:
            users['admin'] = {
                'username': 'admin',
                'password': hashlib.sha256('admin123'.encode()).hexdigest(),
                'user_id': 'admin_00000000',
                'role': 'admin',
                'created_at': datetime.now().isoformat()
            }
            self._save_to_json(users)
    
    def _load_from_json(self):
        """Load users from JSON file"""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    
    def _save_to_json(self, users_dict):
        """Save users to JSON file"""
        try:
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(users_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ Lỗi lưu vào JSON: {e}")
    
    def load_users(self):
        """Compatibility: trả về dict như JSON format"""
        # Use JSON fallback if MongoDB not available
        if self.use_json_fallback:
            return self._load_from_json()
        
        users = {}
        try:
            if self.db is None or self.users is None:
                return self._load_from_json()
            
            for u in self.users.find():
                users[u['username'].lower()] = {
                    'username': u['username'],
                    'password': u['password'],
                    'user_id': u['user_id'],
                    'role': u['role'],
                    'created_at': u['created_at']
                }
        except Exception as e:
            logger.error(f"❌ Lỗi tải users từ MongoDB: {e}. Sử dụng JSON dự phòng.")
            return self._load_from_json()
        return users
    
    def save_users(self, users_dict):
        """Compatibility: lưu dict vào MongoDB hoặc JSON"""
        # Use JSON fallback if MongoDB not available
        if self.use_json_fallback:
            self._save_to_json(users_dict)
            return
        
        try:
            if self.db is None or self.users is None:
                self._save_to_json(users_dict)
                return
            
            for username, user_data in users_dict.items():
                self.users.update_one(
                    {'username': username},
                    {'$set': user_data},
                    upsert=True
                )
        except Exception as e:
            logger.error(f"❌ Lỗi lưu users vào MongoDB: {e}. Sử dụng JSON dự phòng.")
            self._save_to_json(users_dict)


_manager = None
def get_user_manager():
    global _manager
    if _manager is None:
        _manager = UserManager()
    return _manager


def load_users():
    return get_user_manager().load_users()

def save_users(users):
    get_user_manager().save_users(users)