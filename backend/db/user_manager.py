# backend/db/user_manager.py
"""
User management with MongoDB integration
"""
from datetime import datetime
import hashlib
import uuid
from pathlib import Path
from backend.config import get_mongodb

class UserManager:
    def __init__(self):
        self.db = get_mongodb()
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
    
    def load_users(self):
        """Compatibility: trả về dict như JSON format"""
        users = {}
        for u in self.users.find():
            users[u['username'].lower()] = {
                'username': u['username'],
                'password': u['password'],
                'user_id': u['user_id'],
                'role': u['role'],
                'created_at': u['created_at']
            }
        return users
    
    def save_users(self, users_dict):
        """Compatibility: lưu dict vào MongoDB"""
        for username, user_data in users_dict.items():
            self.users.update_one(
                {'username': username},
                {'$set': user_data},
                upsert=True
            )


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