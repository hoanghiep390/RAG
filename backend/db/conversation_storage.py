# backend/db/conversation_storage.py
"""
 Conversation Storage - MongoDB persistence for chat history
Quản lý multi-conversation với message tracking
"""
from typing import List, Dict, Optional
from datetime import datetime
from backend.config import get_mongodb
import logging
import uuid

logger = logging.getLogger(__name__)

class ConversationStorage:
    """
    MongoDB storage cho conversations
    
    Collections:
    - conversations: Metadata của conversation
    - messages: Chi tiết messages trong conversation
    """
    
    def __init__(self, user_id: str):
        """
        Args:
            user_id: User ID for data isolation
        """
        self.user_id = user_id
        try:
            self.db = get_mongodb()
            self.conversations = self.db['conversations']
            self.messages = self.db['messages']
            
            self._create_indexes()
            logger.info(f" ConversationStorage initialized for user: {user_id}")
        except Exception as e:
            logger.error(f" Failed to initialize ConversationStorage: {e}")
            raise
    
    def _create_indexes(self):
        """Tạo indexes cho hiệu suất"""
        try:
            # Indexes cho conversations
            self.conversations.create_index([
                ('user_id', 1), 
                ('conversation_id', 1)
            ], unique=True)
            
            self.conversations.create_index([
                ('user_id', 1), 
                ('updated_at', -1)
            ])
            
            # Indexes cho messages
            self.messages.create_index([
                ('conversation_id', 1), 
                ('message_id', 1)
            ], unique=True)
            
            self.messages.create_index([
                ('conversation_id', 1), 
                ('created_at', 1)
            ])
            
            logger.debug(" Conversation indexes created")
        except Exception as e:
            logger.warning(f" Index creation warning: {e}")
    
    # ============================================
    # QUẢN LÝ CONVERSATION
    # ============================================
    
    def create_conversation(self, title: Optional[str] = None) -> str:
        """
        Tạo conversation mới
        
        Args:
            title: Tiêu đề conversation (optional, auto-generate nếu None)
        
        Returns:
            conversation_id
        """
        conversation_id = f"conv_{uuid.uuid4().hex[:16]}"
        
        if not title:
            title = f"New Chat {datetime.now().strftime('%m/%d %H:%M')}"
        
        try:
            conversation = {
                'user_id': self.user_id,
                'conversation_id': conversation_id,
                'title': title,
                'created_at': datetime.now(),
                'updated_at': datetime.now(),
                'message_count': 0,
                'metadata': {}
            }
            
            self.conversations.insert_one(conversation)
            logger.info(f" Created conversation: {conversation_id}")
            
            return conversation_id
        
        except Exception as e:
            logger.error(f" Failed to create conversation: {e}")
            raise
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """
        Lấy thông tin conversation
        
        Args:
            conversation_id: ID của conversation
        
        Returns:
            Conversation dict hoặc None
        """
        try:
            conv = self.conversations.find_one({
                'user_id': self.user_id,
                'conversation_id': conversation_id
            })
            return conv
        except Exception as e:
            logger.error(f" Failed to get conversation {conversation_id}: {e}")
            return None
    
    def list_conversations(
        self, 
        limit: int = 50, 
        skip: int = 0
    ) -> List[Dict]:
        """
        Lấy danh sách conversations của user
        
        Args:
            limit: Số lượng conversations
            skip: Bỏ qua N conversations đầu tiên
        
        Returns:
            List of conversation dicts
        """
        try:
            conversations = list(
                self.conversations.find(
                    {'user_id': self.user_id}
                )
                .sort('updated_at', -1)
                .skip(skip)
                .limit(limit)
            )
            
            return conversations
        
        except Exception as e:
            logger.error(f" Failed to list conversations: {e}")
            return []
    
    def update_conversation_title(self, conversation_id: str, title: str) -> bool:
        """
        Cập nhật tiêu đề conversation
        
        Args:
            conversation_id: ID của conversation
            title: Tiêu đề mới
        
        Returns:
            True nếu thành công
        """
        try:
            result = self.conversations.update_one(
                {
                    'user_id': self.user_id,
                    'conversation_id': conversation_id
                },
                {
                    '$set': {
                        'title': title,
                        'updated_at': datetime.now()
                    }
                }
            )
            
            return result.modified_count > 0
        
        except Exception as e:
            logger.error(f" Failed to update conversation title: {e}")
            return False
    
    def delete_conversation(self, conversation_id: str) -> Dict:
        """
        Xóa conversation và tất cả messages
        
        Args:
            conversation_id: ID của conversation
        
        Returns:
            Dict với stats
        """
        try:
            # Xóa messages
            msg_result = self.messages.delete_many({
                'conversation_id': conversation_id
            })
            
            # Xóa conversation
            conv_result = self.conversations.delete_one({
                'user_id': self.user_id,
                'conversation_id': conversation_id
            })
            
            logger.info(
                f" Deleted conversation {conversation_id}: "
                f"{msg_result.deleted_count} messages, "
                f"{conv_result.deleted_count} conversation"
            )
            
            return {
                'conversation_deleted': conv_result.deleted_count,
                'messages_deleted': msg_result.deleted_count
            }
        
        except Exception as e:
            logger.error(f" Failed to delete conversation: {e}")
            return {'conversation_deleted': 0, 'messages_deleted': 0}
    
    # ============================================
    # QUẢN LÝ MESSAGE
    # ============================================
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Thêm message vào conversation
        
        Args:
            conversation_id: ID của conversation
            role: 'user' hoặc 'assistant'
            content: Nội dung message
            metadata: Optional metadata (retrieval stats, etc.)
        
        Returns:
            message_id
        """
        message_id = f"msg_{uuid.uuid4().hex[:16]}"
        
        try:
            message = {
                'conversation_id': conversation_id,
                'message_id': message_id,
                'role': role,
                'content': content,
                'created_at': datetime.now(),
                'metadata': metadata or {}
            }
            
            self.messages.insert_one(message)
            
            # Cập nhật conversation
            self.conversations.update_one(
                {
                    'user_id': self.user_id,
                    'conversation_id': conversation_id
                },
                {
                    '$set': {'updated_at': datetime.now()},
                    '$inc': {'message_count': 1}
                }
            )
            
            logger.debug(f" Added message {message_id} to {conversation_id}")
            
            return message_id
        
        except Exception as e:
            logger.error(f" Failed to add message: {e}")
            raise
    
    def get_messages(
        self,
        conversation_id: str,
        limit: int = 100,
        skip: int = 0
    ) -> List[Dict]:
        """
        Lấy messages của conversation
        
        Args:
            conversation_id: ID của conversation
            limit: Số lượng messages
            skip: Bỏ qua N messages đầu tiên
        
        Returns:
            List of message dicts (sorted by created_at)
        """
        try:
            messages = list(
                self.messages.find(
                    {'conversation_id': conversation_id}
                )
                .sort('created_at', 1)  # Tăng dần
                .skip(skip)
                .limit(limit)
            )
            
            return messages
        
        except Exception as e:
            logger.error(f" Failed to get messages: {e}")
            return []
    
    def get_recent_messages(
        self,
        conversation_id: str,
        n: int = 10
    ) -> List[Dict]:
        """
        Lấy N messages gần nhất
        
        Args:
            conversation_id: ID của conversation
            n: Số lượng messages
        
        Returns:
            List of recent messages
        """
        try:
            messages = list(
                self.messages.find(
                    {'conversation_id': conversation_id}
                )
                .sort('created_at', -1)  # Giảm dần
                .limit(n)
            )
            
            # Đảo ngược về thứ tự thời gian
            messages.reverse()
            
            return messages
        
        except Exception as e:
            logger.error(f" Failed to get recent messages: {e}")
            return []
    
    def delete_message(self, message_id: str) -> bool:
        """
        Xóa một message
        
        Args:
            message_id: ID của message
        
        Returns:
            True nếu thành công
        """
        try:
            result = self.messages.delete_one({'message_id': message_id})
            
            if result.deleted_count > 0:
                logger.info(f" Deleted message {message_id}")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f" Failed to delete message: {e}")
            return False
    
    # ============================================
    # THAO TÁC HÀNG LOẠT
    # ============================================
    
    def save_conversation_bulk(
        self,
        conversation_id: str,
        messages: List[Dict]
    ) -> int:
        """
        Lưu nhiều messages cùng lúc
        
        Args:
            conversation_id: ID của conversation
            messages: List of {"role": "...", "content": "...", "metadata": {...}}
        
        Returns:
            Số lượng messages đã lưu
        """
        if not messages:
            return 0
        
        try:
            message_docs = []
            for msg in messages:
                message_id = f"msg_{uuid.uuid4().hex[:16]}"
                message_docs.append({
                    'conversation_id': conversation_id,
                    'message_id': message_id,
                    'role': msg['role'],
                    'content': msg['content'],
                    'created_at': datetime.now(),
                    'metadata': msg.get('metadata', {})
                })
            
            result = self.messages.insert_many(message_docs, ordered=False)
            
            # Update conversation
            self.conversations.update_one(
                {
                    'user_id': self.user_id,
                    'conversation_id': conversation_id
                },
                {
                    '$set': {'updated_at': datetime.now()},
                    '$inc': {'message_count': len(message_docs)}
                }
            )
            
            logger.info(f" Saved {len(result.inserted_ids)} messages to {conversation_id}")
            
            return len(result.inserted_ids)
        
        except Exception as e:
            logger.error(f" Failed to save messages: {e}")
            return 0
    
    # ============================================
    # THỐNG KÊ
    # ============================================
    
    def get_user_statistics(self) -> Dict:
        """
        Lấy thống kê conversations của user
        
        Returns:
            Dict với stats
        """
        try:
            total_conversations = self.conversations.count_documents({
                'user_id': self.user_id
            })
            
            total_messages = self.messages.count_documents({
                'conversation_id': {
                    '$in': [
                        c['conversation_id'] 
                        for c in self.conversations.find(
                            {'user_id': self.user_id},
                            {'conversation_id': 1}
                        )
                    ]
                }
            })
            
            # Lấy conversation gần nhất
            recent = self.conversations.find_one(
                {'user_id': self.user_id},
                sort=[('updated_at', -1)]
            )
            
            return {
                'total_conversations': total_conversations,
                'total_messages': total_messages,
                'recent_conversation': recent['conversation_id'] if recent else None,
                'recent_updated': recent['updated_at'].isoformat() if recent else None
            }
        
        except Exception as e:
            logger.error(f" Failed to get statistics: {e}")
            return {
                'total_conversations': 0,
                'total_messages': 0,
                'recent_conversation': None,
                'recent_updated': None
            }
    
    # ============================================
    # TỰ ĐỘNG TẠO TIÊU ĐỀ
    # ============================================
    
    def auto_generate_title(
        self,
        conversation_id: str,
        llm_func=None
    ) -> Optional[str]:
        """
        Tự động tạo title từ messages đầu tiên
        
        Args:
            conversation_id: ID của conversation
            llm_func: LLM function để generate title
        
        Returns:
            Generated title hoặc None
        """
        if not llm_func:
            return None
        
        try:
            # Lấy vài messages đầu
            messages = self.get_messages(conversation_id, limit=4)
            
            if not messages:
                return None
            
            # Xây dựng context
            context = "\n".join([
                f"{m['role'].title()}: {m['content'][:100]}"
                for m in messages[:4]
            ])
            
            prompt = f"""Based on this conversation, generate a short, descriptive title (max 50 characters).

Conversation:
{context}

Title (short and concise):"""

            # Gọi LLM
            import asyncio
            try:
                title = asyncio.run(llm_func(
                    prompt,
                    system_prompt="You generate short conversation titles.",
                    temperature=0.3,
                    max_tokens=50
                ))
            except:
                from backend.utils.llm_utils import call_llm
                title = call_llm(
                    prompt,
                    system_prompt="You generate short conversation titles.",
                    temperature=0.3,
                    max_tokens=50
                )
            
            # Làm sạch title
            title = title.strip().strip('"\'').strip()
            
            if title and len(title) <= 100:
                # Cập nhật title
                self.update_conversation_title(conversation_id, title)
                logger.info(f" Auto-generated title: {title}")
                return title
            
            return None
        
        except Exception as e:
            logger.error(f" Failed to auto-generate title: {e}")
            return None


# ============================================
# HÀM TIỆN ÍCH
# ============================================

def create_conversation_storage(user_id: str) -> ConversationStorage:
    """
    Factory function
    
    Usage:
        storage = create_conversation_storage('admin_00000000')
    """
    return ConversationStorage(user_id)