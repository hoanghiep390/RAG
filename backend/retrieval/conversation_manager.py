# backend/retrieval/conversation_manager.py
"""
Quản lý Hội thoại
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Cấu trúc message đơn giản"""
    role: str  # 'user' hoặc 'assistant'
    content: str
    message_id: Optional[str] = None

class ConversationManager:
    """
    Quản lý lịch sử chat với MongoDB persistence
    
    Features:
    1. Lưu lịch sử (sliding window)
    2. Rewrite query với context
    3. Format history cho LLM
    4. MongoDB auto-save
    """
    
    def __init__(
        self, 
        max_history: int = 5,
        conv_storage=None,
        conversation_id: Optional[str] = None
    ):
        """
        Args:
            max_history: Số lượng turns lưu (mặc định: 5 = 10 messages)
            conv_storage: Instance ConversationStorage (tùy chọn)
            conversation_id: ID hội thoại hiện tại (tùy chọn)
        """
        self.max_history = max_history
        self.history: List[Message] = []
        
        # Tích hợp MongoDB
        self.conv_storage = conv_storage
        self.conversation_id = conversation_id
    
    def set_conversation(self, conversation_id: str, conv_storage):
        """Chuyển sang hội thoại khác"""
        self.conversation_id = conversation_id
        self.conv_storage = conv_storage
        
        # Tải lịch sử từ MongoDB
        if conv_storage:
            try:
                messages = conv_storage.get_messages(conversation_id)
                self.history = [
                    Message(
                        role=m['role'],
                        content=m['content'],
                        message_id=m.get('message_id')
                    )
                    for m in messages
                ]
                logger.info(f" Đã tải {len(self.history)} messages từ MongoDB")
            except Exception as e:
                logger.error(f" Không thể tải messages: {e}")
                self.history = []
    
    def add_message(self, role: str, content: str, save_to_db: bool = True):
        """
        Thêm message vào history
        
        Args:
            role: 'user' hoặc 'assistant'
            content: Nội dung message
            save_to_db: Tự động lưu vào MongoDB nếu có conv_storage
        """
        message = Message(role=role, content=content)
        self.history.append(message)
        
        # Sliding window: chỉ giữ max_history turns gần nhất
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]
        
        # Tự động lưu vào MongoDB
        if save_to_db and self.conv_storage and self.conversation_id:
            try:
                message_id = self.conv_storage.add_message(
                    conversation_id=self.conversation_id,
                    role=role,
                    content=content
                )
                message.message_id = message_id
                logger.debug(f" Đã lưu message vào MongoDB: {message_id}")
            except Exception as e:
                logger.error(f" Không thể lưu message: {e}")
    
    def get_context_for_llm(self) -> List[Dict]:
        """
        Format history cho LLM API
        
        Returns:
            List[{"role": "user/assistant", "content": "..."}]
        """
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.history[-(self.max_history * 2):]
        ]
    
    def rewrite_query(self, current_query: str, llm_func=None) -> str:
        """
        Rewrite query với context từ lịch sử
        
        Args:
            current_query: Query hiện tại
            llm_func: LLM function (optional)
        
        Returns:
            Rewritten query (standalone)
        """
        if not self.history or not llm_func:
            return current_query
        
        if len(current_query.split()) > 10:
            return current_query
        
        try:
            history_text = self._format_history_for_rewrite()
            
            rewrite_prompt = f"""Given the conversation history, rewrite the current query to be self-contained.

Conversation History:
{history_text}

Current Query: {current_query}

Instructions:
1. If the query uses pronouns (it, that, this, they), replace with specific entities
2. If the query is a follow-up, add necessary context
3. Keep the query concise but complete
4. If already clear, return unchanged

Rewritten Query:"""

            import asyncio
            try:
                rewritten = asyncio.run(llm_func(
                    rewrite_prompt,
                    system_prompt="You are a query rewriting assistant.",
                    temperature=0.0,
                    max_tokens=200
                ))
            except:
                from backend.utils.llm_utils import call_llm
                rewritten = call_llm(
                    rewrite_prompt,
                    system_prompt="You are a query rewriting assistant.",
                    temperature=0.0,
                    max_tokens=200
                )
            
            rewritten = rewritten.strip().strip('"\'')
            
            if rewritten and len(rewritten.split()) <= 50:
                logger.info(f" Đã viết lại: '{current_query}' → '{rewritten}'")
                return rewritten
            
            return current_query
        
        except Exception as e:
            logger.error(f" Lỗi viết lại: {e}")
            return current_query
    
    def _format_history_for_rewrite(self) -> str:
        """Format history cho rewrite prompt"""
        lines = []
        for msg in self.history[-(self.max_history * 2):]:
            prefix = "User" if msg.role == "user" else "Assistant"
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)
    
    def clear(self):
        """Clear lịch sử"""
        self.history = []
    
    def get_summary(self) -> Dict:
        """Get conversation summary"""
        return {
            'total_messages': len(self.history),
            'turns': len(self.history) // 2,
            'last_user_query': next((m.content for m in reversed(self.history) if m.role == 'user'), None),
            'last_bot_response': next((m.content for m in reversed(self.history) if m.role == 'assistant'), None),
            'conversation_id': self.conversation_id
        }


# Hàm Tiện Ích
def create_conversation_manager(
    max_history: int = 5,
    conv_storage=None,
    conversation_id: Optional[str] = None
) -> ConversationManager:
    """
    Hàm factory
    
    Cách dùng:
        conv_manager = create_conversation_manager(
            max_history=5,
            conv_storage=storage,
            conversation_id='conv_123'
        )
    """
    return ConversationManager(
        max_history=max_history,
        conv_storage=conv_storage,
        conversation_id=conversation_id
    )


def format_history_for_prompt(history: List[Message], max_turns: int = 3) -> str:
    """Format lịch sử thành text cho system prompt"""
    if not history:
        return ""
    
    lines = ["Previous conversation:"]
    recent_history = history[-(max_turns * 2):]
    
    for msg in recent_history:
        prefix = "User" if msg.role == "user" else "Assistant"
        content = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
        lines.append(f"{prefix}: {content}")
    
    return "\n".join(lines)