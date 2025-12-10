# backend/retrieval/conversation_manager.py
"""
💬 Conversation Manager 
"""
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Simple message structure"""
    role: str  # 'user' or 'assistant'
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
            max_history: Số lượng turns lưu (default: 5 = 10 messages)
            conv_storage: ConversationStorage instance (optional)
            conversation_id: Current conversation ID (optional)
        """
        self.max_history = max_history
        self.history: List[Message] = []
        
        # MongoDB integration
        self.conv_storage = conv_storage
        self.conversation_id = conversation_id
    
    def set_conversation(self, conversation_id: str, conv_storage):
        """Switch to different conversation"""
        self.conversation_id = conversation_id
        self.conv_storage = conv_storage
        
        # Load history from MongoDB
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
                logger.info(f" Loaded {len(self.history)} messages from MongoDB")
            except Exception as e:
                logger.error(f" Failed to load messages: {e}")
                self.history = []
    
    def add_message(self, role: str, content: str, save_to_db: bool = True):
        """
        Thêm message vào history
        
        Args:
            role: 'user' or 'assistant'
            content: Message content
            save_to_db: Auto-save to MongoDB if conv_storage available
        """
        message = Message(role=role, content=content)
        self.history.append(message)
        
        # Sliding window: chỉ giữ max_history turns gần nhất
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]
        
        # Auto-save to MongoDB
        if save_to_db and self.conv_storage and self.conversation_id:
            try:
                message_id = self.conv_storage.add_message(
                    conversation_id=self.conversation_id,
                    role=role,
                    content=content
                )
                message.message_id = message_id
                logger.debug(f" Saved message to MongoDB: {message_id}")
            except Exception as e:
                logger.error(f" Failed to save message: {e}")
    
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
                logger.info(f" Rewritten: '{current_query}' → '{rewritten}'")
                return rewritten
            
            return current_query
        
        except Exception as e:
            logger.error(f" Rewrite error: {e}")
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


# Utility Functions
def create_conversation_manager(
    max_history: int = 5,
    conv_storage=None,
    conversation_id: Optional[str] = None
) -> ConversationManager:
    """
    Factory function
    
    Usage:
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
    """Format history thành text cho system prompt"""
    if not history:
        return ""
    
    lines = ["Previous conversation:"]
    recent_history = history[-(max_turns * 2):]
    
    for msg in recent_history:
        prefix = "User" if msg.role == "user" else "Assistant"
        content = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
        lines.append(f"{prefix}: {content}")
    
    return "\n".join(lines)