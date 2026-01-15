# backend/db/feedback_storage.py
"""
 Feedback Storage - Lưu đánh giá và feedback từ user
"""
from typing import List, Dict, Optional
from datetime import datetime
from backend.config import get_mongodb
import logging

logger = logging.getLogger(__name__)

class FeedbackStorage:
    """
    MongoDB storage cho user feedback
    
    Collection: feedbacks
    - user_id: ID của user
    - conversation_id: ID của conversation
    - message_index: Vị trí message trong conversation
    - rating: Điểm đánh giá (1-5)
    - feedback_text: Nội dung feedback (optional)
    - created_at: Thời gian tạo
    """
    
    def __init__(self, user_id: str):
        """
        Tham số:
            user_id: User ID cho cô lập dữ liệu
        """
        self.user_id = user_id
        try:
            self.db = get_mongodb()
            self.feedbacks = self.db['feedbacks']
            
            self._create_indexes()
            logger.info(f" FeedbackStorage đã khởi tạo cho user: {user_id}")
        except Exception as e:
            logger.error(f" Không thể khởi tạo FeedbackStorage: {e}")
            raise
    
    def _create_indexes(self):
        """Tạo indexes cho hiệu suất"""
        try:
            # Unique index for (user_id, conversation_id, message_index)
            self.feedbacks.create_index([
                ('user_id', 1),
                ('conversation_id', 1),
                ('message_index', 1)
            ], unique=True)
            
            # Index for user queries
            self.feedbacks.create_index([('user_id', 1), ('created_at', -1)])
            
            logger.debug(" Feedback indexes created")
        except Exception as e:
            logger.warning(f" Index creation warning: {e}")
    
    def save_feedback(
        self,
        conversation_id: str,
        message_index: int,
        rating: int,
        feedback_text: Optional[str] = None
    ) -> bool:
        """
        Lưu feedback cho một message
        
        Args:
            conversation_id: ID của conversation
            message_index: Vị trí message (0-indexed)
            rating: Điểm đánh giá (1-5)
            feedback_text: Nội dung feedback (optional)
        
        Returns:
            True nếu thành công
        """
        try:
            # Validate rating
            if not (1 <= rating <= 5):
                logger.error(f" Rating không hợp lệ: {rating}")
                return False
            
            feedback = {
                'user_id': self.user_id,
                'conversation_id': conversation_id,
                'message_index': message_index,
                'rating': rating,
                'feedback_text': feedback_text or "",
                'created_at': datetime.now()
            }
            
            # Upsert (update if exists, insert if not)
            result = self.feedbacks.update_one(
                {
                    'user_id': self.user_id,
                    'conversation_id': conversation_id,
                    'message_index': message_index
                },
                {'$set': feedback},
                upsert=True
            )
            
            logger.info(
                f" Đã lưu feedback: conv={conversation_id[:8]}..., "
                f"msg={message_index}, rating={rating}"
            )
            return True
        
        except Exception as e:
            logger.error(f" Không thể lưu feedback: {e}")
            return False
    
    def get_feedback(
        self,
        conversation_id: str,
        message_index: int
    ) -> Optional[Dict]:
        """
        Lấy feedback cho một message
        
        Args:
            conversation_id: ID của conversation
            message_index: Vị trí message
        
        Returns:
            Feedback dict hoặc None
        """
        try:
            feedback = self.feedbacks.find_one({
                'user_id': self.user_id,
                'conversation_id': conversation_id,
                'message_index': message_index
            })
            return feedback
        except Exception as e:
            logger.error(f" Không thể lấy feedback: {e}")
            return None
    
    def list_user_feedbacks(
        self,
        limit: int = 50,
        skip: int = 0
    ) -> List[Dict]:
        """
        Lấy danh sách feedbacks của user
        
        Args:
            limit: Số lượng feedbacks
            skip: Bỏ qua N feedbacks đầu tiên
        
        Returns:
            List of feedback dicts
        """
        try:
            feedbacks = list(
                self.feedbacks.find({'user_id': self.user_id})
                .sort('created_at', -1)
                .skip(skip)
                .limit(limit)
            )
            return feedbacks
        except Exception as e:
            logger.error(f" Không thể liệt kê feedbacks: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """
        Lấy thống kê feedbacks của user
        
        Returns:
            Dict với stats
        """
        try:
            total_feedbacks = self.feedbacks.count_documents({'user_id': self.user_id})
            
            if total_feedbacks == 0:
                return {
                    'total_feedbacks': 0,
                    'average_rating': 0.0,
                    'rating_distribution': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                }
            
            # Calculate average rating
            pipeline = [
                {'$match': {'user_id': self.user_id}},
                {'$group': {
                    '_id': None,
                    'avg_rating': {'$avg': '$rating'}
                }}
            ]
            
            result = list(self.feedbacks.aggregate(pipeline))
            avg_rating = result[0]['avg_rating'] if result else 0.0
            
            # Rating distribution
            distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            for rating in range(1, 6):
                count = self.feedbacks.count_documents({
                    'user_id': self.user_id,
                    'rating': rating
                })
                distribution[rating] = count
            
            return {
                'total_feedbacks': total_feedbacks,
                'average_rating': round(avg_rating, 2),
                'rating_distribution': distribution
            }
        
        except Exception as e:
            logger.error(f" Không thể lấy thống kê: {e}")
            return {
                'total_feedbacks': 0,
                'average_rating': 0.0,
                'rating_distribution': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            }
    
    def delete_feedback(
        self,
        conversation_id: str,
        message_index: int
    ) -> bool:
        """
        Xóa feedback
        
        Args:
            conversation_id: ID của conversation
            message_index: Vị trí message
        
        Returns:
            True nếu thành công
        """
        try:
            result = self.feedbacks.delete_one({
                'user_id': self.user_id,
                'conversation_id': conversation_id,
                'message_index': message_index
            })
            
            if result.deleted_count > 0:
                logger.info(f" Đã xóa feedback: conv={conversation_id[:8]}..., msg={message_index}")
                return True
            return False
        
        except Exception as e:
            logger.error(f" Không thể xóa feedback: {e}")
            return False


# ============================================
# UTILITY FUNCTIONS
# ============================================

def create_feedback_storage(user_id: str) -> FeedbackStorage:
    """
    Factory function
    
    Usage:
        storage = create_feedback_storage('user_123')
    """
    return FeedbackStorage(user_id)
