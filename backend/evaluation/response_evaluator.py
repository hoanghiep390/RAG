# backend/evaluation/response_evaluator.py
"""
Hệ Thống Đánh Giá Tự Động (LLM-as-a-Judge)
Đánh giá chất lượng câu trả lời theo 3 tiêu chí: Relevancy, Faithfulness, Response Time
"""
from typing import Dict, Optional, Callable
import asyncio
import json
import logging
import re

logger = logging.getLogger(__name__)

class ResponseEvaluator:
    """
    LLM-as-a-Judge để đánh giá chất lượng câu trả lời
    
    Sử dụng LLM để đánh giá:
    1. Relevancy: Độ liên quan giữa câu hỏi và câu trả lời
    2. Faithfulness: Độ trung thực của câu trả lời so với nguồn tài liệu
    """
    
    def __init__(self):
        """Khởi tạo evaluator"""
        pass
    
    async def evaluate_relevancy(
        self,
        question: str,
        answer: str,
        llm_func: Callable
    ) -> Dict:
        """
        Đánh giá độ liên quan giữa câu hỏi và câu trả lời
        
        Args:
            question: Câu hỏi của người dùng
            answer: Câu trả lời của chatbot
            llm_func: Hàm gọi LLM (async)
        
        Returns:
            Dict với 'score' (1-5) và 'reason' (lý do)
        """
        try:
            prompt = f"""Bạn là một chuyên gia đánh giá chất lượng câu trả lời AI.

Nhiệm vụ: Đánh giá mức độ liên quan giữa câu hỏi và câu trả lời.

Câu hỏi: {question}

Câu trả lời: {answer}

Tiêu chí đánh giá (1-5):
1 = Hoàn toàn không liên quan - Câu trả lời không đề cập đến nội dung câu hỏi
2 = Ít liên quan - Câu trả lời chỉ đề cập một phần nhỏ của câu hỏi
3 = Trung bình - Câu trả lời liên quan nhưng thiếu chi tiết hoặc không trực tiếp
4 = Rất liên quan - Câu trả lời trực tiếp giải quyết câu hỏi với đầy đủ thông tin
5 = Hoàn hảo - Câu trả lời trực tiếp, đầy đủ và chính xác giải quyết câu hỏi

Hãy trả về JSON với format sau (KHÔNG thêm markdown code block):
{{"score": <1-5>, "reason": "<giải thích ngắn gọn bằng tiếng Việt>"}}"""

            # Gọi LLM
            result = await llm_func(
                prompt,
                system_prompt="Bạn là một chuyên gia đánh giá. Chỉ trả về JSON, không thêm text khác.",
                temperature=0.0,
                max_tokens=300
            )
            
            # Parse JSON từ response
            parsed = self._parse_json_response(result)
            
            # Validate
            if not isinstance(parsed.get('score'), int) or not (1 <= parsed['score'] <= 5):
                logger.warning(f"Invalid relevancy score: {parsed.get('score')}, defaulting to 3")
                parsed['score'] = 3
            
            if not parsed.get('reason'):
                parsed['reason'] = "Không có lý do cụ thể"
            
            logger.info(f"✅ Relevancy evaluated: {parsed['score']}/5")
            return parsed
        
        except Exception as e:
            logger.error(f"❌ Lỗi đánh giá relevancy: {e}")
            return {
                'score': 3,
                'reason': f"Lỗi đánh giá: {str(e)}"
            }
    
    async def evaluate_faithfulness(
        self,
        answer: str,
        context: str,
        llm_func: Callable
    ) -> Dict:
        """
        Đánh giá độ trung thực của câu trả lời so với nguồn tài liệu
        
        Args:
            answer: Câu trả lời của chatbot
            context: Nguồn tài liệu được trích dẫn (retrieved context)
            llm_func: Hàm gọi LLM (async)
        
        Returns:
            Dict với 'score' (1-5) và 'reason' (lý do)
        """
        try:
            # Truncate context nếu quá dài
            max_context_length = 2000
            if len(context) > max_context_length:
                context = context[:max_context_length] + "..."
            
            prompt = f"""Bạn là một chuyên gia đánh giá độ trung thực của thông tin.

Nhiệm vụ: Đánh giá mức độ trung thực của câu trả lời so với nguồn tài liệu.

Nguồn tài liệu (Context):
{context}

Câu trả lời:
{answer}

Tiêu chí đánh giá (1-5):
1 = Sai lệch hoàn toàn - Câu trả lời mâu thuẫn với nguồn tài liệu
2 = Ít trung thực - Câu trả lời có nhiều thông tin không có trong nguồn
3 = Trung bình - Câu trả lời một phần dựa trên nguồn, một phần suy luận
4 = Rất trung thực - Câu trả lời chủ yếu dựa trên nguồn tài liệu
5 = Hoàn toàn trung thực - Mọi thông tin đều có trong nguồn tài liệu

Hãy trả về JSON với format sau (KHÔNG thêm markdown code block):
{{"score": <1-5>, "reason": "<giải thích ngắn gọn bằng tiếng Việt>"}}"""

            # Gọi LLM
            result = await llm_func(
                prompt,
                system_prompt="Bạn là một chuyên gia đánh giá. Chỉ trả về JSON, không thêm text khác.",
                temperature=0.0,
                max_tokens=300
            )
            
            # Parse JSON từ response
            parsed = self._parse_json_response(result)
            
            # Validate
            if not isinstance(parsed.get('score'), int) or not (1 <= parsed['score'] <= 5):
                logger.warning(f"Invalid faithfulness score: {parsed.get('score')}, defaulting to 3")
                parsed['score'] = 3
            
            if not parsed.get('reason'):
                parsed['reason'] = "Không có lý do cụ thể"
            
            logger.info(f"✅ Faithfulness evaluated: {parsed['score']}/5")
            return parsed
        
        except Exception as e:
            logger.error(f"❌ Lỗi đánh giá faithfulness: {e}")
            return {
                'score': 3,
                'reason': f"Lỗi đánh giá: {str(e)}"
            }
    
    async def evaluate_response(
        self,
        question: str,
        answer: str,
        context: str,
        response_time_ms: float,
        llm_func: Callable
    ) -> Dict:
        """
        Đánh giá tổng hợp câu trả lời theo 3 tiêu chí
        
        Args:
            question: Câu hỏi của người dùng
            answer: Câu trả lời của chatbot
            context: Nguồn tài liệu được trích dẫn
            response_time_ms: Thời gian phản hồi (milliseconds)
            llm_func: Hàm gọi LLM (async)
        
        Returns:
            Dict với các tiêu chí đánh giá
        """
        try:
            logger.info("🔍 Bắt đầu đánh giá tự động...")
            
            # Đánh giá song song để tiết kiệm thời gian
            relevancy_task = self.evaluate_relevancy(question, answer, llm_func)
            faithfulness_task = self.evaluate_faithfulness(answer, context, llm_func)
            
            relevancy, faithfulness = await asyncio.gather(
                relevancy_task,
                faithfulness_task
            )
            
            result = {
                'relevancy_score': relevancy['score'],
                'relevancy_reason': relevancy['reason'],
                'faithfulness_score': faithfulness['score'],
                'faithfulness_reason': faithfulness['reason'],
                'response_time_ms': round(response_time_ms, 2),
                'auto_evaluated': True
            }
            
            logger.info(f"✅ Đánh giá hoàn tất: R={result['relevancy_score']}/5, F={result['faithfulness_score']}/5, T={result['response_time_ms']}ms")
            return result
        
        except Exception as e:
            logger.error(f"❌ Lỗi đánh giá tổng hợp: {e}")
            return {
                'relevancy_score': 3,
                'relevancy_reason': f"Lỗi: {str(e)}",
                'faithfulness_score': 3,
                'faithfulness_reason': f"Lỗi: {str(e)}",
                'response_time_ms': round(response_time_ms, 2),
                'auto_evaluated': True
            }
    
    def _parse_json_response(self, response: str) -> Dict:
        """
        Parse JSON từ LLM response (xử lý cả markdown code blocks)
        
        Args:
            response: Response từ LLM
        
        Returns:
            Parsed JSON dict
        """
        try:
            # Loại bỏ markdown code blocks nếu có
            response = response.strip()
            
            # Tìm JSON trong markdown code block
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                response = json_match.group(1)
            
            # Tìm JSON object đầu tiên
            json_match = re.search(r'\{.*?\}', response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            
            # Parse JSON
            return json.loads(response)
        
        except Exception as e:
            logger.error(f"❌ Không thể parse JSON: {e}, response: {response[:200]}")
            # Fallback: trả về default
            return {
                'score': 3,
                'reason': "Không thể parse kết quả đánh giá"
            }


# ============================================
# UTILITY FUNCTIONS
# ============================================

def create_evaluator() -> ResponseEvaluator:
    """
    Factory function
    
    Usage:
        evaluator = create_evaluator()
    """
    return ResponseEvaluator()
