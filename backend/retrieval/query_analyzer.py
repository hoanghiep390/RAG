# backend/retrieval/query_analyzer.py
"""
Phân tích câu hỏi để quyết định retrieval
"""
import re
from typing import Dict, List, Literal
from dataclasses import dataclass

@dataclass
class QueryAnalysis:
    """Kết quả phân tích query"""
    original_query: str
    intent: Literal['fact', 'reasoning', 'comparison', 'summary']
    entities: List[str]
    keywords: List[str]
    retrieval_mode: Literal['vector', 'graph', 'hybrid']
    top_k: int

INTENT_PATTERNS = {
    'fact': [
        r'\b(what is|who is|when|where|which|define)\b',
        r'\b(là gì|là ai|khi nào|ở đâu|định nghĩa)\b'
    ],
    'comparison': [
        r'\b(compare|difference|vs|versus|better)\b',
        r'\b(so sánh|khác nhau|tốt hơn)\b'
    ],
    'summary': [
        r'\b(summarize|overview|explain|describe)\b',
        r'\b(tóm tắt|tổng quan|giải thích|mô tả)\b'
    ],
    'reasoning': [
        r'\b(why|how|reason|cause)\b',
        r'\b(tại sao|như thế nào|lý do|nguyên nhân)\b'
    ]
}

ENTITY_MARKERS = [
    r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', 
    r'(?:GPT|API|LLM|RAG|AI|ML|DL)',        
]

STOP_WORDS = {
    'en': {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for'},
    'vi': {'là', 'của', 'và', 'có', 'được', 'một', 'các', 'này', 'đó', 'cho', 'từ', 'trong'}
}

class QueryAnalyzer:
    """Simple query analyzer """
    
    def __init__(self):
        self.stop_words = STOP_WORDS['en'] | STOP_WORDS['vi']
    
    def analyze(self, query: str) -> QueryAnalysis:
        """Phân tích query đơn giản"""
        query_lower = query.lower().strip()
    
        intent = self._detect_intent(query_lower)
        
        entities = self._extract_entities(query)
        
        keywords = self._extract_keywords(query_lower)
        
        retrieval_mode = self._decide_mode(intent, entities)
        
        top_k = self._decide_top_k(intent)
        
        return QueryAnalysis(
            original_query=query,
            intent=intent,
            entities=entities,
            keywords=keywords,
            retrieval_mode=retrieval_mode,
            top_k=top_k
        )
    
    def _detect_intent(self, query: str) -> str:
        """Detect intent bằng regex patterns"""
        for intent, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return intent
        return 'fact' 
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities đơn giản (proper nouns + acronyms)"""
        entities = []
        
        for pattern in ENTITY_MARKERS:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        entities = list(set(entities))
        entities.sort(key=len, reverse=True)
        
        return entities[:5]  
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords """
        tokens = re.findall(r'\b\w+\b', query)
        
        keywords = [
            t for t in tokens 
            if len(t) > 2 and t not in self.stop_words
        ]
        return keywords[:10]  
    
    def _decide_mode(self, intent: str, entities: List[str]) -> str:
        """Quyết định retrieval mode"""
        if entities:
            return 'hybrid'  
        
        if intent in ['reasoning', 'comparison']:
            return 'hybrid'
    
        return 'vector'
    
    def _decide_top_k(self, intent: str) -> int:
        """Quyết định số lượng results"""
        if intent == 'summary':
            return 10  
        elif intent == 'comparison':
            return 8
        else:
            return 5  
