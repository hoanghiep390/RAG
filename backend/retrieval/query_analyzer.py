# backend/retrieval/query_analyzer.py
"""
Phân tích câu hỏi để quyết định retrieval
"""
import re
from typing import Dict, List, Literal, Optional
from dataclasses import dataclass
from difflib import SequenceMatcher

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
    """Bộ phân tích query nâng cao với nhận diện entity ngữ nghĩa"""
    
    def __init__(self, mongo_storage=None):
        """
        Tham số:
            mongo_storage: Instance MongoStorage cho tra cứu entity (tùy chọn)
        """
        self.stop_words = STOP_WORDS['en'] | STOP_WORDS['vi']
        self.mongo_storage = mongo_storage
        self._entity_cache = None
        self._cache_loaded = False
        
        # ✅ OPTIMIZED: Load cache once at initialization instead of per-query
        if mongo_storage:
            self._load_entity_cache()
    
    def _load_entity_cache(self):
        """✅ OPTIMIZED: Load entities once at init, reduced from 500 to 300 for speed"""
        if self._cache_loaded or not self.mongo_storage:
            return
        
        try:
            from backend.config import Config
            max_entities = Config.MAX_ENTITY_CACHE
            
            entities = self.mongo_storage.entities.find(
                {'user_id': self.mongo_storage.user_id},
                {'entity_name': 1, 'entity_type': 1}
            ).limit(max_entities)
            
            # Build cache: {entity_name_lower: entity_name}
            self._entity_cache = {}
            for e in entities:
                entity_name = e['entity_name']
                self._entity_cache[entity_name.lower()] = entity_name
            
            self._cache_loaded = True
            
            if self._entity_cache:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"✅ Loaded {len(self._entity_cache)} entities for query analysis")
        
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"❌ Failed to load entity cache: {e}")
            self._entity_cache = {}
    
    def _extract_entities_semantic(self, query: str) -> List[str]:
        """
        NÂNG CAO: Trích xuất entity ngữ nghĩa sử dụng tra cứu DB
        TỐI ƯU: Cache đã tải sẵn khi init, không cần tải lại
        
        Chiến lược khớp:
        1. Khớp chính xác (không phân biệt hoa thường)
        2. Khớp mờ (similarity > 0.90, giảm từ 0.85)
        3. Regex patterns (dự phòng)
        """
        entities = []
        
        # ✅ OPTIMIZED: No need to load cache here, already loaded at init
        if not self._entity_cache:
            # Fallback to regex if no cache
            return self._extract_entities_regex(query)
        
        query_lower = query.lower()
        
        # Level 1: Exact match (case-insensitive)
        for entity_lower, entity_name in self._entity_cache.items():
            if entity_lower in query_lower:
                entities.append(entity_name)
        
        # Level 2: Fuzzy match for multi-word entities
        # ✅ OPTIMIZED: Reduced from 5 to 3 words (1-3 word phrases only)
        # Split query into n-grams (1-3 words)
        query_tokens = query.split()
        for i in range(len(query_tokens)):
            for j in range(i+1, min(i+4, len(query_tokens)+1)):  # Up to 3-word phrases
                phrase = ' '.join(query_tokens[i:j]).lower()
                
                # Skip if already found exact match
                if phrase in [e.lower() for e in entities]:
                    continue
                
                # Fuzzy match against entity cache
                for entity_lower, entity_name in self._entity_cache.items():
                    # Skip short entities (risky for fuzzy match)
                    if len(entity_lower) < 4:
                        continue
                    
                    # ✅ OPTIMIZED: Increased threshold from 0.85 to 0.90 for fewer comparisons
                    score = SequenceMatcher(None, phrase, entity_lower).ratio()
                    if score > 0.90:
                        entities.append(entity_name)
                        break
        
        # Level 3: Regex fallback for entities not in DB
        if not entities:
            entities = self._extract_entities_regex(query)
        
        # Deduplicate
        entities = list(dict.fromkeys(entities))  # Preserve order
        
        return entities[:5]  # Top 5
    
    def _extract_entities_regex(self, query: str) -> List[str]:
        """Fallback: Extract entities using regex patterns"""
        entities = []
        
        for pattern in ENTITY_MARKERS:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        entities = list(set(entities))
        entities.sort(key=len, reverse=True)
        
        return entities[:5]
    
    def analyze(self, query: str) -> QueryAnalysis:
        """Phân tích query với semantic entity recognition"""
        query_lower = query.lower().strip()
    
        intent = self._detect_intent(query_lower)
        
        # ✅ ENHANCED: Use semantic entity extraction
        entities = self._extract_entities_semantic(query)
        
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
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords"""
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
