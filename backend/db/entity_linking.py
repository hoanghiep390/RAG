# backend/db/entity_linking.py
"""
 Liên kết Entity Nâng cao
Khớp mờ nhiều cấp độ để loại bỏ trùng lặp entity tốt hơn
"""

import re
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)


def normalize_entity_name(name: str) -> str:
    """
    Chuẩn hóa tên entity để so sánh
    - Loại bỏ khoảng trắng thừa
    - Chuyển thành chữ thường
    - Loại bỏ ký tự đặc biệt (cho khớp từ viết tắt)
    """
    # Chuẩn hóa cơ bản
    normalized = name.strip().lower()
    return normalized


def extract_acronym(name: str) -> str:
    """
    Trích xuất từ viết tắt từ tên entity
    Ví dụ:
        "OpenAI Inc." → "openaiinc"
        "GPT-4" → "gpt4"
        "United States" → "us"
    """
    # Loại bỏ tất cả ký tự không phải chữ-số
    clean = re.sub(r'[^a-zA-Z0-9]', '', name).lower()
    return clean


def is_acronym_match(name1: str, name2: str) -> bool:
    """
    Kiểm tra nếu một tên là từ viết tắt/rút gọn của tên kia
    
    Ví dụ:
        "GPT4" vs "GPT-4" → True
        "OpenAI Inc" vs "OpenAI" → True
        "US" vs "United States" → False (quá ngắn, rủi ro)
    """
    acronym1 = extract_acronym(name1)
    acronym2 = extract_acronym(name2)
    
    # Khớp chính xác sau khi loại bỏ ký tự đặc biệt
    if acronym1 == acronym2:
        return True
    
    
    if len(acronym1) >= 4 and len(acronym2) >= 4:
        if acronym1 in acronym2 or acronym2 in acronym1:
            return True
    
    return False


def calculate_similarity(name1: str, name2: str) -> float:
    """
    Tính độ tương đồng giữa hai tên entity
    Sử dụng SequenceMatcher ratio
    """
    normalized1 = normalize_entity_name(name1)
    normalized2 = normalize_entity_name(name2)
    
    return SequenceMatcher(None, normalized1, normalized2).ratio()


def fuzzy_match_entity(
    entity_name: str,
    existing_entities: List[Dict],
    strict_mode: bool = False,
    min_length: int = 3
) -> Optional[Tuple[str, float, str]]:
    """
    Khớp mờ nhiều cấp độ cho liên kết entity
    
    Các cấp độ khớp:
    1. Khớp chính xác (không phân biệt hoa thường)
    2. Độ tương đồng cao (>0.9) - cho chế độ nghiêm ngặt
    3. Độ tương đồng trung bình (>0.8) - cho chế độ bình thường  
    4. Khớp từ viết tắt/rút gọn
    
    Args:
        entity_name: Tên entity cần khớp
        existing_entities: Danh sách entity hiện có với trường 'entity_name'
        strict_mode: Sử dụng ngưỡng nghiêm ngặt hơn (0.9 vs 0.8)
        min_length: Độ dài tối thiểu của tên entity (tránh khớp tên ngắn)
    
    Returns:
        Tuple của (canonical_name, similarity_score, match_type) hoặc None
        match_type: 'exact', 'high_similarity', 'medium_similarity', 'acronym'
    """
    if not entity_name or len(entity_name) < min_length:
        return None
    
    entity_lower = normalize_entity_name(entity_name)
    
    # Cấp 1: Khớp chính xác (không phân biệt hoa thường)
    for existing in existing_entities:
        existing_name = existing['entity_name']
        if entity_lower == normalize_entity_name(existing_name):
            return (existing_name, 1.0, 'exact')
    
    # Cấp 2-3: Khớp theo độ tương đồng
    threshold = 0.9 if strict_mode else 0.8
    best_match = None
    best_score = threshold
    
    for existing in existing_entities:
        existing_name = existing['entity_name']
        score = calculate_similarity(entity_name, existing_name)
        
        if score > best_score:
            best_score = score
            best_match = existing_name
    
    if best_match:
        match_type = 'high_similarity' if best_score >= 0.9 else 'medium_similarity'
        return (best_match, best_score, match_type)
    
    # Cấp 4: Khớp từ viết tắt
    # Chỉ cho entities có độ dài hợp lý
    if len(entity_name) >= 4:
        for existing in existing_entities:
            existing_name = existing['entity_name']
            if len(existing_name) >= 4 and is_acronym_match(entity_name, existing_name):
                # Tính độ tương đồng để ghi log
                score = calculate_similarity(entity_name, existing_name)
                return (existing_name, score, 'acronym')
    
    return None


def link_entities_batch(
    entities_dict: Dict[str, List[Dict]],
    existing_entities: List[Dict],
    strict_mode: bool = False
) -> Tuple[Dict[str, str], Dict[str, Tuple[float, str]]]:
    """
    Liên kết entity theo batch
    
    Args:
        entities_dict: Dict của {entity_name: [entity_dicts]}
        existing_entities: Danh sách entities hiện có từ DB
        strict_mode: Sử dụng ngưỡng khớp nghiêm ngặt hơn
    
    Returns:
        Tuple của:
        - canonical_mapping: Dict của {original_name: canonical_name}
        - match_info: Dict của {original_name: (similarity_score, match_type)}
    """
    canonical_mapping = {}
    match_info = {}
    
    for entity_name in entities_dict.keys():
        match_result = fuzzy_match_entity(
            entity_name,
            existing_entities,
            strict_mode=strict_mode
        )
        
        if match_result:
            canonical_name, score, match_type = match_result
            canonical_mapping[entity_name] = canonical_name
            match_info[entity_name] = (score, match_type)
            
            logger.debug(
                f" Linked '{entity_name}' → '{canonical_name}' "
                f"(score: {score:.2f}, type: {match_type})"
            )
        else:
            # Không khớp, sử dụng tên gốc
            canonical_mapping[entity_name] = entity_name
    
    return canonical_mapping, match_info


def get_linking_statistics(match_info: Dict[str, Tuple[float, str]]) -> Dict:
    """
    Lấy thống kê về liên kết entity
    
    Returns:
        Dict với số lượng theo loại khớp
    """
    stats = {
        'total': len(match_info),
        'exact': 0,
        'high_similarity': 0,
        'medium_similarity': 0,
        'acronym': 0,
        'no_match': 0
    }
    
    for entity_name, (score, match_type) in match_info.items():
        stats[match_type] = stats.get(match_type, 0) + 1
    
    return stats


# ================= Export =================

__all__ = [
    'fuzzy_match_entity',
    'link_entities_batch',
    'is_acronym_match',
    'calculate_similarity',
    'normalize_entity_name',
    'get_linking_statistics'
]
