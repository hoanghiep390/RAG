# backend/utils/utils.py
"""
Module tiện ích cho các module core
Cung cấp cấu hình logging và các hàm tiện ích cho xử lý dữ liệu
"""
        
import logging
import sys
from pathlib import Path
from datetime import datetime


if sys.platform == 'win32':
    import os
    os.system('chcp 65001 > nul')  

# ============================================
# LOGGING
# ============================================

# Thiết lập thư mục lưu log files
log_dir = Path("backend/data/logs")
log_dir.mkdir(parents=True, exist_ok=True)

# Tạo logger chính cho ứng dụng
logger = logging.getLogger("lightrag")
logger.setLevel(logging.INFO)

#  hiển thị log ra màn hình
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(console_formatter)


log_file = log_dir / f"lightrag_{datetime.now().strftime('%Y%m%d')}.log"
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)  
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(file_formatter)


if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


logger.propagate = False



def log_vietnamese(message: str, level: str = "info"):
    """
    Log văn bản tiếng Việt một cách an toàn
    
    Args:
        message: Nội dung cần log
        level: Mức độ log (debug/info/warning/error)
    """
    # Chuyển đổi bytes sang string nếu cần
    if isinstance(message, bytes):
        message = message.decode('utf-8', errors='replace')
    
    # Log theo level tương ứng
    if level == "debug":
        logger.debug(message)
    elif level == "info":
        logger.info(message)
    elif level == "warning":
        logger.warning(message)
    elif level == "error":
        logger.error(message)
    else:
        logger.info(message)

def verify_chunk_content(chunk: dict) -> bool:
    """
    Kiểm tra chunk có nội dung hợp lệ hay không
    
    Args:
        chunk: Dictionary chứa thông tin chunk cần kiểm tra
        
    Returns:
        bool: True nếu chunk hợp lệ, False nếu không
    """
    content = chunk.get('content', '')
    
    # Kiểm tra chunk rỗng
    if not content.strip():
        logger.warning(f" Empty chunk: {chunk.get('chunk_id')}")
        return False
    
    # Kiểm tra chunk quá ngắn (< 10 ký tự)
    if len(content) < 10:
        logger.warning(f" Short chunk: {chunk.get('chunk_id')} ({len(content)} chars)")
        return False
    
    # Kiểm tra lỗi encoding 
    encoding_artifacts = ['Ã', '¡', '©', 'á»', 'Ä']
    artifact_count = sum(content.count(artifact) for artifact in encoding_artifacts)
    
    # Nếu có quá nhiều ký tự lỗi (>10% nội dung) thì chunk lỗi
    if artifact_count > len(content) * 0.1:
        logger.warning(f" Encoding issue: {chunk.get('chunk_id')}")
        return False
    
    return True