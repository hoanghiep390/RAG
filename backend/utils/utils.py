# backend/utils/utils.py
"""
Tiện ích cho các module core
"""
        
import logging
import sys
from pathlib import Path
from datetime import datetime

# Bắt buộc UTF-8 trên Windows
if sys.platform == 'win32':
    import os
    os.system('chcp 65001 > nul')

# Thiết lập thư mục logging
log_dir = Path("backend/data/logs")
log_dir.mkdir(parents=True, exist_ok=True)

# Tạo logger
logger = logging.getLogger("lightrag")
logger.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler.setFormatter(console_formatter)

# File handler với UTF-8
log_file = log_dir / f"lightrag_{datetime.now().strftime('%Y%m%d')}.log"
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(file_formatter)

# Thêm handlers
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

logger.propagate = False

def log_vietnamese(message: str, level: str = "info"):
    """Log văn bản tiếng Việt an toàn"""
    if isinstance(message, bytes):
        message = message.decode('utf-8', errors='replace')
    
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
    """Kiểm tra chunk có nội dung hợp lệ"""
    content = chunk.get('content', '')
    
    if not content.strip():
        logger.warning(f" Empty chunk: {chunk.get('chunk_id')}")
        return False
    
    if len(content) < 10:
        logger.warning(f" Short chunk: {chunk.get('chunk_id')} ({len(content)} chars)")
        return False
    
    encoding_artifacts = ['Ã', '¡', '©', 'á»', 'Ä']
    artifact_count = sum(content.count(artifact) for artifact in encoding_artifacts)
    
    if artifact_count > len(content) * 0.1:
        logger.warning(f" Encoding issue: {chunk.get('chunk_id')}")
        return False
    
    return True