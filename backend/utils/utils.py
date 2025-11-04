# backend/utils/utils.py
"""
Utilities for core modules
"""
        
import logging
import sys
from pathlib import Path
from datetime import datetime

# Setup logging directory
log_dir = Path("backend/data/logs")
log_dir.mkdir(parents=True, exist_ok=True)

# Create logger
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

# File handler
log_file = log_dir / f"lightrag_{datetime.now().strftime('%Y%m%d')}.log"
file_handler = logging.FileHandler(log_file, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
file_handler.setFormatter(file_formatter)

# Add handlers
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# Prevent propagation to root logger
logger.propagate = False