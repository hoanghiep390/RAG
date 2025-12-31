# backend/utils/file_utils.py
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def ensure_dir(path):
    """Đảm bảo thư mục tồn tại"""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_uploaded_file(uploaded_file, user_id: Optional[str] = None) -> str:
    upload_dir = Path("backend/data/uploads")
    if user_id:
        upload_dir = upload_dir / user_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = "".join(c for c in uploaded_file.name if c.isalnum() or c in "._- ")
    filename = f"{timestamp}_{safe_filename}"
    filepath = upload_dir / filename

    try:
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer()) 
        logger.info(f" Saved uploaded file: {filepath}")
    except Exception as e:
        logger.error(f" Failed to save file {filename}: {str(e)}")
        raise IOError(f"Failed to save file {filename}: {str(e)}")

    return str(filepath.resolve())




