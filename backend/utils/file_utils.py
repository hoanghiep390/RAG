# backend/utils/file_utils.py
"""
Module tiện ích xử lý file và thư mục
Cung cấp các hàm hỗ trợ cho việc quản lý file upload và thư mục trong hệ thống
"""

import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def ensure_dir(path):
    """
    Đảm bảo thư mục tồn tại, tạo mới nếu chưa có
    
    Args:
        path: Đường dẫn thư mục cần kiểm tra/tạo
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_uploaded_file(uploaded_file, user_id: Optional[str] = None) -> str:
    """
    Lưu file được upload từ người dùng vào thư mục uploads
    
    Args:
        uploaded_file: File object từ Streamlit uploader
        user_id: ID người dùng (tùy chọn), dùng để tạo thư mục riêng cho từng user
        
    Returns:
        str: Đường dẫn tuyệt đối đến file đã lưu
        
    Raises:
        IOError: Nếu không thể lưu file
    """
    # Xác định thư mục upload, tạo thư mục riêng cho user nếu có user_id
    upload_dir = Path("backend/data/uploads")
    if user_id:
        upload_dir = upload_dir / user_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Tạo tên file an toàn để tránh trùng lặp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = "".join(c for c in uploaded_file.name if c.isalnum() or c in "._- ")
    filename = f"{timestamp}_{safe_filename}"
    filepath = upload_dir / filename

    try:
        # Ghi nội dung file vào đĩa
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer()) 
        logger.info(f" Saved uploaded file: {filepath}")
    except Exception as e:
        logger.error(f" Failed to save file {filename}: {str(e)}")
        raise IOError(f"Failed to save file {filename}: {str(e)}")

    # Trả về đường dẫn tuyệt đối
    return str(filepath.resolve())




