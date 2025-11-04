# backend/utils/file_utils.py
"""
Tiện ích xử lý file: lưu, đọc, xóa file upload
"""

import os
import json
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Any
import logging
from pathlib import Path

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


# ============================================
# JSON Operations (Required by extraction.py and graph_builder.py)
# ============================================

def save_to_json(data: Any, filename: str) -> None:
    """
    Lưu dữ liệu JSON an toàn (UTF-8, có tạo thư mục nếu cần)
    
    Args:
        data: Dữ liệu cần lưu (dict, list, etc.)
        filename: Đường dẫn file output
    """
    try:
        # Create directory if not exists
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        
        # Save to JSON
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Saved JSON to {filename}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save JSON file {filename}: {e}")
        raise


def load_from_json(filename: str) -> Any:
    """
    Đọc dữ liệu từ file JSON
    
    Args:
        filename: Đường dẫn file JSON
        
    Returns:
        Dữ liệu đã parse từ JSON
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        logger.info(f"✅ Loaded JSON from {filename}")
        return data
        
    except FileNotFoundError:
        logger.error(f"❌ File not found: {filename}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"❌ Invalid JSON in {filename}: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to load JSON file {filename}: {e}")
        raise


# ============================================
# File Upload Operations
# ============================================

def save_uploaded_file(uploaded_file, user_id: Optional[str] = None) -> str:
    """
    Lưu file upload vào thư mục backend/data/uploads/
    
    Args:
        uploaded_file: File object từ Streamlit (có .name và .getbuffer())
        user_id: ID người dùng (optional, để tổ chức theo user)
    
    Returns:
        str: Đường dẫn file đã lưu (absolute path)
    
    Raises:
        IOError: Nếu không thể lưu file
    """
    # Tạo thư mục uploads
    upload_dir = Path("backend/data/uploads")
    if user_id:
        upload_dir = upload_dir / user_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Tạo tên file unique với timestamp để tránh trùng
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Làm sạch tên file gốc (remove special chars)
    safe_filename = "".join(c for c in uploaded_file.name if c.isalnum() or c in "._- ")
    filename = f"{timestamp}_{safe_filename}"
    
    filepath = upload_dir / filename
    
    # Lưu file
    try:
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())
        logger.info(f"✅ Saved uploaded file: {filepath}")
    except Exception as e:
        logger.error(f"❌ Failed to save file {filename}: {str(e)}")
        raise IOError(f"Failed to save file {filename}: {str(e)}")
    
    return str(filepath.resolve())


def get_uploaded_files(user_id: Optional[str] = None) -> List[Path]:
    """
    Lấy danh sách file đã upload
    
    Args:
        user_id: ID người dùng (optional)
        
    Returns:
        List[Path]: Danh sách Path objects của các file
    """
    upload_dir = Path("backend/data/uploads")
    if user_id:
        upload_dir = upload_dir / user_id
    
    if not upload_dir.exists():
        return []
    
    return [f for f in upload_dir.iterdir() if f.is_file()]


def delete_uploaded_file(filepath: str) -> bool:
    """
    Xóa file đã upload
    
    Args:
        filepath: Đường dẫn file cần xóa
        
    Returns:
        bool: True nếu xóa thành công, False nếu có lỗi
    """
    try:
        Path(filepath).unlink(missing_ok=True)
        logger.info(f"✅ Deleted file: {filepath}")
        return True
    except Exception as e:
        logger.error(f"❌ Error deleting file {filepath}: {e}")
        return False


# ============================================
# File Information
# ============================================

def get_file_info(filepath: str) -> dict:
    """
    Lấy thông tin về file
    
    Args:
        filepath: Đường dẫn file
        
    Returns:
        dict: Thông tin file (size, created_time, modified_time, extension)
    """
    path = Path(filepath)
    
    if not path.exists():
        return {}
    
    stat = path.stat()
    
    return {
        'filename': path.name,
        'size_bytes': stat.st_size,
        'size_mb': round(stat.st_size / (1024 * 1024), 2),
        'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'extension': path.suffix,
        'absolute_path': str(path.resolve())
    }


def format_file_size(size_bytes: int) -> str:
    """
    Format file size thành dạng human-readable
    
    Args:
        size_bytes: Kích thước (bytes)
        
    Returns:
        String formatted (vd: '1.5 MB', '320 KB')
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


# ============================================
# Directory Operations
# ============================================

def ensure_directory(dir_path: str) -> Path:
    """
    Đảm bảo thư mục tồn tại, tạo mới nếu chưa có
    
    Args:
        dir_path: Đường dẫn thư mục
        
    Returns:
        Path: Path object của thư mục
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files_in_directory(dir_path: str, pattern: str = "*") -> List[Path]:
    """
    Liệt kê tất cả file trong thư mục theo pattern
    
    Args:
        dir_path: Đường dẫn thư mục
        pattern: Pattern để filter (e.g., "*.txt", "*.json")
        
    Returns:
        List[Path]: Danh sách file paths
    """
    path = Path(dir_path)
    
    if not path.exists():
        return []
    
    return list(path.glob(pattern))


def get_directory_size(dir_path: str) -> dict:
    """
    Tính tổng dung lượng của thư mục
    
    Args:
        dir_path: Đường dẫn thư mục
        
    Returns:
        dict: Thông tin về size và số file
    """
    path = Path(dir_path)
    
    if not path.exists():
        return {'total_size_bytes': 0, 'total_size_mb': 0, 'file_count': 0}
    
    total_size = 0
    file_count = 0
    
    for item in path.rglob('*'):
        if item.is_file():
            total_size += item.stat().st_size
            file_count += 1
    
    return {
        'total_size_bytes': total_size,
        'total_size_mb': round(total_size / (1024 * 1024), 2),
        'file_count': file_count
    }


def delete_directory(directory: str, recursive: bool = False) -> bool:
    """
    Xóa thư mục
    
    Args:
        directory: Đường dẫn thư mục
        recursive: Có xóa đệ quy không (bao gồm tất cả files và subdirs)
        
    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    try:
        import shutil
        dir_path = Path(directory)
        
        if recursive:
            shutil.rmtree(dir_path)
        else:
            dir_path.rmdir()  # Chỉ xóa nếu thư mục rỗng
        
        logger.info(f"✅ Deleted directory: {directory}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to delete directory {directory}: {e}")
        return False


# ============================================
# Text File Operations
# ============================================

def read_file_content(filepath: str, encoding: str = 'utf-8') -> str:
    """
    Đọc nội dung file text
    
    Args:
        filepath: Đường dẫn file
        encoding: Encoding của file (default: utf-8)
        
    Returns:
        str: Nội dung file
    """
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return f.read()
    except Exception as e:
        logger.error(f"❌ Failed to read file {filepath}: {e}")
        raise


def write_text_file(filepath: str, content: str, encoding: str = "utf-8") -> bool:
    """
    Ghi nội dung vào text file
    
    Args:
        filepath: Đường dẫn file
        content: Nội dung cần ghi
        encoding: Encoding (default: utf-8)
        
    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    try:
        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w", encoding=encoding) as f:
            f.write(content)
        
        logger.info(f"✅ Wrote to file: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to write file {filepath}: {e}")
        return False


# ============================================
# Additional Utilities
# ============================================

def file_exists(filepath: str) -> bool:
    """
    Kiểm tra file có tồn tại không
    
    Args:
        filepath: Đường dẫn file
        
    Returns:
        bool: True nếu tồn tại, False nếu không
    """
    return Path(filepath).exists()


def get_file_extension(filepath: str) -> str:
    """
    Lấy extension của file (lowercase, không có dấu chấm)
    
    Args:
        filepath: Đường dẫn file
        
    Returns:
        str: Extension (vd: 'pdf', 'txt', 'docx')
    """
    return Path(filepath).suffix.lower().lstrip('.')


def copy_file(src: str, dst: str, overwrite: bool = False) -> bool:
    """
    Copy file
    
    Args:
        src: Đường dẫn file nguồn
        dst: Đường dẫn file đích
        overwrite: Có ghi đè nếu file đích đã tồn tại không
        
    Returns:
        bool: True nếu thành công, False nếu thất bại
    """
    try:
        import shutil
        dst_path = Path(dst)
        
        # Check if destination exists
        if dst_path.exists() and not overwrite:
            logger.warning(f"⚠️ File already exists: {dst}")
            return False
        
        # Create destination directory if needed
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        shutil.copy2(src, dst)
        logger.info(f"✅ Copied file: {src} → {dst}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to copy file {src} to {dst}: {e}")
        return False


# ============================================
# Backward Compatibility Aliases
# ============================================

def save_json(data: Any, filename: str) -> None:
    """Alias for save_to_json (backward compatibility)"""
    return save_to_json(data, filename)


def load_json(filename: str) -> Any:
    """Alias for load_from_json (backward compatibility)"""
    return load_from_json(filename)