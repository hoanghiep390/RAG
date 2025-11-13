# backend/utils/file_utils.py
"""
✅ CLEANED: File utilities - CHỈ giữ các functions cần thiết
Loại bỏ: save_to_json, load_from_json (sẽ dùng MongoDB)
Giữ lại: File upload, file info, directory operations
"""

import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def ensure_dir(path):
    """Ensure directory exists"""
    Path(path).mkdir(parents=True, exist_ok=True)


# ============================================
# FILE UPLOAD (KEEP - Required for saving uploaded files)
# ============================================

def save_uploaded_file(uploaded_file, user_id: Optional[str] = None) -> str:
    """
    ✅ KEEP: Save uploaded file to uploads directory
    
    Args:
        uploaded_file: File object from Streamlit
        user_id: User ID for organization
    
    Returns:
        str: Absolute path to saved file
    """
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
        logger.info(f"✅ Saved uploaded file: {filepath}")
    except Exception as e:
        logger.error(f"❌ Failed to save file {filename}: {str(e)}")
        raise IOError(f"Failed to save file {filename}: {str(e)}")

    return str(filepath.resolve())


def get_uploaded_files(user_id: Optional[str] = None) -> List[Path]:
    """
    ✅ KEEP: Get list of uploaded files
    
    Args:
        user_id: User ID
        
    Returns:
        List[Path]: List of uploaded file paths
    """
    upload_dir = Path("backend/data/uploads")
    if user_id:
        upload_dir = upload_dir / user_id
    
    if not upload_dir.exists():
        return []
    
    return [f for f in upload_dir.iterdir() if f.is_file()]


def delete_uploaded_file(filepath: str) -> bool:
    """
    ✅ KEEP: Delete uploaded file
    
    Args:
        filepath: File path to delete
        
    Returns:
        bool: True if successful
    """
    try:
        Path(filepath).unlink(missing_ok=True)
        logger.info(f"✅ Deleted file: {filepath}")
        return True
    except Exception as e:
        logger.error(f"❌ Error deleting file {filepath}: {e}")
        return False


# ============================================
# FILE INFORMATION (KEEP - Useful utilities)
# ============================================

def get_file_info(filepath: str) -> dict:
    """
    ✅ KEEP: Get file information
    
    Args:
        filepath: File path
        
    Returns:
        dict: File information
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
    ✅ KEEP: Format file size to human-readable
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        str: Formatted size (e.g., '1.5 MB')
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


# ============================================
# DIRECTORY OPERATIONS (KEEP - Useful utilities)
# ============================================

def ensure_directory(dir_path: str) -> Path:
    """
    ✅ KEEP: Ensure directory exists
    
    Args:
        dir_path: Directory path
        
    Returns:
        Path: Path object
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files_in_directory(dir_path: str, pattern: str = "*") -> List[Path]:
    """
    ✅ KEEP: List files in directory
    
    Args:
        dir_path: Directory path
        pattern: Glob pattern (e.g., "*.txt")
        
    Returns:
        List[Path]: List of file paths
    """
    path = Path(dir_path)
    
    if not path.exists():
        return []
    
    return list(path.glob(pattern))


def get_directory_size(dir_path: str) -> dict:
    """
    ✅ KEEP: Calculate directory size
    
    Args:
        dir_path: Directory path
        
    Returns:
        dict: Size info
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
    ✅ KEEP: Delete directory
    
    Args:
        directory: Directory path
        recursive: Delete recursively
        
    Returns:
        bool: True if successful
    """
    try:
        import shutil
        dir_path = Path(directory)
        
        if recursive:
            shutil.rmtree(dir_path)
        else:
            dir_path.rmdir()  # Only if empty
        
        logger.info(f"✅ Deleted directory: {directory}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to delete directory {directory}: {e}")
        return False


# ============================================
# TEXT FILE OPERATIONS (KEEP - Used by chunking)
# ============================================

def read_file_content(filepath: str, encoding: str = 'utf-8') -> str:
    """
    ✅ KEEP: Read text file content (used by chunking.py)
    
    Args:
        filepath: File path
        encoding: File encoding
        
    Returns:
        str: File content
    """
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return f.read()
    except Exception as e:
        logger.error(f"❌ Failed to read file {filepath}: {e}")
        raise


def write_text_file(filepath: str, content: str, encoding: str = "utf-8") -> bool:
    """
    ✅ KEEP: Write text file
    
    Args:
        filepath: File path
        content: Content to write
        encoding: File encoding
        
    Returns:
        bool: True if successful
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w", encoding=encoding) as f:
            f.write(content)
        
        logger.info(f"✅ Wrote to file: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to write file {filepath}: {e}")
        return False


# ============================================
# FILE CHECKS (KEEP - Useful utilities)
# ============================================

def file_exists(filepath: str) -> bool:
    """
    ✅ KEEP: Check if file exists
    
    Args:
        filepath: File path
        
    Returns:
        bool: True if exists
    """
    return Path(filepath).exists()


def get_file_extension(filepath: str) -> str:
    """
    ✅ KEEP: Get file extension (used by chunking.py)
    
    Args:
        filepath: File path
        
    Returns:
        str: Extension without dot (e.g., 'pdf')
    """
    return Path(filepath).suffix.lower().lstrip('.')


def copy_file(src: str, dst: str, overwrite: bool = False) -> bool:
    """
    ✅ KEEP: Copy file
    
    Args:
        src: Source file path
        dst: Destination file path
        overwrite: Overwrite if exists
        
    Returns:
        bool: True if successful
    """
    try:
        import shutil
        dst_path = Path(dst)
        
        if dst_path.exists() and not overwrite:
            logger.warning(f"⚠️ File already exists: {dst}")
            return False
        
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        logger.info(f"✅ Copied file: {src} → {dst}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to copy file {src} to {dst}: {e}")
        return False


"""
MIGRATION NOTE:
--------------
If you need to save/load JSON data, use MongoDB:

OLD CODE:
    from backend.utils.file_utils import save_to_json, load_from_json
    save_to_json(data, "file.json")
    data = load_from_json("file.json")

NEW CODE:
    from backend.db.mongo_storage import MongoStorage
    storage = MongoStorage(user_id)
    storage.save_chunks(doc_id, chunks)
    chunks = storage.get_chunks(doc_id)
"""