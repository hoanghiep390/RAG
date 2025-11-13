# backend/utils/file_utils.py
import os
from pathlib import Path
from datetime import datetime
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


def ensure_dir(path):
    """Ensure directory exists"""
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
        logger.info(f"✅ Saved uploaded file: {filepath}")
    except Exception as e:
        logger.error(f"❌ Failed to save file {filename}: {str(e)}")
        raise IOError(f"Failed to save file {filename}: {str(e)}")

    return str(filepath.resolve())


def get_uploaded_files(user_id: Optional[str] = None) -> List[Path]:
    upload_dir = Path("backend/data/uploads")
    if user_id:
        upload_dir = upload_dir / user_id
    
    if not upload_dir.exists():
        return []
    
    return [f for f in upload_dir.iterdir() if f.is_file()]


def delete_uploaded_file(filepath: str) -> bool:
    try:
        Path(filepath).unlink(missing_ok=True)
        logger.info(f"✅ Deleted file: {filepath}")
        return True
    except Exception as e:
        logger.error(f"❌ Error deleting file {filepath}: {e}")
        return False


def get_file_info(filepath: str) -> dict:
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
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"



def ensure_directory(dir_path: str) -> Path:
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files_in_directory(dir_path: str, pattern: str = "*") -> List[Path]:
    path = Path(dir_path)
    
    if not path.exists():
        return []
    
    return list(path.glob(pattern))


def get_directory_size(dir_path: str) -> dict:
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
    try:
        import shutil
        dir_path = Path(directory)
        
        if recursive:
            shutil.rmtree(dir_path)
        else:
            dir_path.rmdir()  
        
        logger.info(f"✅ Deleted directory: {directory}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to delete directory {directory}: {e}")
        return False


def read_file_content(filepath: str, encoding: str = 'utf-8') -> str:
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            return f.read()
    except Exception as e:
        logger.error(f"❌ Failed to read file {filepath}: {e}")
        raise


def write_text_file(filepath: str, content: str, encoding: str = "utf-8") -> bool:
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w", encoding=encoding) as f:
            f.write(content)
        
        logger.info(f"✅ Wrote to file: {filepath}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to write file {filepath}: {e}")
        return False


def file_exists(filepath: str) -> bool:
    return Path(filepath).exists()


def get_file_extension(filepath: str) -> str:
    return Path(filepath).suffix.lower().lstrip('.')


def copy_file(src: str, dst: str, overwrite: bool = False) -> bool:
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

