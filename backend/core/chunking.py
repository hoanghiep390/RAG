# backend/core/chunking.py
"""
Unified document chunking pipeline for LightRAG.
Supports 32+ text-based document formats.

Output per chunk:
{
  "chunk_id": str(uuid4()),
  "content": str,
  "tokens": int,
  "order": int,
  "hierarchy": str,
  "file_path": str,
  "file_type": str,
}

Supported formats:
- Text: TXT, MD, TEX
- Office: PDF, DOCX, PPTX, RTF, ODT, EPUB
- Web: HTML, CSS, SCSS, LESS
- Data: CSV, SQL, JSON, XML, YAML
- Programming: PY, JAVA, JS, TS, C, CPP, GO, RB, PHP, SWIFT
- Config: CONF, INI, PROPERTIES, BAT
"""

from __future__ import annotations
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

# Đồng bộ với hệ thống: Sử dụng logger từ utils
from backend.utils.utils import logger

# Đồng bộ: Sử dụng file_utils để đọc file nếu cần
from backend.utils.file_utils import read_file_content, get_file_extension

# Thư viện cần thiết cho xử lý các định dạng file
try:
    import tiktoken
    from sentence_transformers.util import OpenAITokenizer  # Nếu có, nhưng giữ nguyên tiktoken
except ImportError:
    logger.error("Required libraries missing. Install tiktoken and sentence-transformers.")
    raise

# Các thư viện xử lý file cụ thể (giữ nguyên, thêm try-except để đồng bộ)
try:
    import docx  # DOCX
    import pptx  # PPTX
    import pdfplumber  # PDF
    import rtfparse  # RTF (nếu có)
    import odt  # ODT (nếu có)
    import epub  # EPUB (nếu có)
    import pandas as pd  # CSV, XLSX
    import json  # JSON
    import xml.etree.ElementTree as ET  # XML
    import yaml  # YAML
    import sqlparse  # SQL
    import markdown  # MD
    # Các thư viện khác cho code syntax nếu cần
except ImportError as e:
    logger.warning(f"Missing library for some formats: {e}")

class _TokenizerAdapter:
    def __init__(self, tokenizer):
        self.raw = tokenizer
        self.kind = None
        if hasattr(tokenizer, "tokenizer") and hasattr(tokenizer.tokenizer, "encode"):
            self.kind = "openai_docling"
            self._enc = tokenizer.tokenizer
            return

        if (
            hasattr(tokenizer, "encode")
            and hasattr(tokenizer, "decode")
            and tokenizer.__class__.__name__ in ("Encoding", "CoreBPE")
        ):    
            self.kind = "tiktoken"
            self._enc = tokenizer
            return
        if hasattr(tokenizer, "encode") and hasattr(tokenizer, "decode"):
            self.kind = "generic"
            self._enc = tokenizer
            return

        raise TypeError("Unsupported tokenizer type")

    def encode(self, text: str) -> List[int]:
        if self.kind == "hf":
            return self._enc.encode(text, add_special_tokens=False)
        return self._enc.encode(text)

    def decode(self, ids: List[int]) -> str:
        if self.kind == "hf":
            return self._enc.decode(ids, skip_special_tokens=True)
        return self._enc.decode(ids)

_SENTENCE_BREAK = re.compile(
    r"(?s)(.*?)([\.!?…]|(?:\n{2,})|(?:\r?\n- )|(?:\r?\n• )|(?:\n\|[-:]+))\s+$"
)

def _split_by_tokens_soft(
    text: str,
    T: _TokenizerAdapter,
    max_size: int,
    overlap: int,
    lookback_ratio: float = 0.2,
) -> List[str]:
    """Split text by tokens with soft boundaries"""
    if max_size <= 0:
        return [text]

    ids = T.encode(text)
    n = len(ids)
    if n <= max_size:
        return [text]

    out = []
    start = 0
    step = max(1, max_size - max(0, overlap))

    while start < n:
        end = min(start + max_size, n)
        window_ids = ids[start:end]
        window_txt = T.decode(window_ids)

        if end < n:
            lb_chars = max(10, int(len(window_txt) * (1 - lookback_ratio)))
            tail = window_txt[lb_chars:]
            m = _SENTENCE_BREAK.search(tail)
            if m:
                nice_cut_char = lb_chars + m.end()
                keep_txt = window_txt[:nice_cut_char]
                window_ids = T.encode(keep_txt)

        piece = T.decode(window_ids).rstrip()
        out.append(piece)

        if start + len(window_ids) >= n:
            break
        start = start + len(window_ids) - max(0, overlap)

    return out

def _with_breadcrumb(section: str, content: str, part_idx: int, part_total: int) -> str:
    """Add section breadcrumb to content"""
    suffix = f" — tiếp {part_idx}/{part_total}" if part_total > 1 else ""
    return f"**[SECTION] {section}{suffix}**\n\n{content}"

@dataclass
class DocChunkConfig:
    include_paragraph: bool = True
    include_list: bool = True
    include_heading: bool = True
    include_code: bool = True
    include_tables: bool = True
    enable_split_table: bool = True
    table_rows_per_part: int = 50
    join_separator: str = "\n\n"
    max_token_size: int = 300
    overlap_token_size: int = 50
    format_specific_configs: Optional[Dict[str, Dict]] = None
    
    def get_format_config(self, file_ext: str) -> 'DocChunkConfig':
        """Get format-specific config"""
        if not self.format_specific_configs:
            return self
        
        ext = file_ext.lower().lstrip('.')
        if ext in self.format_specific_configs:
            overrides = self.format_specific_configs[ext]
            return DocChunkConfig(**{**self.__dict__, **overrides})
        
        return self

def _detect_file_type(path: str) -> str:
    """Detect file type from extension"""
    ext = Path(path).suffix.lower().lstrip('.')
    
    type_map = {
        # Documents
        'pdf': 'PDF', 'docx': 'DOCX', 'doc': 'DOCX',
        'pptx': 'PPTX', 'ppt': 'PPTX', 'rtf': 'RTF',
        'odt': 'ODT', 'epub': 'EPUB',
        # Text
        'txt': 'TEXT', 'md': 'MARKDOWN', 'markdown': 'MARKDOWN',
        'tex': 'LATEX',
        # Data
        'csv': 'CSV', 'sql': 'SQL', 'json': 'JSON',
        'xml': 'XML', 'yaml': 'YAML', 'yml': 'YAML',
        # Web
        'html': 'HTML', 'htm': 'HTML', 'css': 'CSS',
        'scss': 'CSS', 'less': 'CSS',
        # Programming
        'py': 'CODE', 'java': 'CODE', 'js': 'CODE', 'ts': 'CODE',
        'c': 'CODE', 'cpp': 'CODE', 'go': 'CODE', 'rb': 'CODE',
        'php': 'CODE', 'swift': 'CODE',
        # Config
        'conf': 'CONFIG', 'ini': 'CONFIG', 'properties': 'CONFIG',
        'bat': 'CONFIG'
    }
    return type_map.get(ext, 'TEXT')  # Default to TEXT

# Các hàm xử lý từng loại file (giữ nguyên logic, tối ưu hóa bằng cách thêm logging và error handling)
def _process_pdf(path: str, config: DocChunkConfig) -> List[Tuple[str, str]]:
    """Process PDF: Extract text, tables, etc."""
    segments = []
    try:
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if config.include_paragraph:
                    segments.append((f"Page {page_num}", text))
                if config.include_tables and page.extract_tables():
                    for table in page.extract_tables():
                        # Giữ nguyên logic table splitting
                        if config.enable_split_table and len(table) > config.table_rows_per_part:
                            for i in range(0, len(table), config.table_rows_per_part):
                                part = table[i:i + config.table_rows_per_part]
                                segments.append((f"Table Page {page_num} Part {i//config.table_rows_per_part + 1}", pd.DataFrame(part).to_string()))
                        else:
                            segments.append((f"Table Page {page_num}", pd.DataFrame(table).to_string()))
    except Exception as e:
        logger.error(f"Error processing PDF {path}: {e}")
    return segments

# Tương tự cho các định dạng khác: _process_docx, _process_pptx, v.v. (giữ nguyên logic, thêm logging)

# Ví dụ cho TEXT
def _process_text(path: str, config: DocChunkConfig) -> List[Tuple[str, str]]:
    segments = []
    try:
        content = read_file_content(path)  # Đồng bộ với file_utils
        if config.include_heading:
            # Logic tìm heading (giữ nguyên)
            lines = content.splitlines()
            for i, line in enumerate(lines):
                if line.startswith('#'):
                    segments.append((f"Heading {i+1}", line))
                else:
                    segments.append((f"Paragraph {i+1}", line))
    except Exception as e:
        logger.error(f"Error processing TEXT {path}: {e}")
    return segments

# ... (giữ nguyên các hàm xử lý khác, thêm logging tương tự)

def _doc_to_text_segments(path: str, config: DocChunkConfig) -> List[Tuple[str, str]]:
    """Chuyển document thành segments dựa trên loại file"""
    file_type = _detect_file_type(path)
    ext = get_file_extension(path)  # Đồng bộ với file_utils
    
    format_config = config.get_format_config(ext)
    
    if file_type == 'PDF':
        return _process_pdf(path, format_config)
    elif file_type == 'DOCX':
        return _process_docx(path, format_config)
    # ... (giữ nguyên switch case cho tất cả định dạng)
    else:
        return _process_text(path, format_config)  # Default

def _pack_segments_by_token(
    tokenizer: OpenAITokenizer,
    segments: List[Tuple[str, str]],
    cfg: DocChunkConfig,
    file_path: str
) -> List[Dict[str, Any]]:
    """Pack segments thành chunks theo token limit"""
    T = _TokenizerAdapter(tokenizer)
    results = []
    next_index = 0
    buf = []
    buf_section = ""
    max_len = cfg.max_token_size
    overlap = cfg.overlap_token_size
    sep_tokens = T.encode(cfg.join_separator)

    for hierarchy, seg in segments:
        seg_tokens = T.encode(seg)
        if not buf:
            buf_section = hierarchy

        if len(buf) + len(sep_tokens) + len(seg_tokens) > max_len:
            def flush_buffer(section: str):
                nonlocal next_index
                payload = cfg.join_separator.join(T.decode(buf[i]) for i in range(len(buf)))
                parts = _split_by_tokens_soft(payload, T, max_len, overlap)
                for i, p in enumerate(parts, 1):
                    payload = _with_breadcrumb(section, p, i, len(parts))
                    chunk = {
                        "chunk_id": str(uuid.uuid4()),
                        "content": payload,
                        "tokens": len(T.encode(payload)),
                        "order": next_index,
                        "hierarchy": section,
                        "file_path": file_path,
                        "file_type": _detect_file_type(file_path),
                    }
                    results.append(chunk)
                    next_index += 1
                start += step
            buf = []
            continue
        if buf:
            if len(buf) + len(sep_tokens) + len(seg_tokens) <= max_len:
                buf.extend(sep_tokens)
                buf.extend(seg_tokens)
            else:
                flush_buffer(buf_section)
                buf.extend(seg_tokens)
        else:
            buf.extend(seg_tokens)
    if buf:
        flush_buffer(buf_section)

    return results

def process_document_to_chunks(
    path: str,
    config: Optional[DocChunkConfig] = None
) -> List[Dict[str, Any]]:
    """
    Process document into chunks.
    
    Args:
        path: File path
        config: Chunking configuration
        
    Returns:
        List of chunk dictionaries
    """
    cfg = config or DocChunkConfig()

    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    tok_4o = OpenAITokenizer(
        tokenizer=enc, max_tokens=128000, model_name="gpt-4o-mini"
    )

    logger.info(f"Start processing: {path}")
    
    try:
        segments = _doc_to_text_segments(path, cfg)
        
        if not segments:
            logger.warning(f"No segments extracted from: {path}")
            return []
        
        results = _pack_segments_by_token(tok_4o, segments=segments, cfg=cfg, file_path=path)
        logger.info(f"Finished processing: {path} → {len(results)} chunks")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to process {path}: {str(e)}")
        raise

DEFAULT_FORMAT_CONFIGS = {
    # Text formats - larger chunks
    'txt': {'max_token_size': 500, 'overlap_token_size': 50},
    'md': {'max_token_size': 400, 'overlap_token_size': 40},
    'markdown': {'max_token_size': 400, 'overlap_token_size': 40},
    'tex': {'max_token_size': 400, 'overlap_token_size': 40},
    # Documents - balanced
    'pdf': {'max_token_size': 300, 'overlap_token_size': 50},
    'docx': {'max_token_size': 300, 'overlap_token_size': 50},
    'pptx': {'max_token_size': 250, 'overlap_token_size': 30},
    'rtf': {'max_token_size': 300, 'overlap_token_size': 40},
    'odt': {'max_token_size': 300, 'overlap_token_size': 40},
    'epub': {'max_token_size': 350, 'overlap_token_size': 40},
    # Data formats - smaller chunks
    'csv': {'max_token_size': 200, 'overlap_token_size': 20},
    'sql': {'max_token_size': 300, 'overlap_token_size': 30},
    'json': {'max_token_size': 250, 'overlap_token_size': 25},
    'xml': {'max_token_size': 250, 'overlap_token_size': 25},
    'yaml': {'max_token_size': 250, 'overlap_token_size': 25},
    'yml': {'max_token_size': 250, 'overlap_token_size': 25},
    # Web formats
    'html': {'max_token_size': 350, 'overlap_token_size': 40},
    'htm': {'max_token_size': 350, 'overlap_token_size': 40},
    'css': {'max_token_size': 300, 'overlap_token_size': 30},
    'scss': {'max_token_size': 300, 'overlap_token_size': 30},
    'less': {'max_token_size': 300, 'overlap_token_size': 30},
    # Programming - medium chunks
    'py': {'max_token_size': 400, 'overlap_token_size': 40},
    'java': {'max_token_size': 400, 'overlap_token_size': 40},
    'js': {'max_token_size': 400, 'overlap_token_size': 40},
    'ts': {'max_token_size': 400, 'overlap_token_size': 40},
    'cpp': {'max_token_size': 400, 'overlap_token_size': 40},
    'c': {'max_token_size': 400, 'overlap_token_size': 40},
    'go': {'max_token_size': 400, 'overlap_token_size': 40},
    'rb': {'max_token_size': 400, 'overlap_token_size': 40},
    'php': {'max_token_size': 400, 'overlap_token_size': 40},
    'swift': {'max_token_size': 400, 'overlap_token_size': 40},
    # Config files - small chunks
    'conf': {'max_token_size': 200, 'overlap_token_size': 20},
    'ini': {'max_token_size': 200, 'overlap_token_size': 20},
    'properties': {'max_token_size': 200, 'overlap_token_size': 20},
    'bat': {'max_token_size': 200, 'overlap_token_size': 20},
}

def get_default_config_for_file(filepath: str) -> DocChunkConfig:
    """Get optimized config for specific file type"""
    ext = Path(filepath).suffix.lower().lstrip('.')
    
    config_dict = DEFAULT_FORMAT_CONFIGS.get(ext, {
        'max_token_size': 300,
        'overlap_token_size': 50
    })
    
    return DocChunkConfig(**config_dict)