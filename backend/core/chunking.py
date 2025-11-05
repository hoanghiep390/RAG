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

# === Đồng bộ với hệ thống ===
from backend.utils.utils import logger
from backend.utils.file_utils import read_file_content, get_file_extension

# === Tokenizer ===
try:
    import tiktoken
except ImportError as e:
    logger.error(f"Missing required package: {e}")
    raise

# === File-specific processors (với try/except an toàn) ===
try:
    import pdfplumber
except ImportError:
    pdfplumber = None
    logger.warning("pdfplumber not installed. PDF support disabled.")

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None
    logger.warning("python-docx not installed. DOCX support disabled.")

try:
    from pptx import Presentation
except ImportError:
    Presentation = None
    logger.warning("python-pptx not installed. PPTX support disabled.")

import pandas as pd
import json
import xml.etree.ElementTree as ET
import yaml


# === Tokenizer Adapter ===
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
        return self._enc.encode(text)

    def decode(self, ids: List[int]) -> str:
        return self._enc.decode(ids)


# === Soft split by sentence ===
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
    suffix = f" — tiếp {part_idx}/{part_total}" if part_total > 1 else ""
    return f"**[SECTION] {section}{suffix}**\n\n{content}"


# === Config ===
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
        if not self.format_specific_configs:
            return self
        ext = file_ext.lower().lstrip('.')
        if ext in self.format_specific_configs:
            overrides = self.format_specific_configs[ext]
            return DocChunkConfig(**{**self.__dict__, **overrides})
        return self


# === File Type Detection ===
def _detect_file_type(path: str) -> str:
    ext = Path(path).suffix.lower().lstrip('.')
    type_map = {
        'pdf': 'PDF', 'docx': 'DOCX', 'doc': 'DOCX',
        'pptx': 'PPTX', 'ppt': 'PPTX',
        'txt': 'TEXT', 'md': 'MARKDOWN', 'markdown': 'MARKDOWN', 'tex': 'LATEX',
        'csv': 'CSV', 'json': 'JSON', 'xml': 'XML', 'yaml': 'YAML', 'yml': 'YAML',
        'html': 'HTML', 'htm': 'HTML', 'css': 'CSS', 'scss': 'CSS', 'less': 'CSS',
        'py': 'CODE', 'java': 'CODE', 'js': 'CODE', 'ts': 'CODE',
        'c': 'CODE', 'cpp': 'CODE', 'go': 'CODE', 'rb': 'CODE', 'php': 'CODE', 'swift': 'CODE',
        'conf': 'CONFIG', 'ini': 'CONFIG', 'properties': 'CONFIG', 'bat': 'CONFIG',
    }
    return type_map.get(ext, 'TEXT')


# === File Processors ===
def _process_pdf(path: str, cfg: DocChunkConfig) -> List[Tuple[str, str]]:
    segments = []
    if not pdfplumber:
        logger.error("pdfplumber not available")
        return segments
    try:
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                if cfg.include_paragraph and text.strip():
                    segments.append((f"Page {page_num}", text))
                if cfg.include_tables:
                    tables = page.extract_tables()
                    for i, table in enumerate(tables):
                        if table:
                            df = pd.DataFrame(table[1:], columns=table[0]) if len(table) > 1 else pd.DataFrame(table)
                            table_text = df.to_string(index=False)
                            if cfg.enable_split_table and len(df) > cfg.table_rows_per_part:
                                for j in range(0, len(df), cfg.table_rows_per_part):
                                    part = df.iloc[j:j + cfg.table_rows_per_part]
                                    segments.append((f"Table Page {page_num} Part {j//cfg.table_rows_per_part + 1}", part.to_string(index=False)))
                            else:
                                segments.append((f"Table Page {page_num} #{i+1}", table_text))
    except Exception as e:
        logger.error(f"PDF processing error: {e}")
    return segments


def _process_docx(path: str, cfg: DocChunkConfig) -> List[Tuple[str, str]]:
    segments = []
    if not DocxDocument:
        logger.error("python-docx not available")
        return segments
    try:
        doc = DocxDocument(path)
        current_section = "Document"
        para_idx = 0
        for para in doc.paragraphs:
            if para.style.name.startswith('Heading'):
                current_section = para.text.strip() or f"Heading {para_idx}"
                if cfg.include_heading:
                    segments.append((current_section, para.text))
            elif para.text.strip():
                para_idx += 1
                if cfg.include_paragraph:
                    segments.append((f"Paragraph {para_idx}", para.text))
        if doc.tables and cfg.include_tables:
            for i, table in enumerate(doc.tables):
                df = pd.DataFrame([[cell.text for cell in row.cells] for row in table.rows])
                table_text = df.to_string(index=False)
                if cfg.enable_split_table and len(df) > cfg.table_rows_per_part:
                    for j in range(0, len(df), cfg.table_rows_per_part):
                        part = df.iloc[j:j + cfg.table_rows_per_part]
                        segments.append((f"Table #{i+1} Part {j//cfg.table_rows_per_part + 1}", part.to_string(index=False)))
                else:
                    segments.append((f"Table #{i+1}", table_text))
    except Exception as e:
        logger.error(f"DOCX processing error: {e}")
    return segments


def _process_pptx(path: str, cfg: DocChunkConfig) -> List[Tuple[str, str]]:
    segments = []
    if not Presentation:
        logger.error("python-pptx not available")
        return segments
    try:
        prs = Presentation(path)
        for slide_num, slide in enumerate(prs.slides, 1):
            text = []
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text.append(shape.text)
            content = "\n".join(text)
            if content.strip():
                segments.append((f"Slide {slide_num}", content))
    except Exception as e:
        logger.error(f"PPTX processing error: {e}")
    return segments


def _process_text(path: str, cfg: DocChunkConfig) -> List[Tuple[str, str]]:
    segments = []
    try:
        content = read_file_content(path)
        lines = content.splitlines()
        current_section = "Document"
        para_idx = 0
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if cfg.include_heading and stripped.startswith(('# ', '## ', '### ')):
                current_section = stripped
                segments.append((current_section, line))
            else:
                para_idx += 1
                if cfg.include_paragraph:
                    segments.append((f"Line {para_idx}", line))
    except Exception as e:
        logger.error(f"Text file error: {e}")
    return segments


def _process_data_file(path: str, cfg: DocChunkConfig, file_type: str) -> List[Tuple[str, str]]:
    segments = []
    try:
        content = read_file_content(path)
        if file_type == 'CSV':
            df = pd.read_csv(path)
            if cfg.enable_split_table and len(df) > cfg.table_rows_per_part:
                for i in range(0, len(df), cfg.table_rows_per_part):
                    part = df.iloc[i:i + cfg.table_rows_per_part]
                    segments.append((f"CSV Part {i//cfg.table_rows_per_part + 1}", part.to_csv(index=False)))
            else:
                segments.append(("CSV Data", df.to_csv(index=False)))
        elif file_type == 'JSON':
            data = json.loads(content)
            segments.append(("JSON", json.dumps(data, indent=2, ensure_ascii=False)))
        elif file_type == 'XML':
            tree = ET.parse(path)
            segments.append(("XML", ET.tostring(tree.getroot(), encoding='unicode')))
        elif file_type in ('YAML', 'YML'):
            data = yaml.safe_load(content)
            segments.append(("YAML", yaml.dump(data, allow_unicode=True)))
    except Exception as e:
        logger.error(f"Data file error: {e}")
    return segments


def _doc_to_text_segments(path: str, cfg: DocChunkConfig) -> List[Tuple[str, str]]:
    file_type = _detect_file_type(path)
    ext = get_file_extension(path)
    format_cfg = cfg.get_format_config(ext)

    if file_type == 'PDF':
        return _process_pdf(path, format_cfg)
    elif file_type == 'DOCX':
        return _process_docx(path, format_cfg)
    elif file_type == 'PPTX':
        return _process_pptx(path, format_cfg)
    elif file_type in ('CSV', 'JSON', 'XML', 'YAML', 'YML'):
        return _process_data_file(path, format_cfg, file_type)
    else:
        return _process_text(path, format_cfg)


# === Packing Logic (SỬA LỖI `step`, `start`) ===
def _pack_segments_by_token(
    tokenizer: Any,
    segments: List[Tuple[str, str]],
    cfg: DocChunkConfig,
    file_path: str
) -> List[Dict[str, Any]]:
    T = _TokenizerAdapter(tokenizer)
    results = []
    next_index = 0
    buf: List[int] = []
    buf_section = ""
    max_len = cfg.max_token_size
    overlap = cfg.overlap_token_size
    sep_tokens = T.encode(cfg.join_separator)

    def flush_buffer(section: str):
        nonlocal next_index, buf
        if not buf:
            return
        payload = cfg.join_separator.join(T.decode(ids) for ids in [buf])
        parts = _split_by_tokens_soft(payload, T, max_len, overlap)
        for i, p in enumerate(parts, 1):
            payload_with_breadcrumb = _with_breadcrumb(section, p, i, len(parts))
            chunk = {
                "chunk_id": str(uuid.uuid4()),
                "content": payload_with_breadcrumb,
                "tokens": len(T.encode(payload_with_breadcrumb)),
                "order": next_index,
                "hierarchy": section,
                "file_path": file_path,
                "file_type": _detect_file_type(file_path),
            }
            results.append(chunk)
            next_index += 1
        buf = []

    for hierarchy, seg in segments:
        seg_tokens = T.encode(seg)
        if not buf:
            buf_section = hierarchy

        if buf and len(buf) + len(sep_tokens) + len(seg_tokens) > max_len:
            flush_buffer(buf_section)
            buf.extend(seg_tokens)
        elif buf:
            buf.extend(sep_tokens)
            buf.extend(seg_tokens)
        else:
            buf.extend(seg_tokens)

    if buf:
        flush_buffer(buf_section)

    return results


# === Main Function ===
def process_document_to_chunks(
    path: str,
    config: Optional[DocChunkConfig] = None
) -> List[Dict[str, Any]]:
    cfg = config or DocChunkConfig()

    enc = tiktoken.encoding_for_model("gpt-4o-mini")
    tok_4o = _TokenizerAdapter(enc)

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
        logger.error(f"Failed to process {path}: {str(e)}", exc_info=True)
        raise


# === Default Configs ===
DEFAULT_FORMAT_CONFIGS = {
    'txt': {'max_token_size': 500, 'overlap_token_size': 50},
    'md': {'max_token_size': 400, 'overlap_token_size': 40},
    'markdown': {'max_token_size': 400, 'overlap_token_size': 40},
    'tex': {'max_token_size': 400, 'overlap_token_size': 40},
    'pdf': {'max_token_size': 300, 'overlap_token_size': 50},
    'docx': {'max_token_size': 300, 'overlap_token_size': 50},
    'pptx': {'max_token_size': 250, 'overlap_token_size': 30},
    'csv': {'max_token_size': 200, 'overlap_token_size': 20},
    'json': {'max_token_size': 250, 'overlap_token_size': 25},
    'xml': {'max_token_size': 250, 'overlap_token_size': 25},
    'yaml': {'max_token_size': 250, 'overlap_token_size': 25},
    'yml': {'max_token_size': 250, 'overlap_token_size': 25},
    'html': {'max_token_size': 350, 'overlap_token_size': 40},
    'py': {'max_token_size': 400, 'overlap_token_size': 40},
    'conf': {'max_token_size': 200, 'overlap_token_size': 20},
}


def get_default_config_for_file(filepath: str) -> DocChunkConfig:
    ext = Path(filepath).suffix.lower().lstrip('.')
    config_dict = DEFAULT_FORMAT_CONFIGS.get(ext, {
        'max_token_size': 300,
        'overlap_token_size': 50
    })
    return DocChunkConfig(**config_dict)