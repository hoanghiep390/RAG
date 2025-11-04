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

from backend.utils.utils import logger


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
        'html': 'HTML', 'htm': 'HTML',
        'css': 'CSS', 'scss': 'SCSS', 'less': 'LESS',
        # Programming
        'py': 'PYTHON', 'java': 'JAVA', 'js': 'JAVASCRIPT',
        'ts': 'TYPESCRIPT', 'cpp': 'CPP', 'c': 'C',
        'go': 'GO', 'rb': 'RUBY', 'php': 'PHP',
        'swift': 'SWIFT', 'bat': 'BATCH',
        # Config
        'conf': 'CONFIG', 'ini': 'CONFIG', 'properties': 'CONFIG',
    }
    
    return type_map.get(ext, 'UNKNOWN')

def _norm(s: str) -> str:
    """Normalize string"""
    return (s or "").strip()


def _get_section_from_chunk(c) -> str:
    """Extract section name from chunk metadata"""
    meta = getattr(c, "meta", None)
    if meta and hasattr(meta, "headings") and meta.headings:
        parts = [h.strip() for h in meta.headings if isinstance(h, str) and h.strip()]
        if parts:
            return " > ".join(parts)
    return "Document"


def _simple_text_extraction(path: str) -> List[Tuple[str, str]]:
    """Simple text extraction for text-based files"""
    logger.info(f"Using simple text extraction for: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        if content.strip():
            ext = Path(path).suffix.lower().lstrip('.')
            
            if ext in ['md', 'markdown']:
                sections = re.split(r'\n#{1,6}\s+', content)
                if len(sections) > 1:
                    return [("Document", sections[0])] + [
                        (f"Section {i}", sec) for i, sec in enumerate(sections[1:], 1)
                    ]
            
            elif ext in ['py', 'java', 'js', 'ts', 'cpp', 'c', 'go', 'rb', 'php', 'swift']:
                if 'def ' in content or 'function ' in content or 'class ' in content:
                    return [("Code", content)]
            
            elif ext in ['json', 'yaml', 'yml', 'xml']:
                return [("Data", content)]
            
            elif ext == 'csv':
                lines = content.split('\n')
                if len(lines) > 100:  
                    header = lines[0] if lines else ""
                    return [(f"Rows 1-100", '\n'.join([header] + lines[1:101]))] + [
                        (f"Rows {i+1}-{min(i+100, len(lines))}", 
                         '\n'.join([header] + lines[i+1:min(i+100, len(lines))]))
                        for i in range(100, len(lines), 100)
                    ]
                return [("Data", content)]
        
            return [("Document", content)]
        else:
            logger.warning(f"Empty content from: {path}")
            return []
            
    except Exception as e:
        logger.error(f"Text extraction failed for {path}: {str(e)}")
        return []


def _doc_to_text_segments(path: str, cfg: DocChunkConfig) -> List[Tuple[str, str]]:
    """Extract text segments from document"""
    
    file_type = _detect_file_type(path)
    file_ext = Path(path).suffix.lower().lstrip('.')
    cfg = cfg.get_format_config(Path(path).suffix)
    
    logger.info(f"Processing {file_type} document: {path}")

    text_formats = [
        'txt', 'md', 'tex', 'csv', 'sql', 'json', 'xml', 'yaml', 'yml',
        'html', 'htm', 'css', 'scss', 'less', 'py', 'java', 'js', 'ts',
        'c', 'cpp', 'go', 'rb', 'php', 'swift', 'bat', 'conf', 'ini', 
        'properties', 'markdown'
    ]
    
    if file_ext in text_formats:
        return _simple_text_extraction(path)

    import pipmaster as pm
    
    if not pm.is_installed("docling"):
        logger.info("Installing docling...")
        pm.install("docling")
        pm.install("docling_core")

    try:
        from docling.document_converter import DocumentConverter
        from docling_core.transforms.chunker import HierarchicalChunker

        converter = DocumentConverter()
        
        try:
            result = converter.convert(path)
            logger.info(f"Docling conversion successful: {path}")
        except Exception as e:
            logger.error(f"Docling conversion failed for {path}: {str(e)}")
            return _simple_text_extraction(path)
        
        chunker = HierarchicalChunker()
        chunk_iter = chunker.chunk(dl_doc=result.document)
        
    except ImportError as e:
        logger.error(f"Docling import failed: {str(e)}")
        return _simple_text_extraction(path)

    segments: List[Tuple[str, str]] = []
    processed_chunks = 0

    for chunk in chunk_iter:
        processed_chunks += 1
        doc_items = getattr(chunk.meta, "doc_items", []) or []
        primary = doc_items[0] if doc_items else None
        
        try:
            if cfg.include_tables and primary and hasattr(primary, 'export_to_dataframe'):
                try:
                    df = primary.export_to_dataframe()
                    if df is not None:
                        md = df.to_markdown(index=False)
                        caption = _norm(getattr(primary, "caption", ""))
                        seg = (caption + "\n" + md).strip() if caption else md
                        if seg:
                            segments.append((_get_section_from_chunk(chunk), seg))
                except:
                    txt = _norm(getattr(chunk, "text", ""))
                    if txt:
                        segments.append((_get_section_from_chunk(chunk), txt))
            else:
                txt = _norm(getattr(chunk, "text", ""))
                if txt:
                    segments.append((_get_section_from_chunk(chunk), txt))

        except Exception as e:
            logger.warning(f"Error processing chunk {processed_chunks}: {str(e)}")
            txt = _norm(getattr(chunk, "text", ""))
            if txt:
                segments.append((_get_section_from_chunk(chunk), txt))

    logger.info(f"Extracted {len(segments)} segments from {processed_chunks} chunks")
    return segments

import tiktoken
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer


def _pack_segments_by_token(
    tokenizer, 
    segments: List[Tuple[str, str]],
    cfg: DocChunkConfig, 
    file_path: str
) -> List[Dict[str, Any]]:
    """Pack segments into token-sized chunks"""
    
    T = _TokenizerAdapter(tokenizer)
    results: List[Dict[str, Any]] = []

    max_len = cfg.max_token_size
    overlap = max(0, min(cfg.overlap_token_size, cfg.max_token_size - 1))
    sep_tokens = T.encode(cfg.join_separator) if cfg.join_separator else []
    file_type = _detect_file_type(file_path)

    buf: List[int] = []
    buf_section = "Document"
    next_index = 0

    def flush_buffer(section: str):
        nonlocal buf, next_index, buf_section
        if not buf:
            return

        content = T.decode(buf).strip()
        parts = _split_by_tokens_soft(content, T, max_len, overlap)
        total = len(parts)

        for i, p in enumerate(parts, 1):
            payload = _with_breadcrumb(section, p, i, total)
            chunk = {
                "chunk_id": str(uuid.uuid4()),
                "content": payload,
                "tokens": len(T.encode(payload)),
                "order": next_index,
                "hierarchy": section,
                "file_path": file_path,
                "file_type": file_type,
            }
            results.append(chunk)
            next_index += 1

        buf = []

    for section, seg in segments:
        text = seg.strip()
        if not text:
            continue

        if buf and section != buf_section:
            flush_buffer(buf_section)
            buf_section = section

        seg_tokens = T.encode(("" if not buf else cfg.join_separator) + text)

        if len(seg_tokens) > max_len:
            flush_buffer(buf_section)
            step = max_len - overlap if max_len > overlap else max_len
            start = 0
            while start < len(seg_tokens):
                end = min(start + max_len, len(seg_tokens))
                window = seg_tokens[start:end]
                payload_txt = T.decode(window).strip()
                parts = _split_by_tokens_soft(payload_txt, T, max_len, overlap)
                for i, p in enumerate(parts, 1):
                    payload = _with_breadcrumb(section, p, i, len(parts))
                    chunk = {
                        "chunk_id": str(uuid.uuid4()),
                        "content": payload,
                        "tokens": len(T.encode(payload)),
                        "order": next_index,
                        "hierarchy": section,
                        "file_path": file_path,
                        "file_type": file_type,
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