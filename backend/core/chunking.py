# backend/core/chunking.py - DOCLING ONLY (FIXED API)

from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path
import uuid, re, tiktoken, os, logging

logger = logging.getLogger(__name__)

#  Check Docling Availability 
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
    logger.info(" Docling available for PDF/DOCX processing")
except ImportError:
    DOCLING_AVAILABLE = False
    logger.error(" Docling not available! PDF/DOCX processing will fail.")

#  Config 
@dataclass
class ChunkConfig:
    max_tokens: int = 300
    overlap_tokens: int = 50
    lookback_ratio: float = 0.2
    table_rows_per_chunk: int = 50 
    join_separator: str = "\n"

DocChunkConfig = ChunkConfig

class DoclingExtractor:
    """
    🚀 DOCLING-ONLY EXTRACTOR
    Required for PDF and DOCX processing
    """
    
    _instance = None  
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        if not DOCLING_AVAILABLE:
            raise ImportError(
                "❌ Docling is required but not installed!\n"
                "Install with: pip install docling\n"
                "Or: pip install -r requirements.txt"
            )
        
        try:
            self.converter = DocumentConverter()
            logger.info(" Docling initialized (using latest API)")
        
        except TypeError as e:
            try:
                from docling.datamodel.pipeline_options import PdfPipelineOptions
                
                device = os.getenv('DOCLING_DEVICE', 'cpu')
                do_ocr = os.getenv('DOCLING_OCR', 'false').lower() == 'true'
                
                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_ocr = do_ocr
                pipeline_options.do_table_structure = True
                
                self.converter = DocumentConverter(
                    allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
                    pipeline_options=pipeline_options
                )
                logger.info(f" Docling initialized (using old API, device={device}, ocr={do_ocr})")
            
            except Exception as e2:
                logger.error(f" Docling initialization failed: {e2}")
                raise RuntimeError(
                    f"Failed to initialize Docling with both API versions:\n"
                    f"New API error: {e}\n"
                    f"Old API error: {e2}\n"
                    f"Please check your Docling version: pip show docling"
                )
        
        self._initialized = True
    
    def extract(self, filepath: str) -> str:
        """
        Extract text with structure preservation
        
        Args:
            filepath: Path to PDF or DOCX file
            
        Returns:
            Structured markdown text
            
        Raises:
            Exception if extraction fails
        """
        try:
            logger.info(f" Docling extracting: {Path(filepath).name}")
            
            # Convert document
            result = self.converter.convert(filepath)
            
            # Try built-in markdown export first
            try:
                markdown = result.document.export_to_markdown()
                if markdown and len(markdown.strip()) > 0:
                    logger.info(f" Docling markdown export: {len(markdown)} chars")
                    return markdown
            except Exception as e:
                logger.warning(f" Markdown export failed: {e}, trying custom export")
            
            # Try text export
            try:
                text = result.document.export_to_text()
                if text and len(text.strip()) > 0:
                    logger.info(f" Docling text export: {len(text)} chars")
                    return text
            except Exception as e:
                logger.warning(f" Text export failed: {e}, trying custom export")
            
            # Fallback: custom export
            text = self._custom_export(result)
            logger.info(f" Docling custom export: {len(text)} chars")
            return text
        
        except Exception as e:
            logger.error(f" Docling extraction failed for {Path(filepath).name}: {e}")
            raise RuntimeError(
                f"Docling extraction failed: {e}\n"
                f"Make sure the file is not corrupted and Docling is properly installed."
            )
    
    def _custom_export(self, result) -> str:
        """Custom structured export with enhanced formatting"""
        lines = []
        
        try:
            # Try to iterate through document items
            if hasattr(result.document, 'iterate_items'):
                for element in result.document.iterate_items():
                    label = getattr(element, 'label', 'unknown')
                    text = getattr(element, 'text', '').strip()
                    
                    if not text:
                        continue
                    
                    # Format based on element type
                    if label == "title":
                        lines.append(f"# {text}")
                    elif label == "section_header":
                        lines.append(f"## {text}")
                    elif label == "subtitle":
                        lines.append(f"### {text}")
                    elif label == "table":
                        # Try to export table as markdown
                        if hasattr(element, 'export_to_dataframe'):
                            try:
                                df = element.export_to_dataframe()
                                lines.append(df.to_markdown(index=False))
                            except:
                                lines.append(text)
                        else:
                            lines.append(text)
                    elif label == "list_item":
                        lines.append(f"- {text}")
                    elif label == "paragraph":
                        lines.append(text)
                    elif label == "caption":
                        lines.append(f"*{text}*")
                    elif label == "footnote":
                        lines.append(f"^{text}")
                    else:
                        lines.append(text)
                    
                    lines.append("")  # Add spacing
            
            # If we got content, return it
            if lines:
                return "\n".join(lines)
            
            # Final fallback: try to get raw text
            if hasattr(result.document, 'text'):
                return result.document.text
            
            # Last resort: convert result to string
            return str(result.document)
        
        except Exception as e:
            logger.warning(f" Custom export failed: {e}")
            
            # Try to get any text from result
            try:
                if hasattr(result.document, 'text'):
                    return result.document.text
                return str(result.document)
            except:
                raise RuntimeError(f"Cannot extract text from document: {e}")


# Helpers 
_SENTENCE_BREAK = re.compile(
    r"(?s)(.*?)([\.!?…]|(?:\n{2,})|(?:\r?\n- )|(?:\r?\n• ))\s+$"
)

def _with_breadcrumb(section: str, content: str, part_idx: int, part_total: int) -> str:
    """Add section breadcrumb to chunk"""
    suffix = f" — part {part_idx}/{part_total}" if part_total > 1 else ""
    return f"**[SECTION] {section}{suffix}**\n\n{content}"

def _soft_split(text: str, enc, max_size: int, overlap: int, lookback_ratio: float = 0.2) -> List[str]:
    """Smart text splitting with sentence boundaries"""
    ids = enc.encode(text)
    n = len(ids)
    if n <= max_size:
        return [text]

    out = []
    start = 0

    while start < n:
        end = min(start + max_size, n)
        window_ids = ids[start:end]
        window_text = enc.decode(window_ids)

        if end < n:
            lb_chars = max(10, int(len(window_text)*(1-lookback_ratio)))
            tail = window_text[lb_chars:]
            m = _SENTENCE_BREAK.search(tail)
            if m:
                cut_char = lb_chars + m.end()
                window_ids = enc.encode(window_text[:cut_char])

        piece = enc.decode(window_ids).rstrip()
        out.append(piece)
        if start + len(window_ids) >= n:
            break
        start += len(window_ids) - overlap
    return out

def _split_table_text(header: str, rows: List[str], prefix: str, enc, max_tokens: int, overlap: int) -> List[str]:
    """Chunk table text with sticky header"""
    sticky = header + "\n"
    chunks, cur_rows = [], []

    def emit():
        if not cur_rows:
            return
        body = "".join([r+"\n" for r in cur_rows])
        content = f"{prefix}\n\n{sticky}{body}".strip() if prefix else f"{sticky}{body}".strip()
        if len(enc.encode(content)) > max_tokens:
            chunks.extend(_soft_split(content, enc, max_tokens, overlap))
        else:
            chunks.append(content)

    for r in rows:
        candidate = cur_rows + [r]
        body = "".join([x+"\n" for x in candidate])
        content = f"{prefix}\n\n{sticky}{body}".strip() if prefix else f"{sticky}{body}".strip()
        if len(enc.encode(content)) <= max_tokens:
            cur_rows.append(r)
        else:
            emit()
            cur_rows = [r]
    emit()
    if not chunks:
        chunks.append(f"{prefix}\n\n{sticky}".strip() if prefix else sticky.strip())
    return chunks

def _split_table_markdown(text: str, enc, max_tokens: int, overlap: int) -> List[str]:
    """Split markdown tables intelligently"""
    lines = text.splitlines()
    block_start, block_end = None, None
    for i in range(len(lines)-1):
        s1, s2 = lines[i].strip(), lines[i+1].strip()
        if s1.startswith("|") and s1.endswith("|") and all(c in "-: " for c in s2.replace("|","")):
            block_start, block_end = i, i+2
            while block_end < len(lines) and lines[block_end].lstrip().startswith("|"):
                block_end += 1
            break
    if block_start is None:
        return _soft_split(text, enc, max_tokens, overlap)

    prefix = "\n".join(lines[:block_start]).strip()
    table_lines = lines[block_start:block_end]
    header, separator = table_lines[:2]
    rows = table_lines[2:]

    return _split_table_text(header + "\n" + separator, rows, prefix, enc, max_tokens, overlap)

# Chunker 
class Chunker:
    """
    Token-aware text chunker
    """
    def __init__(self, config: ChunkConfig = None):
        if config is None:
            try:
                from backend.config import Config
                config = ChunkConfig(
                    max_tokens=Config.DEFAULT_CHUNK_SIZE,
                    overlap_tokens=Config.DEFAULT_CHUNK_OVERLAP
                )
            except:
                config = ChunkConfig()
        
        self.config = config
        self.enc = tiktoken.encoding_for_model("gpt-4o-mini")

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.enc.encode(text))

    def chunk_text(self, text: str, filepath: str, section: str = "ROOT") -> List[Dict]:
        """
        Chunk text with token-aware splitting and table handling
        """
        lines = text.split('\n')
        chunks, buf, buf_tokens, order = [], [], 0, 0

        for line in lines:
            line_tokens = self.count_tokens(line)
            if buf_tokens + line_tokens > self.config.max_tokens and buf:
                combined = "\n".join(buf).strip()
                pieces = _split_table_markdown(combined, self.enc, self.config.max_tokens, self.config.overlap_tokens)
                for i, p in enumerate(pieces, 1):
                    chunks.append({
                        "chunk_id": str(uuid.uuid4()),
                        "content": _with_breadcrumb(section, p, i, len(pieces)),
                        "tokens": self.count_tokens(p),
                        "order": order,
                        "file_path": filepath,
                        "file_type": Path(filepath).suffix[1:].upper(),
                        "section": section
                    })
                    order += 1
                overlap_lines = buf[-3:] if len(buf) > 3 else buf
                buf = overlap_lines + [line]
                buf_tokens = sum(self.count_tokens(l) for l in buf)
            else:
                buf.append(line)
                buf_tokens += line_tokens

        if buf:
            combined = "\n".join(buf).strip()
            pieces = _split_table_markdown(combined, self.enc, self.config.max_tokens, self.config.overlap_tokens)
            for i, p in enumerate(pieces, 1):
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "content": _with_breadcrumb(section, p, i, len(pieces)),
                    "tokens": self.count_tokens(p),
                    "order": order,
                    "file_path": filepath,
                    "file_type": Path(filepath).suffix[1:].upper(),
                    "section": section
                })
                order += 1

        return chunks


#  File Extraction (DOCLING for PDF/DOCX) 

def extract_text_from_file(filepath: str) -> str:
    """ EXTRACTION with DOCLING-ONLY for PDF/DOCX
    Strategy:
    - PDF & DOCX: MANDATORY Docling (no fallback)
    - Other formats: Legacy extractors
    
    Args:
        filepath: Path to document
        
    Returns:
        Extracted text
        
    Raises:
        RuntimeError if Docling fails for PDF/DOCX
    """
    ext = Path(filepath).suffix.lower()
    
    if ext in ['.pdf', '.docx', '.doc']:
        if not DOCLING_AVAILABLE:
            raise RuntimeError(
                f" Docling is required for {ext} files but not installed!\n"
                f"Install with: pip install docling"
            )
        
        use_docling = os.getenv('USE_DOCLING', 'true').lower() == 'true'
        
        if not use_docling:
            logger.warning(f" USE_DOCLING=false but {ext} requires Docling! Enabling...")
        
        # Force Docling for PDF/DOCX
        logger.info(f" Using Docling for {ext}: {Path(filepath).name}")
        extractor = DoclingExtractor()
        text = extractor.extract(filepath)
        
        if not text or len(text.strip()) == 0:
            raise RuntimeError(
                f" Docling returned empty text for {Path(filepath).name}\n"
                f"The file may be corrupted or empty."
            )
        
        logger.info(f" Docling extracted {len(text)} chars from {Path(filepath).name}")
        return text
    
    # ⚡ Other formats: Use legacy extractors
    elif ext in ['.md', '.markdown']:
        return _extract_markdown(filepath)
    
    elif ext in ['.html', '.htm']:
        return _extract_html(filepath)
    
    elif ext == '.json':
        return _extract_json(filepath)
    
    elif ext == '.xml':
        return _extract_xml(filepath)
    
    elif ext in ['.txt', '.py', '.js', '.java', '.cpp', '.c', '.h', '.hpp', 
                 '.css', '.scss', '.sql', '.sh', '.bash', '.yml', '.yaml',
                 '.toml', '.ini', '.cfg', '.conf', '.log', '.r', '.rb', 
                 '.php', '.go', '.rs', '.swift', '.kt', '.ts', '.jsx', '.tsx']:
        return _extract_text(filepath)
    
    elif ext in ['.xlsx', '.xls']:
        return _extract_excel(filepath)
    
    elif ext == '.csv':
        return _extract_csv(filepath)
    
    elif ext in ['.pptx', '.ppt']:
        return _extract_pptx(filepath)
    
    else:
        logger.warning(f"⚠️ Unsupported file type: {ext}")
        return ""


def _extract_markdown(filepath: str) -> str:
    """Markdown extraction"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"❌ Markdown extraction error: {e}")
        return ""


def _extract_html(filepath: str) -> str:
    """HTML extraction"""
    try:
        from bs4 import BeautifulSoup
        with open(filepath, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator='\n', strip=True)
    except Exception as e:
        logger.error(f"❌ HTML extraction error: {e}")
        return ""


def _extract_json(filepath: str) -> str:
    """JSON extraction"""
    try:
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"❌ JSON extraction error: {e}")
        return ""


def _extract_xml(filepath: str) -> str:
    """XML extraction"""
    try:
        from bs4 import BeautifulSoup
        with open(filepath, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'xml')
            return soup.get_text(separator='\n', strip=True)
    except Exception as e:
        logger.error(f"❌ XML extraction error: {e}")
        return ""


def _extract_text(filepath: str) -> str:
    """Text file extraction"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                return f.read()
        except Exception as e:
            logger.error(f"❌ Text file extraction error: {e}")
            return ""
    except Exception as e:
        logger.error(f"❌ Text file extraction error: {e}")
        return ""


def _extract_excel(filepath: str) -> str:
    """Excel extraction"""
    try:
        import pandas as pd
        dfs = pd.read_excel(filepath, sheet_name=None)
        text_parts = []
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
        
        for sheet_name, df in dfs.items():
            if df.empty:
                continue
            header = " | ".join(str(col) for col in df.columns)
            rows = [" | ".join(map(str, row)) for row in df.values]
            sheet_prefix = f"=== Sheet: {sheet_name} ==="
            text_parts.extend(_split_table_text(header, rows, sheet_prefix, enc,
                                                max_tokens=300, overlap=50))
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.error(f"❌ Excel extraction error: {e}")
        return ""


def _extract_csv(filepath: str) -> str:
    """CSV extraction"""
    try:
        import pandas as pd
        df = pd.read_csv(filepath)
        if df.empty:
            return ""
        enc = tiktoken.encoding_for_model("gpt-4o-mini")
        header = " | ".join(str(col) for col in df.columns)
        rows = [" | ".join(map(str, row)) for row in df.values]
        return "\n\n".join(_split_table_text(header, rows, "", enc,
                                            max_tokens=300, overlap=50))
    except Exception as e:
        logger.error(f"❌ CSV extraction error: {e}")
        return ""


def _extract_pptx(filepath: str) -> str:
    """PPTX extraction"""
    try:
        from pptx import Presentation
        prs = Presentation(filepath)
        text_parts = []
        for idx, slide in enumerate(prs.slides, 1):
            slide_text = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text.append(shape.text.strip())
            if slide_text:
                text_parts.append(f"=== Slide {idx} ===\n" + "\n".join(slide_text))
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.error(f"❌ PPTX extraction error: {e}")
        return ""


#  Main Entry 

def process_document_to_chunks(filepath: str, config: ChunkConfig = None) -> List[Dict]:
    """
    Main entry point for document processing  
    Args:
        filepath: Path to document
        config: Chunking configuration (uses Config defaults if None)
        
    Returns:
        List of chunk dictionaries with format:
        {
            'chunk_id': str,
            'content': str,
            'tokens': int,
            'order': int,
            'file_path': str,
            'file_type': str,
            'section': str
        }
        
    Raises:
        RuntimeError if processing fails for PDF/DOCX
    """
    if config is None:
        try:
            from backend.config import Config
            config = ChunkConfig(
                max_tokens=Config.DEFAULT_CHUNK_SIZE,
                overlap_tokens=Config.DEFAULT_CHUNK_OVERLAP
            )
        except:
            config = ChunkConfig()
    
    logger.info(f"📄 Processing: {filepath}")
    
    text = extract_text_from_file(filepath)
    
    if not text.strip():
        logger.warning(f"⚠️ Warning: No text extracted from {filepath}")
        return []
    
    logger.info(f" Extracted {len(text)} characters")
    

    chunker = Chunker(config)
    chunks = chunker.chunk_text(text, filepath)
    
    logger.info(f" Created {len(chunks)} chunks")
    return chunks


__all__ = [
    'ChunkConfig',
    'DocChunkConfig',  
    'Chunker',
    'DoclingExtractor',
    'extract_text_from_file',
    'process_document_to_chunks',
    'DOCLING_AVAILABLE'
]