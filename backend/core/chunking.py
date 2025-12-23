# backend/core/chunking.py 

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
    logger.info("✅ Docling khả dụng cho xử lý PDF/DOCX")
except ImportError:
    DOCLING_AVAILABLE = False
    logger.error("❌ Docling không khả dụng! Xử lý PDF/DOCX sẽ thất bại.")

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
            )
        
        try:
            self.converter = DocumentConverter()
            logger.info("✅ Đã khởi tạo Docling (sử dụng API mới nhất)")
        
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
                logger.info(f"✅ Đã khởi tạo Docling (sử dụng API cũ, thiết bị={device}, ocr={do_ocr})")
            
            except Exception as e2:
                logger.error(f"❌ Khởi tạo Docling thất bại: {e2}")
                raise RuntimeError(
                    f"Failed to initialize Docling with both API versions:\n"
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
            logger.info(f"📄 Docling đang trích xuất: {Path(filepath).name}")
            
            # Convert document
            result = self.converter.convert(filepath)
            
            # Try built-in markdown export first
            try:
                markdown = result.document.export_to_markdown()
                if markdown and len(markdown.strip()) > 0:
                    logger.info(f"✅ Xuất Docling markdown: {len(markdown)} ký tự")
                    return markdown
            except Exception as e:
                logger.warning(f"⚠️ Xuất markdown thất bại: {e}, đang thử xuất tùy chỉnh")
            
            # Try text export
            try:
                text = result.document.export_to_text()
                if text and len(text.strip()) > 0:
                    logger.info(f"✅ Xuất văn bản Docling: {len(text)} ký tự")
                    return text
            except Exception as e:
                logger.warning(f"⚠️ Xuất văn bản thất bại: {e}, đang thử xuất tùy chỉnh")
            
            # Fallback: custom export
            text = self._custom_export(result)
            logger.info(f"✅ Xuất tùy chỉnh Docling: {len(text)} ký tự")
            return text
        
        except Exception as e:
            logger.error(f"❌ Trích xuất Docling thất bại cho {Path(filepath).name}: {e}")
            raise RuntimeError(
                f"Docling extraction failed: {e}\n"
                f"Make sure the file is not corrupted and Docling is properly installed."
            )
    
    def _custom_export(self, result) -> str:
        """Custom structured export with enhanced formatting and comprehensive element type support"""
        lines = []
        
        try:
            # Try to iterate through document items
            if hasattr(result.document, 'iterate_items'):
                for element in result.document.iterate_items():
                    label = getattr(element, 'label', 'unknown')
                    text = getattr(element, 'text', '').strip()
                    
                    if not text:
                        continue
                    
                    # Format based on element type - EXPANDED COVERAGE
                    if label == "title":
                        lines.append(f"# {text}")
                    elif label == "section_header":
                        lines.append(f"## {text}")
                    elif label == "subtitle":
                        lines.append(f"### {text}")
                    elif label in ["heading", "heading_1", "h1"]:
                        lines.append(f"# {text}")
                    elif label in ["heading_2", "h2"]:
                        lines.append(f"## {text}")
                    elif label in ["heading_3", "h3"]:
                        lines.append(f"### {text}")
                    elif label in ["heading_4", "h4"]:
                        lines.append(f"#### {text}")
                    elif label in ["heading_5", "h5"]:
                        lines.append(f"##### {text}")
                    elif label in ["heading_6", "h6"]:
                        lines.append(f"###### {text}")
                    
                    # Table handling with improved error logging
                    elif label == "table":
                        if hasattr(element, 'export_to_dataframe'):
                            try:
                                df = element.export_to_dataframe()
                                table_md = df.to_markdown(index=False)
                                lines.append(table_md)
                                logger.debug(f"✅ Bảng đã xuất dạng markdown ({len(df)} dòng)")
                            except Exception as table_err:
                                logger.warning(f"⚠️ Xuất bảng sang dataframe thất bại: {table_err}, sử dụng văn bản thô")
                                lines.append(f"```\n{text}\n```")
                        else:
                            logger.debug(f"ℹ️ Bảng không có export_to_dataframe, sử dụng văn bản thô")
                            lines.append(f"```\n{text}\n```")
                    
                    # Lists
                    elif label == "list_item":
                        lines.append(f"- {text}")
                    elif label in ["numbered_list", "ordered_list"]:
                        lines.append(f"1. {text}")
                    elif label in ["bullet_list", "unordered_list"]:
                        lines.append(f"- {text}")
                    
                    # Code blocks
                    elif label in ["code_block", "code", "pre"]:
                        lines.append(f"```\n{text}\n```")
                    
                    # Formulas and equations
                    elif label in ["formula", "equation", "math"]:
                        lines.append(f"$$\n{text}\n$$")
                    
                    # Quotes
                    elif label in ["quote", "block_quote", "blockquote"]:
                        quoted = "\n".join(f"> {line}" for line in text.split("\n"))
                        lines.append(quoted)
                    
                    # Figures and images (preserve caption)
                    elif label in ["figure", "picture", "image"]:
                        lines.append(f"![Figure: {text}]")
                    
                    # Headers and footers
                    elif label in ["page_header", "header"]:
                        lines.append(f"*[Header: {text}]*")
                    elif label in ["page_footer", "footer"]:
                        lines.append(f"*[Footer: {text}]*")
                    
                    # References and citations
                    elif label in ["reference", "citation", "bibliography"]:
                        lines.append(f"[^{text}]")
                    
                    # Text boxes and callouts
                    elif label in ["text_box", "callout", "note"]:
                        lines.append(f"📌 **Note:** {text}")
                    
                    # Captions and footnotes
                    elif label == "caption":
                        lines.append(f"*{text}*")
                    elif label == "footnote":
                        lines.append(f"[^{text}]")
                    
                    # Paragraph (default)
                    elif label == "paragraph":
                        lines.append(text)
                    
                    # Unknown types - log and preserve
                    else:
                        logger.warning(f"⚠️ Loại phần tử không xác định '{label}', giữ nguyên dạng văn bản")
                        lines.append(text)
                    
                    lines.append("")  # Add spacing
            
            # If we got content, return it
            if lines:
                return "\n".join(lines)
            
            # Final fallback: try to get raw text
            if hasattr(result.document, 'text'):
                logger.warning("⚠️ Sử dụng document.text dự phòng")
                return result.document.text
            
            # Last resort: try dict representation
            if hasattr(result.document, 'to_dict'):
                logger.warning("⚠️ Sử dụng document.to_dict dự phòng")
                doc_dict = result.document.to_dict()
                return str(doc_dict.get('text', ''))
            
            logger.error("❌ Tất cả phương thức xuất đều thất bại, trả về chuỗi rỗng")
            return ""
        
        except Exception as e:
            logger.error(f"❌ Xuất tùy chỉnh thất bại: {e}")
            
            # Try to get any text from result
            try:
                if hasattr(result.document, 'text'):
                    return result.document.text
                if hasattr(result.document, 'to_dict'):
                    return str(result.document.to_dict().get('text', ''))
                logger.error("❌ Không có phương thức dự phòng khả dụng")
                return ""
            except Exception as fallback_err:
                logger.error(f"❌ Phương thức dự phòng cũng thất bại: {fallback_err}")
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
    """Split markdown tables intelligently - FIXED: preserve text after table"""
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
    
    # ✅ FIX: Don't lose text after table!
    suffix = "\n".join(lines[block_end:]).strip()
    
    # Process table
    table_chunks = _split_table_text(header + "\n" + separator, rows, prefix, enc, max_tokens, overlap)
    
    # ✅ FIX: Append suffix to last chunk or create new chunk
    if suffix:
        if table_chunks:
            # Try to append suffix to last chunk
            last_chunk = table_chunks[-1]
            combined = last_chunk + "\n\n" + suffix
            combined_tokens = len(enc.encode(combined))
            
            if combined_tokens <= max_tokens:
                # Can fit in last chunk
                table_chunks[-1] = combined
            else:
                # Need separate chunks for suffix
                suffix_chunks = _soft_split(suffix, enc, max_tokens, overlap)
                table_chunks.extend(suffix_chunks)
        else:
            # No table chunks, just return suffix
            table_chunks = _soft_split(suffix, enc, max_tokens, overlap)
    
    return table_chunks

# Chunker 
class Chunker:
    """
    Token-aware text chunker with heading hierarchy tracking
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
        self.heading_stack = [] 

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.enc.encode(text))
    
    def _parse_heading(self, line: str) -> tuple:
        """Parse markdown heading and return (level, title) or (0, None)"""
        line = line.strip()
        if line.startswith('#'):
            level = 0
            while level < len(line) and line[level] == '#':
                level += 1
            if level <= 6 and level < len(line) and line[level] == ' ':
                title = line[level:].strip()
                return (level, title)
        return (0, None)
    
    def _update_heading_stack(self, level: int, title: str):
        """Update heading hierarchy stack"""
        # Remove headings at same or lower level
        self.heading_stack = [h for h in self.heading_stack if h[0] < level]
        # Add new heading
        self.heading_stack.append((level, title))
    
    def _get_current_section(self) -> str:
        """Get current section path from heading stack"""
        if not self.heading_stack:
            return "ROOT"
        return " > ".join(h[1] for h in self.heading_stack)

    def chunk_text(self, text: str, filepath: str, section: str = "ROOT") -> List[Dict]:
        """
        Chunk text with token-aware splitting, table handling, and heading hierarchy tracking
        """
        lines = text.split('\n')
        chunks, buf, buf_tokens, order = [], [], 0, 0
        self.heading_stack = []  # Reset hierarchy for new document

        for line in lines:
            # Check if line is a heading
            level, title = self._parse_heading(line)
            if level > 0:
                self._update_heading_stack(level, title)
            
            line_tokens = self.count_tokens(line)
            if buf_tokens + line_tokens > self.config.max_tokens and buf:
                combined = "\n".join(buf).strip()
                pieces = _split_table_markdown(combined, self.enc, self.config.max_tokens, self.config.overlap_tokens)
                current_section = self._get_current_section()
                for i, p in enumerate(pieces, 1):
                    #  Skip empty or tiny chunks
                    if len(p.strip()) < 10:
                        logger.warning(f"⚠️ Bỏ qua chunk nhỏ ({len(p)} ký tự): {p[:50]}...")
                        continue
                    
                    chunks.append({
                        "chunk_id": str(uuid.uuid4()),
                        "content": _with_breadcrumb(current_section, p, i, len(pieces)),
                        "tokens": self.count_tokens(p),
                        "order": order,
                        "file_path": filepath,
                        "file_type": Path(filepath).suffix[1:].upper(),
                        "section": current_section
                    })
                    order += 1
                #  set overlap to 5 lines for better context
                overlap_lines = buf[-5:] if len(buf) > 5 else buf
                buf = overlap_lines + [line]
                buf_tokens = sum(self.count_tokens(l) for l in buf)
            else:
                buf.append(line)
                buf_tokens += line_tokens

        if buf:
            combined = "\n".join(buf).strip()
            pieces = _split_table_markdown(combined, self.enc, self.config.max_tokens, self.config.overlap_tokens)
            current_section = self._get_current_section()
            for i, p in enumerate(pieces, 1):
                if len(p.strip()) < 10:
                    logger.warning(f"⚠️ Skipping tiny chunk ({len(p)} chars): {p[:50]}...")
                    continue
                
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "content": _with_breadcrumb(current_section, p, i, len(pieces)),
                    "tokens": self.count_tokens(p),
                    "order": order,
                    "file_path": filepath,
                    "file_type": Path(filepath).suffix[1:].upper(),
                    "section": current_section
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
            logger.warning(f"⚠️ USE_DOCLING=false nhưng {ext} yêu cầu Docling! Đang bật...")
        
        # Force Docling for PDF/DOCX
        logger.info(f"📄 Sử dụng Docling cho {ext}: {Path(filepath).name}")
        extractor = DoclingExtractor()
        text = extractor.extract(filepath)
        
        if not text or len(text.strip()) == 0:
            raise RuntimeError(
                f" Docling returned empty text for {Path(filepath).name}\n"
                f"The file may be corrupted or empty."
            )
        
        logger.info(f"✅ Docling đã trích xuất {len(text)} ký tự từ {Path(filepath).name}")
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
        logger.warning(f"⚠️ Loại file không được hỗ trợ: {ext}")
        return ""


def _extract_markdown(filepath: str) -> str:
    """Markdown extraction"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"❌ Lỗi trích xuất Markdown: {e}")
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
        logger.error(f"❌ Lỗi trích xuất HTML: {e}")
        return ""


def _extract_json(filepath: str) -> str:
    """JSON extraction"""
    try:
        import json
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return json.dumps(data, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"❌ Lỗi trích xuất JSON: {e}")
        return ""


def _extract_xml(filepath: str) -> str:
    """XML extraction"""
    try:
        from bs4 import BeautifulSoup
        with open(filepath, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'xml')
            return soup.get_text(separator='\n', strip=True)
    except Exception as e:
        logger.error(f"❌ Lỗi trích xuất XML: {e}")
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
            logger.error(f"❌ Lỗi trích xuất file văn bản: {e}")
            return ""
    except Exception as e:
        logger.error(f"❌ Lỗi trích xuất file văn bản: {e}")
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
        logger.error(f"❌ Lỗi trích xuất Excel: {e}")
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
        logger.error(f"❌ Lỗi trích xuất CSV: {e}")
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
        logger.error(f"❌ Lỗi trích xuất PPTX: {e}")
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
    
    logger.info(f"📄 Đang xử lý: {filepath}")
    
    text = extract_text_from_file(filepath)
    
    if not text.strip():
        logger.warning(f"⚠️ Cảnh báo: Không trích xuất được văn bản từ {filepath}")
        return []
    
    logger.info(f"✅ Đã trích xuất {len(text)} ký tự")
    

    chunker = Chunker(config)
    chunks = chunker.chunk_text(text, filepath)
    
    logger.info(f"✅ Đã tạo {len(chunks)} chunks")
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