# backend/core/chunking.py 

from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path
import uuid, re, tiktoken, os, logging

logger = logging.getLogger(__name__)

# ================= Check Docling Availability =================
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    DOCLING_AVAILABLE = True
    logger.info(" Docling available for advanced PDF processing")
except ImportError:
    DOCLING_AVAILABLE = False
    logger.warning(" Docling not available, using legacy extractors")

# ================= Config =================
@dataclass
class ChunkConfig:
    max_tokens: int = 300
    overlap_tokens: int = 50
    lookback_ratio: float = 0.2
    table_rows_per_chunk: int = 50 
    join_separator: str = "\n"

DocChunkConfig = ChunkConfig

# ================= Docling Extractor =================
class DoclingExtractor:
    """Docling-based extractor for advanced PDF/DOCX processing"""
    
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
            raise ImportError("Docling not installed")
        
        # Get device from env
        device = os.getenv('DOCLING_DEVICE', 'cpu')
        
        # Configure pipeline
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = os.getenv('DOCLING_OCR', 'false').lower() == 'true'
        pipeline_options.do_table_structure = True
        
        self.converter = DocumentConverter(
            allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
            pipeline_options=pipeline_options
        )
        
        logger.info(f"✅ Docling initialized (device={device}, ocr={pipeline_options.do_ocr})")
        self._initialized = True
    
    def extract(self, filepath: str) -> str:
        """Extract text with structure preservation"""
        try:
            result = self.converter.convert(filepath)
            
            # Try built-in markdown export first
            try:
                markdown = result.document.export_to_markdown()
                if markdown and len(markdown.strip()) > 0:
                    return markdown
            except:
                pass
            
            # Fallback: custom export
            return self._custom_export(result)
        
        except Exception as e:
            logger.error(f"❌ Docling extraction failed: {e}")
            raise
    
    def _custom_export(self, result) -> str:
        """Custom structured export"""
        lines = []
        
        try:
            for element in result.document.iterate_items():
                label = getattr(element, 'label', 'unknown')
                text = getattr(element, 'text', '').strip()
                
                if not text:
                    continue
                
                if label == "title":
                    lines.append(f"# {text}")
                elif label == "section_header":
                    lines.append(f"## {text}")
                elif label == "subtitle":
                    lines.append(f"### {text}")
                elif label == "table":
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
                else:
                    lines.append(text)
                
                lines.append("")  
            
            return "\n".join(lines)
        
        except:
            # Final fallback
            return result.document.export_to_text()

# ================= Helpers =================
_SENTENCE_BREAK = re.compile(
    r"(?s)(.*?)([\.!?…]|(?:\n{2,})|(?:\r?\n- )|(?:\r?\n• ))\s+$"
)

def _with_breadcrumb(section: str, content: str, part_idx: int, part_total: int) -> str:
    suffix = f" — tiếp {part_idx}/{part_total}" if part_total > 1 else ""
    return f"**[SECTION] {section}{suffix}**\n\n{content}"

def _soft_split(text: str, enc, max_size: int, overlap: int, lookback_ratio: float = 0.2) -> List[str]:
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
    """Chunk table text giữ header cố định"""
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

# ================= Chunker =================
class Chunker:
    def __init__(self, config: ChunkConfig):
        self.config = config
        self.enc = tiktoken.encoding_for_model("gpt-4o-mini")

    def count_tokens(self, text: str) -> int:
        return len(self.enc.encode(text))

    def chunk_text(self, text: str, filepath: str, section: str = "ROOT") -> List[Dict]:
        """Chunk text theo token, xử lý table & soft split"""
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

# ================= File Extraction (HYBRID) =================
def extract_text_from_file(filepath: str) -> str:
    """
    🔄 HYBRID EXTRACTION:
    - PDF: Try Docling first, fallback to pdfplumber
    - DOCX/TXT/Excel/etc: Use legacy extractors (already optimized)
    """
    ext = Path(filepath).suffix.lower()
    
    # ✅ PDF: Try Docling first (if enabled and available)
    if ext == '.pdf':
        use_docling = os.getenv('USE_DOCLING', 'true').lower() == 'true'
        
        if use_docling and DOCLING_AVAILABLE:
            try:
                logger.info(f"🚀 Using Docling for PDF: {Path(filepath).name}")
                extractor = DoclingExtractor()
                text = extractor.extract(filepath)
                
                if text and len(text.strip()) > 0:
                    logger.info(f"✅ Docling success: {len(text)} chars")
                    return text
                else:
                    logger.warning("⚠️ Docling returned empty, using fallback")
            
            except Exception as e:
                logger.warning(f"⚠️ Docling failed ({e}), using pdfplumber fallback")
        
        # Fallback: pdfplumber
        return _extract_pdf_legacy(filepath)
    
    # ⚡ DOCX: Legacy (python-docx is already good)
    elif ext in ['.docx', '.doc']:
        return _extract_docx_legacy(filepath)
    
    # ⚡ Markdown
    elif ext in ['.md', '.markdown']:
        return _extract_markdown(filepath)
    
    # ⚡ HTML
    elif ext in ['.html', '.htm']:
        return _extract_html(filepath)
    
    # ⚡ JSON
    elif ext == '.json':
        return _extract_json(filepath)
    
    # ⚡ XML
    elif ext == '.xml':
        return _extract_xml(filepath)
    
    # ⚡ Text & Code
    elif ext in ['.txt', '.py', '.js', '.java', '.cpp', '.c', '.h', '.hpp', 
                 '.css', '.scss', '.sql', '.sh', '.bash', '.yml', '.yaml',
                 '.toml', '.ini', '.cfg', '.conf', '.log', '.r', '.rb', 
                 '.php', '.go', '.rs', '.swift', '.kt', '.ts', '.jsx', '.tsx']:
        return _extract_text(filepath)
    
    # ⚡ Excel
    elif ext in ['.xlsx', '.xls']:
        return _extract_excel(filepath)
    
    # ⚡ CSV
    elif ext == '.csv':
        return _extract_csv(filepath)
    
    # ⚡ PPTX
    elif ext in ['.pptx', '.ppt']:
        return _extract_pptx(filepath)
    
    # Unsupported
    else:
        logger.warning(f"⚠️ Unsupported file type: {ext}")
        return ""


# ================= Legacy Extractors =================

def _extract_pdf_legacy(filepath: str) -> str:
    """Legacy PDF extraction with pdfplumber"""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.error(f"❌ PDF extraction error: {e}")
        return ""


def _extract_docx_legacy(filepath: str) -> str:
    """Legacy DOCX extraction with python-docx"""
    try:
        from docx import Document
        
        doc = Document(filepath)
        output = []
        
        # Process paragraphs with formatting
        for para in doc.paragraphs:
            if not para.text.strip():
                output.append("")
                continue
            
            # Build formatted text (Markdown style)
            text_parts = []
            for run in para.runs:
                text = run.text
                if not text:
                    continue
                
                # Apply basic formatting
                if run.bold and run.italic:
                    text = f"***{text}***"
                elif run.bold:
                    text = f"**{text}**"
                elif run.italic:
                    text = f"*{text}*"
                
                if run.underline:
                    text = f"<u>{text}</u>"
                
                if run.font.strike:
                    text = f"~~{text}~~"
                
                # Color support (if available)
                if run.font.color and run.font.color.rgb:
                    try:
                        rgb = run.font.color.rgb
                        if rgb != (0, 0, 0):  # Not black
                            hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                            text = f'<span style="color:{hex_color}">{text}</span>'
                    except:
                        pass  # Ignore color errors
                
                text_parts.append(text)
            
            full_text = "".join(text_parts)
            
            # Apply heading styles
            style = para.style.name.lower()
            if 'heading 1' in style or 'title' in style:
                full_text = f"# {full_text}"
            elif 'heading 2' in style or 'subtitle' in style:
                full_text = f"## {full_text}"
            elif 'heading 3' in style:
                full_text = f"### {full_text}"
            elif 'heading 4' in style:
                full_text = f"#### {full_text}"
            elif 'quote' in style:
                full_text = f"> {full_text}"
            
            output.append(full_text)
        
        # Process tables (Markdown format)
        for table in doc.tables:
            if not table.rows:
                continue
            
            try:
                # Header row
                header_cells = [cell.text.strip() for cell in table.rows[0].cells]
                output.append("\n| " + " | ".join(header_cells) + " |")
                output.append("| " + " | ".join(["---"] * len(header_cells)) + " |")
                
                # Data rows
                for row in table.rows[1:]:
                    cells = [cell.text.strip() for cell in row.cells]
                    output.append("| " + " | ".join(cells) + " |")
                
                output.append("")
            except Exception as e:
                logger.warning(f"⚠️ Table extraction warning: {e}")
                continue
        
        result = "\n".join(output)
        return result if result.strip() else ""
        
    except ImportError:
        # Fallback nếu chưa cài python-docx
        logger.warning("⚠️ python-docx not installed, using basic extraction")
        try:
            import docx2txt
            return docx2txt.process(filepath)
        except:
            return ""
    except Exception as e:
        logger.error(f"❌ DOCX extraction error: {e}")
        # Fallback to docx2txt
        try:
            import docx2txt
            return docx2txt.process(filepath)
        except:
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
            # Remove script and style elements
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


# ================= Main Entry Point =================
def process_document_to_chunks(filepath: str, config: ChunkConfig = None) -> List[Dict]:
    """Main entry point for document processing"""
    config = config or ChunkConfig()
    
    logger.info(f"📄 Processing: {filepath}")
    text = extract_text_from_file(filepath)
    
    if not text.strip():
        logger.warning(f"⚠️ Warning: No text extracted from {filepath}")
        return []
    
    logger.info(f"✅ Extracted {len(text)} characters")
    
    chunker = Chunker(config)
    chunks = chunker.chunk_text(text, filepath)
    
    logger.info(f"✅ Created {len(chunks)} chunks")
    return chunks