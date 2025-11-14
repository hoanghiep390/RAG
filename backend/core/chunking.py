# ==========================================
# backend/core/chunking.py - FIXED VERSION
# ==========================================
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path
import uuid, re, tiktoken

# ================= Config =================
@dataclass
class ChunkConfig:
    max_tokens: int = 300
    overlap_tokens: int = 50
    lookback_ratio: float = 0.2
    table_rows_per_chunk: int = 50 
    join_separator: str = "\n"

# Compatibility alias: some modules expect DocChunkConfig name
DocChunkConfig = ChunkConfig

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

        # tìm ngắt đẹp trong lookback
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

# ================= File extraction (FIXED) =================
def extract_text_from_file(filepath: str) -> str:
    """Extract text from various file formats"""
    ext = Path(filepath).suffix.lower()

    # ========== PDF ==========
    if ext == '.pdf':
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
            print(f"❌ PDF extraction error: {e}")
            return ""

    # ========== DOCX ==========
    elif ext in ['.docx', '.doc']:
        try:
            import docx2txt
            text = docx2txt.process(filepath)
            return text if text else ""
        except Exception as e:
            print(f"❌ DOCX extraction error: {e}")
            return ""

    # ========== Markdown ==========
    elif ext in ['.md', '.markdown']:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"❌ Markdown extraction error: {e}")
            return ""

    # ========== HTML ==========
    elif ext in ['.html', '.htm']:
        try:
            from bs4 import BeautifulSoup
            with open(filepath, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                return soup.get_text(separator='\n', strip=True)
        except Exception as e:
            print(f"❌ HTML extraction error: {e}")
            return ""

    # ========== JSON ==========
    elif ext == '.json':
        try:
            import json
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return json.dumps(data, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ JSON extraction error: {e}")
            return ""

    # ========== XML ==========
    elif ext == '.xml':
        try:
            from bs4 import BeautifulSoup
            with open(filepath, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'xml')
                return soup.get_text(separator='\n', strip=True)
        except Exception as e:
            print(f"❌ XML extraction error: {e}")
            return ""

    # ========== Text & Code Files ==========
    elif ext in ['.txt', '.py', '.js', '.java', '.cpp', '.c', '.h', '.hpp', 
                 '.css', '.scss', '.sql', '.sh', '.bash', '.yml', '.yaml',
                 '.toml', '.ini', '.cfg', '.conf', '.log', '.r', '.rb', 
                 '.php', '.go', '.rs', '.swift', '.kt', '.ts', '.jsx', '.tsx']:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(filepath, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                print(f"❌ Text file extraction error: {e}")
                return ""
        except Exception as e:
            print(f"❌ Text file extraction error: {e}")
            return ""

    # ========== Excel ==========
    elif ext in ['.xlsx', '.xls']:
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
            print(f"❌ Excel extraction error: {e}")
            return ""

    # ========== CSV ==========
    elif ext == '.csv':
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
            print(f"❌ CSV extraction error: {e}")
            return ""

    # ========== PPTX ==========
    elif ext in ['.pptx', '.ppt']:
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
            print(f"❌ PPTX extraction error: {e}")
            return ""

    # ========== Unsupported ==========
    else:
        print(f"⚠️ Unsupported file type: {ext}")
        return ""

# ================= Main entry =================
def process_document_to_chunks(filepath: str, config: ChunkConfig = None) -> List[Dict]:
    """Main entry point for document processing"""
    config = config or ChunkConfig()
    
    print(f"📄 Processing: {filepath}")
    text = extract_text_from_file(filepath)
    
    if not text.strip():
        print(f"⚠️ Warning: No text extracted from {filepath}")
        return []
    
    print(f"✅ Extracted {len(text)} characters")
    
    chunker = Chunker(config)
    chunks = chunker.chunk_text(text, filepath)
    
    print(f"✅ Created {len(chunks)} chunks")
    return chunks