# ==========================================
# backend/core/chunking.py
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

# ================= File extraction =================
def extract_text_from_file(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()

    # PDF/DOCX handled as before...

    # Excel
    if ext in ['.xlsx', '.xls']:
        try:
            import pandas as pd
            dfs = pd.read_excel(filepath, sheet_name=None)
            text_parts = []
            for sheet_name, df in dfs.items():
                if df.empty:
                    continue
                header = " | ".join(df.columns)
                rows = [" | ".join(map(str, row)) for row in df.values]
                sheet_prefix = f"=== Sheet: {sheet_name} ==="
                text_parts.extend(_split_table_text(header, rows, sheet_prefix, tiktoken.encoding_for_model("gpt-4o-mini"),
                                                    max_tokens=300, overlap=50))
            return "\n\n".join(text_parts)
        except Exception as e:
            print(f"Excel extraction error: {e}")
            return ""

    # CSV
    elif ext == '.csv':
        try:
            import pandas as pd
            df = pd.read_csv(filepath)
            if df.empty:
                return ""
            header = " | ".join(df.columns)
            rows = [" | ".join(map(str, row)) for row in df.values]
            return "\n\n".join(_split_table_text(header, rows, "", tiktoken.encoding_for_model("gpt-4o-mini"),
                                                max_tokens=300, overlap=50))
        except Exception as e:
            print(f"CSV extraction error: {e}")
            return ""

    # PPTX
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
            print(f"PPTX extraction error: {e}")
            return ""

    # Các định dạng khác (PDF, DOCX, Markdown, HTML, JSON, code, text…) vẫn giữ nguyên như trước
    return ""  # fallback

# ================= Main entry =================
def process_document_to_chunks(filepath: str, config: ChunkConfig = None) -> List[Dict]:
    config = config or ChunkConfig()
    text = extract_text_from_file(filepath)
    if not text.strip():
        print(f"Warning: No text extracted from {filepath}")
        return []
    chunker = Chunker(config)
    return chunker.chunk_text(text, filepath)
