# backend/core/chunking.py

from __future__ import annotations
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List
from pathlib import Path
from functools import lru_cache

from backend.utils.utils import logger
from backend.utils.file_utils import read_file_content, get_file_extension

try:
    import tiktoken
except ImportError as e:
    logger.error(f"Missing tiktoken: {e}")
    raise

try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

import pandas as pd
import json
import xml.etree.ElementTree as ET
import yaml



@dataclass
class ChunkConfig:
    """Chunking configuration"""
    max_tokens: int = 300
    overlap_tokens: int = 50
    include_hierarchy: bool = True
    merge_small_segments: bool = True



class Tokenizer:
    """Optimized tokenizer with LRU cache"""
    
    def __init__(self):
        self.enc = tiktoken.encoding_for_model("gpt-4o-mini")
    
    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text)
    
    def decode(self, ids: List[int]) -> str:
        return self.enc.decode(ids)
    
    @lru_cache(maxsize=10000)
    def count(self, text: str) -> int:
        """Cached token counting"""
        return len(self.encode(text))



class Segment:
    """Document segment with hierarchy"""
    
    def __init__(self, hierarchy: List[str], content: str):
        self.hierarchy = hierarchy
        self.content = content
        self.tokens = 0
    
    def full_hierarchy(self) -> str:
        return " > ".join(self.hierarchy)


class Chunk:
    """Final chunk with segments"""
    
    def __init__(self, order: int, file_path: str, file_type: str):
        self.chunk_id = str(uuid.uuid4())
        self.order = order
        self.file_path = file_path
        self.file_type = file_type
        self.segments: List[Segment] = []
        self.tokens = 0
    
    def add_segment(self, segment: Segment):
        self.segments.append(segment)
        self.tokens += segment.tokens
    
    def get_content(self, include_hierarchy: bool = True) -> str:
        """Build content with optional hierarchy"""
        if not include_hierarchy:
            return "\n\n".join(s.content for s in self.segments)
        
        parts = [f"**[{s.full_hierarchy()}]**\n{s.content}" for s in self.segments]
        return "\n\n".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        hierarchies = list(dict.fromkeys([s.full_hierarchy() for s in self.segments]))
        return {
            "chunk_id": self.chunk_id,
            "content": self.get_content(),
            "tokens": self.tokens,
            "order": self.order,
            "hierarchy": hierarchies,
            "hierarchy_list": hierarchies,
            "file_path": self.file_path,
            "file_type": self.file_type,
        }



def extract_pdf(path: str) -> List[Segment]:
    """Extract from PDF"""
    if not pdfplumber:
        return []
    
    segments = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                segments.append(Segment([f"Page {page_num}"], text))
            
            
            for i, table in enumerate(page.extract_tables() or []):
                if table:
                    df = pd.DataFrame(table[1:], columns=table[0]) if len(table) > 1 else pd.DataFrame(table)
                    segments.append(Segment([f"Page {page_num}", f"Table {i+1}"], df.to_string(index=False)))
    
    return segments


def extract_docx(path: str) -> List[Segment]:
    """Extract from DOCX"""
    if not DocxDocument:
        return []
    
    segments = []
    doc = DocxDocument(path)
    section_stack = ["Document"]
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        
        
        if para.style.name.startswith('Heading'):
            level = int(para.style.name.replace('Heading', '').strip() or '1')
            section_stack = section_stack[:level]
            section_stack.append(text)
        
        segments.append(Segment(section_stack.copy(), text))
    
    
    for i, table in enumerate(doc.tables):
        df = pd.DataFrame([[cell.text for cell in row.cells] for row in table.rows])
        segments.append(Segment(["Document", f"Table {i+1}"], df.to_string(index=False)))
    
    return segments


def extract_text(path: str) -> List[Segment]:
    """Extract from text/markdown"""
    content = read_file_content(path)
    segments = []
    section_stack = ["Document"]
    
    for line in content.splitlines():
        text = line.strip()
        if not text:
            continue
        
        if text.startswith('#'):
            level = len(text) - len(text.lstrip('#'))
            heading = text.lstrip('#').strip()
            section_stack = section_stack[:level]
            section_stack.append(heading)
        
        segments.append(Segment(section_stack.copy(), text))
    
    return segments


def extract_data(path: str, file_type: str) -> List[Segment]:
    """Extract from data files"""
    content = read_file_content(path)
    hierarchy = [file_type]
    
    extractors = {
        'CSV': lambda: [Segment(hierarchy, pd.read_csv(path).to_csv(index=False))],
        'JSON': lambda: [Segment(hierarchy, json.dumps(json.loads(content), indent=2, ensure_ascii=False))],
        'XML': lambda: [Segment(hierarchy, ET.tostring(ET.parse(path).getroot(), encoding='unicode'))],
        'YAML': lambda: [Segment(hierarchy, yaml.dump(yaml.safe_load(content), allow_unicode=True))],
    }
    
    return extractors.get(file_type, lambda: [Segment(hierarchy, content)])()


def extract_segments(path: str) -> List[Segment]:
    """Extract segments from file"""
    ext = get_file_extension(path).lower()
    
    extractors = {
        'pdf': extract_pdf,
        'docx': extract_docx,
        'doc': extract_docx,
        'txt': extract_text,
        'md': extract_text,
        'markdown': extract_text,
        'csv': lambda p: extract_data(p, 'CSV'),
        'json': lambda p: extract_data(p, 'JSON'),
        'xml': lambda p: extract_data(p, 'XML'),
        'yaml': lambda p: extract_data(p, 'YAML'),
        'yml': lambda p: extract_data(p, 'YAML'),
    }
    
    return extractors.get(ext, extract_text)(path)



class Chunker:
    """Smart chunker with optimization"""
    
    def __init__(self, config: ChunkConfig):
        self.config = config
        self.tokenizer = Tokenizer()
        self.sentence_break = re.compile(r'[.!?…]\s+|\n{2,}')
    
    def chunk_segments(self, segments: List[Segment], file_path: str) -> List[Chunk]:
        """Convert segments to chunks"""
        for seg in segments:
            seg.tokens = self.tokenizer.count(seg.content)
        if self.config.merge_small_segments:
            segments = self._merge_small(segments)
        segments = self._split_large(segments)
        
        return self._pack_with_overlap(segments, file_path)
    
    def _merge_small(self, segments: List[Segment]) -> List[Segment]:
        """Merge consecutive small segments"""
        merged = []
        buffer = []
        buffer_tokens = 0
        
        for seg in segments:
            if seg.tokens < 50 and buffer_tokens + seg.tokens < self.config.max_tokens:
                buffer.append(seg)
                buffer_tokens += seg.tokens
            else:
                if buffer:
                    if len(buffer) == 1:
                        merged.append(buffer[0])
                    else:
                        merged_seg = Segment(buffer[0].hierarchy, "\n".join(s.content for s in buffer))
                        merged_seg.tokens = buffer_tokens
                        merged.append(merged_seg)
                
                buffer = [seg]
                buffer_tokens = seg.tokens
        
        if buffer:
            merged.append(buffer[0] if len(buffer) == 1 else 
                         Segment(buffer[0].hierarchy, "\n".join(s.content for s in buffer)))
        
        return merged
    
    def _split_large(self, segments: List[Segment]) -> List[Segment]:
        """Split large segments"""
        result = []
        
        for seg in segments:
            if seg.tokens <= self.config.max_tokens:
                result.append(seg)
            else:
                parts = self._soft_split(seg.content)
                for i, part in enumerate(parts):
                    new_seg = Segment(seg.hierarchy + [f"Part {i+1}/{len(parts)}"], part)
                    new_seg.tokens = self.tokenizer.count(part)
                    result.append(new_seg)
        
        return result
    
    def _soft_split(self, text: str) -> List[str]:
        """Split text at sentence boundaries"""
        ids = self.tokenizer.encode(text)
        if len(ids) <= self.config.max_tokens:
            return [text]
        
        parts = []
        start = 0
        
        while start < len(ids):
            end = min(start + self.config.max_tokens, len(ids))
            chunk = self.tokenizer.decode(ids[start:end])
            
            if end < len(ids):
                sentences = self.sentence_break.split(chunk)
                if len(sentences) > 1:
                    chunk = "".join(sentences[:-1])
            
            parts.append(chunk.strip())
            start += self.tokenizer.count(chunk)
        
        return parts
    
    def _pack_with_overlap(self, segments: List[Segment], file_path: str) -> List[Chunk]:
        """Pack segments into chunks with overlap"""
        chunks = []
        i = 0
        order = 0
        
        while i < len(segments):
            chunk = Chunk(order, file_path, get_file_extension(file_path).upper())
            
            while i < len(segments) and chunk.tokens + segments[i].tokens <= self.config.max_tokens:
                chunk.add_segment(segments[i])
                i += 1
            
            if not chunk.segments and i < len(segments):
                chunk.add_segment(segments[i])
                i += 1
            
            chunks.append(chunk)
            order += 1
            
            if i < len(segments) and self.config.overlap_tokens > 0:
                overlap = 0
                backtrack = 0
                for j in range(len(chunk.segments) - 1, -1, -1):
                    if overlap >= self.config.overlap_tokens:
                        break
                    overlap += chunk.segments[j].tokens
                    backtrack += 1
                i -= backtrack
        
        return chunks


def process_document_to_chunks(path: str, config: ChunkConfig = None, use_cache: bool = False) -> List[Dict[str, Any]]:
    """    
    Args:
        path: File path
        config: Chunking configuration
        use_cache: IGNORED - no caching in cleaned version
    
    Returns:
        List of chunk dictionaries
    """
    config = config or ChunkConfig()
    
    logger.info(f"🔄 Processing: {path}")
    
    try:
        segments = extract_segments(path)
        if not segments:
            logger.warning(f"No segments from: {path}")
            return []
        
        logger.info(f"📊 Extracted {len(segments)} segments")
        
        chunker = Chunker(config)
        chunks = chunker.chunk_segments(segments, path)
        
        logger.info(f"✅ Created {len(chunks)} chunks")
        
        result = [c.to_dict() for c in chunks]
        
        return result
    
    except Exception as e:
        logger.error(f"Failed: {path}: {e}", exc_info=True)
        raise

DocChunkConfig = ChunkConfig

def get_default_config_for_file(filepath: str) -> ChunkConfig:
    """Get default config by file type"""
    ext = get_file_extension(filepath).lower()
    configs = {
        'pdf': ChunkConfig(max_tokens=300, overlap_tokens=50),
        'docx': ChunkConfig(max_tokens=300, overlap_tokens=50),
        'txt': ChunkConfig(max_tokens=500, overlap_tokens=50),
        'md': ChunkConfig(max_tokens=400, overlap_tokens=40),
        'csv': ChunkConfig(max_tokens=200, overlap_tokens=20),
    }
    return configs.get(ext, ChunkConfig())

def normalize_hierarchy(hierarchy: Any) -> str:
    """Convert hierarchy to string"""
    return " > ".join(hierarchy) if isinstance(hierarchy, list) else str(hierarchy)

def ensure_hierarchy_list(hierarchy: Any) -> List[str]:
    """Convert hierarchy to list"""
    return hierarchy if isinstance(hierarchy, list) else [hierarchy]