# backend/core/chunking_v2.py
from __future__ import annotations
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
from pathlib import Path

from backend.utils.utils import logger
from backend.utils.file_utils import read_file_content, get_file_extension

try:
    import tiktoken
except ImportError:
    logger.error("Missing tiktoken. Install: pip install tiktoken")
    raise

# File processors
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

try:
    from pptx import Presentation
except ImportError:
    Presentation = None

import pandas as pd
import json
import xml.etree.ElementTree as ET
import yaml


# ==================== CONFIG ====================
@dataclass
class ChunkConfig:
    max_tokens: int = 300
    overlap_tokens: int = 50
    include_hierarchy: bool = True  # Add hierarchy to content
    merge_small_segments: bool = True  # Merge segments < 50 tokens


# ==================== TOKENIZER ====================
class Tokenizer:
    def __init__(self):
        self.enc = tiktoken.encoding_for_model("gpt-4o-mini")
    
    def encode(self, text: str) -> List[int]:
        return self.enc.encode(text)
    
    def decode(self, ids: List[int]) -> str:
        return self.enc.decode(ids)
    
    def count(self, text: str) -> int:
        return len(self.encode(text))


# ==================== SEGMENT ====================
class Segment:
    """Represents a document segment with hierarchy"""
    
    def __init__(self, hierarchy: List[str], content: str):
        self.hierarchy = hierarchy  # ["Page 1", "Section A"]
        self.content = content
        self.tokens = 0
    
    def full_hierarchy(self) -> str:
        """Return full hierarchy path"""
        return " > ".join(self.hierarchy)
    
    def __repr__(self):
        return f"Segment({self.full_hierarchy()}, {self.tokens} tokens)"


# ==================== CHUNK ====================
class Chunk:
    """Represents a final chunk with multiple segments"""
    
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
    
    def get_hierarchies(self) -> List[str]:
        """Get all unique hierarchies"""
        return list(dict.fromkeys([s.full_hierarchy() for s in self.segments]))
    
    def get_content(self, include_hierarchy: bool = True) -> str:
        """Build content with optional hierarchy breadcrumbs"""
        if not include_hierarchy:
            return "\n\n".join(s.content for s in self.segments)
        
        parts = []
        for seg in self.segments:
            breadcrumb = f"**[{seg.full_hierarchy()}]**"
            parts.append(f"{breadcrumb}\n{seg.content}")
        
        return "\n\n".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.get_content(include_hierarchy=True),
            "tokens": self.tokens,
            "order": self.order,
            "hierarchy": self.get_hierarchies(),
            "hierarchy_list": self.get_hierarchies(),
            "file_path": self.file_path,
            "file_type": self.file_type,
        }


# ==================== SOFT SPLIT ====================
SENTENCE_BREAK = re.compile(r'[.!?…]\s+|\n{2,}')

def soft_split(text: str, tokenizer: Tokenizer, max_tokens: int) -> List[str]:
    """Split text at sentence boundaries"""
    ids = tokenizer.encode(text)
    if len(ids) <= max_tokens:
        return [text]
    
    parts = []
    start = 0
    
    while start < len(ids):
        end = min(start + max_tokens, len(ids))
        chunk_text = tokenizer.decode(ids[start:end])
        
        # Find last sentence boundary
        if end < len(ids):
            sentences = SENTENCE_BREAK.split(chunk_text)
            if len(sentences) > 1:
                # Keep all but last incomplete sentence
                chunk_text = "".join(sentences[:-1])
        
        parts.append(chunk_text.strip())
        start += tokenizer.count(chunk_text)
    
    return parts


# ==================== EXTRACTORS ====================

def extract_pdf(path: str) -> List[Segment]:
    """Extract segments from PDF"""
    if not pdfplumber:
        logger.error("pdfplumber not available")
        return []
    
    segments = []
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            hierarchy = [f"Page {page_num}"]
            
            # Text
            text = page.extract_text() or ""
            if text.strip():
                segments.append(Segment(hierarchy, text))
            
            # Tables
            for i, table in enumerate(page.extract_tables() or []):
                if not table:
                    continue
                
                df = pd.DataFrame(table[1:], columns=table[0]) if len(table) > 1 else pd.DataFrame(table)
                table_text = df.to_string(index=False)
                table_hierarchy = hierarchy + [f"Table {i+1}"]
                segments.append(Segment(table_hierarchy, table_text))
    
    return segments


def extract_docx(path: str) -> List[Segment]:
    """Extract segments from DOCX"""
    if not DocxDocument:
        logger.error("python-docx not available")
        return []
    
    segments = []
    doc = DocxDocument(path)
    
    # Track current section hierarchy
    section_stack = ["Document"]
    
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        
        # Handle headings
        if para.style.name.startswith('Heading'):
            level = int(para.style.name.replace('Heading', '').strip() or '1')
            
            # Update section stack
            section_stack = section_stack[:level]
            section_stack.append(text)
            
            # Add heading as segment
            segments.append(Segment(section_stack.copy(), text))
        else:
            # Paragraph belongs to current section
            segments.append(Segment(section_stack.copy(), text))
    
    # Tables
    for i, table in enumerate(doc.tables):
        df = pd.DataFrame([[cell.text for cell in row.cells] for row in table.rows])
        table_text = df.to_string(index=False)
        table_hierarchy = ["Document", f"Table {i+1}"]
        segments.append(Segment(table_hierarchy, table_text))
    
    return segments


def extract_text(path: str) -> List[Segment]:
    """Extract segments from text/markdown"""
    content = read_file_content(path)
    segments = []
    
    # Track current section hierarchy
    section_stack = ["Document"]
    
    for line in content.splitlines():
        text = line.strip()
        if not text:
            continue
        
        # Handle markdown headings
        if text.startswith('#'):
            level = len(text) - len(text.lstrip('#'))
            heading_text = text.lstrip('#').strip()
            
            # Update section stack
            section_stack = section_stack[:level]
            section_stack.append(heading_text)
            
            # Add heading as segment
            segments.append(Segment(section_stack.copy(), heading_text))
        else:
            # Line belongs to current section
            segments.append(Segment(section_stack.copy(), text))
    
    return segments


def extract_data(path: str, file_type: str) -> List[Segment]:
    """Extract segments from data files"""
    content = read_file_content(path)
    hierarchy = [file_type]
    
    if file_type == 'CSV':
        df = pd.read_csv(path)
        return [Segment(hierarchy, df.to_csv(index=False))]
    elif file_type == 'JSON':
        data = json.loads(content)
        return [Segment(hierarchy, json.dumps(data, indent=2, ensure_ascii=False))]
    elif file_type == 'XML':
        tree = ET.parse(path)
        return [Segment(hierarchy, ET.tostring(tree.getroot(), encoding='unicode'))]
    elif file_type in ('YAML', 'YML'):
        data = yaml.safe_load(content)
        return [Segment(hierarchy, yaml.dump(data, allow_unicode=True))]
    else:
        return [Segment(hierarchy, content)]


# ==================== MAIN EXTRACTOR ====================
def extract_segments(path: str) -> List[Segment]:
    """Extract segments from any file type"""
    ext = get_file_extension(path).lower()
    
    # Map extensions to extractors
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
    
    extractor = extractors.get(ext, extract_text)
    return extractor(path)


# ==================== CHUNKER ====================
class Chunker:
    """Smart chunker with hierarchy preservation"""
    
    def __init__(self, config: ChunkConfig):
        self.config = config
        self.tokenizer = Tokenizer()
    
    def chunk_segments(self, segments: List[Segment], file_path: str) -> List[Chunk]:
        """Convert segments to chunks"""
        # Calculate tokens for all segments
        for seg in segments:
            seg.tokens = self.tokenizer.count(seg.content)
        
        # Merge small segments
        if self.config.merge_small_segments:
            segments = self._merge_small(segments)
        
        # Split large segments
        segments = self._split_large(segments)
        
        # Pack into chunks with overlap
        chunks = self._pack_with_overlap(segments, file_path)
        
        return chunks
    
    def _merge_small(self, segments: List[Segment]) -> List[Segment]:
        """Merge consecutive small segments"""
        merged = []
        buffer = []
        buffer_tokens = 0
        
        for seg in segments:
            # If segment is small, add to buffer
            if seg.tokens < 50 and buffer_tokens + seg.tokens < self.config.max_tokens:
                buffer.append(seg)
                buffer_tokens += seg.tokens
            else:
                # Flush buffer
                if buffer:
                    if len(buffer) == 1:
                        merged.append(buffer[0])
                    else:
                        # Merge buffer segments
                        merged_seg = Segment(
                            hierarchy=buffer[0].hierarchy,
                            content="\n".join(s.content for s in buffer)
                        )
                        merged_seg.tokens = buffer_tokens
                        merged.append(merged_seg)
                
                # Start new buffer
                buffer = [seg]
                buffer_tokens = seg.tokens
        
        # Flush remaining
        if buffer:
            if len(buffer) == 1:
                merged.append(buffer[0])
            else:
                merged_seg = Segment(
                    hierarchy=buffer[0].hierarchy,
                    content="\n".join(s.content for s in buffer)
                )
                merged_seg.tokens = buffer_tokens
                merged.append(merged_seg)
        
        return merged
    
    def _split_large(self, segments: List[Segment]) -> List[Segment]:
        """Split segments larger than max_tokens"""
        result = []
        
        for seg in segments:
            if seg.tokens <= self.config.max_tokens:
                result.append(seg)
            else:
                # Split with soft boundaries
                parts = soft_split(seg.content, self.tokenizer, self.config.max_tokens)
                
                for i, part in enumerate(parts):
                    new_seg = Segment(
                        hierarchy=seg.hierarchy + [f"Part {i+1}/{len(parts)}"],
                        content=part
                    )
                    new_seg.tokens = self.tokenizer.count(part)
                    result.append(new_seg)
        
        return result
    
    def _pack_with_overlap(self, segments: List[Segment], file_path: str) -> List[Chunk]:
        """Pack segments into chunks with overlap"""
        chunks = []
        chunk_order = 0
        
        i = 0
        while i < len(segments):
            chunk = Chunk(
                order=chunk_order,
                file_path=file_path,
                file_type=get_file_extension(file_path).upper()
            )
            
            # Add segments until max_tokens
            while i < len(segments) and chunk.tokens + segments[i].tokens <= self.config.max_tokens:
                chunk.add_segment(segments[i])
                i += 1
            
            # If chunk is empty, force add one segment (shouldn't happen after split)
            if not chunk.segments and i < len(segments):
                chunk.add_segment(segments[i])
                i += 1
            
            chunks.append(chunk)
            chunk_order += 1
            
            # Overlap: backtrack for next chunk
            if i < len(segments) and self.config.overlap_tokens > 0:
                overlap_tokens = 0
                backtrack = 0
                
                # Count back until we have enough overlap
                for j in range(len(chunk.segments) - 1, -1, -1):
                    if overlap_tokens >= self.config.overlap_tokens:
                        break
                    overlap_tokens += chunk.segments[j].tokens
                    backtrack += 1
                
                # Move index back
                i -= backtrack
        
        return chunks


# ==================== PUBLIC API ====================
def process_document_to_chunks(
    path: str,
    config: ChunkConfig = None
) -> List[Dict[str, Any]]:
    """
    Main entry point: process document to chunks
    
    Args:
        path: File path
        config: Chunking configuration
    
    Returns:
        List of chunk dicts with improved hierarchy tracking
    """
    config = config or ChunkConfig()
    
    logger.info(f"Processing: {path}")
    
    try:
        # Extract segments with hierarchy
        segments = extract_segments(path)
        
        if not segments:
            logger.warning(f"No segments extracted from: {path}")
            return []
        
        logger.info(f"Extracted {len(segments)} segments")
        
        # Chunk segments
        chunker = Chunker(config)
        chunks = chunker.chunk_segments(segments, path)
        
        logger.info(f"Created {len(chunks)} chunks")
        
        # Convert to dicts
        return [c.to_dict() for c in chunks]
    
    except Exception as e:
        logger.error(f"Failed to process {path}: {e}", exc_info=True)
        raise


# ==================== COMPATIBILITY ====================
# For backward compatibility with old code
DocChunkConfig = ChunkConfig


def get_default_config_for_file(filepath: str) -> ChunkConfig:
    """Get default config based on file type"""
    ext = get_file_extension(filepath).lower()
    
    configs = {
        'pdf': ChunkConfig(max_tokens=300, overlap_tokens=50),
        'docx': ChunkConfig(max_tokens=300, overlap_tokens=50),
        'txt': ChunkConfig(max_tokens=500, overlap_tokens=50),
        'md': ChunkConfig(max_tokens=400, overlap_tokens=40),
        'csv': ChunkConfig(max_tokens=200, overlap_tokens=20),
    }
    
    return configs.get(ext, ChunkConfig())


# ==================== HIERARCHY HELPERS ====================
def normalize_hierarchy(hierarchy: Any) -> str:
    """
    Convert hierarchy to string for backward compatibility
    
    Args:
        hierarchy: Can be string (v1) or list (v2)
    
    Returns:
        String representation
    """
    if isinstance(hierarchy, list):
        return " > ".join(hierarchy)
    return str(hierarchy)


def ensure_hierarchy_list(hierarchy: Any) -> List[str]:
    """
    Convert hierarchy to list format (v2)
    
    Args:
        hierarchy: Can be string (v1) or list (v2)
    
    Returns:
        List representation
    """
    if isinstance(hierarchy, str):
        return [hierarchy]
    return hierarchy