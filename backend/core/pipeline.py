# backend/core/pipeline.py
"""
✅ OPTIMIZED: Document Pipeline with Parallel Processing
- Concurrent extraction (8 parallel LLM calls)
- Batch embedding generation
- Efficient memory usage
- Progress tracking
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import sys
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.core.chunking import process_document_to_chunks, DocChunkConfig
from backend.utils.file_utils import save_uploaded_file

try:
    from backend.core.extraction import extract_entities_relations
    from backend.core.graph_builder import build_knowledge_graph
    from backend.core.embedding import generate_embeddings, generate_entity_embeddings
    ADVANCED_FEATURES = True
except ImportError as e:
    logging.warning(f"⚠️ Advanced features not available: {e}")
    ADVANCED_FEATURES = False

logger = logging.getLogger(__name__)


class DocumentPipeline:
    """✅ OPTIMIZED: High-performance document processing pipeline"""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        
        # Upload directory
        self.upload_dir = Path("backend/data") / user_id / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # ✅ OPTIMIZED: Configuration
        self.config = {
            'entity_types': ['PERSON', 'ORGANIZATION', 'LOCATION', 'EVENT', 
                           'PRODUCT', 'CONCEPT', 'TECHNOLOGY'],
            'max_concurrent_llm': int(os.getenv('MAX_CONCURRENT_LLM_CALLS', 16)),  # Increased from 8
            'extraction_batch_size': int(os.getenv('EXTRACTION_BATCH_SIZE', 20)),  # Increased from 10
            'embedding_batch_size': int(os.getenv('EMBEDDING_BATCH_SIZE', 128)),   # Increased from 64
            'enable_parallel': True,
        }
        
        logger.info(f"✅ Pipeline initialized for user: {user_id} (parallel={self.config['enable_parallel']})")
    
    def process_file(self, 
                    uploaded_file, 
                    chunk_config: Optional[DocChunkConfig] = None,
                    enable_extraction: bool = True,
                    enable_graph: bool = True,
                    enable_embedding: bool = True,
                    progress_callback: Optional[Callable[[str, float], None]] = None) -> Dict[str, Any]:
        """
        ✅ OPTIMIZED: Process uploaded file with progress tracking
        
        Args:
            uploaded_file: Streamlit uploaded file
            chunk_config: Chunking configuration
            enable_extraction: Enable entity extraction
            enable_graph: Enable graph building
            enable_embedding: Enable embedding generation
            progress_callback: Optional callback(message, progress)
        
        Returns:
            Result dict WITHOUT saving to disk (caller handles storage)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in uploaded_file.name if c.isalnum() or c in "._- ")
        doc_id = f"{timestamp}_{safe_name}"
        
        result = {
            'success': False,
            'filename': uploaded_file.name,
            'doc_id': doc_id,
            'error': None
        }
        
        def update_progress(msg: str, pct: float):
            if progress_callback:
                progress_callback(msg, pct)
            logger.info(f"[{pct:.0f}%] {msg}")
        
        try:
            # 1. Save uploaded file (5%)
            update_progress("Saving file...", 5)
            filepath = save_uploaded_file(uploaded_file, user_id=self.user_id)
            result['filepath'] = filepath
            
            # 2. Chunking (20%)
            update_progress("Chunking document...", 20)
            config = chunk_config or DocChunkConfig(max_tokens=300, overlap_tokens=50)
            chunks = process_document_to_chunks(filepath, config=config, use_cache=False)
            
            if not chunks:
                result['error'] = 'No chunks generated'
                return result
            
            result['chunks'] = chunks
            result['stats'] = {
                'chunks_count': len(chunks),
                'total_tokens': sum(c['tokens'] for c in chunks)
            }
            
            # 3. Entity Extraction (40%)
            if enable_extraction and ADVANCED_FEATURES:
                update_progress(f"Extracting entities ({self.config['max_concurrent_llm']} parallel)...", 40)
                try:
                    entities, relationships = extract_entities_relations(
                        chunks, 
                        {**self.config, 'max_concurrent': self.config['max_concurrent_llm']}
                    )
                    
                    result['entities'] = entities
                    result['relationships'] = relationships
                    result['stats']['entities_count'] = sum(len(v) for v in entities.values())
                    result['stats']['relationships_count'] = sum(len(v) for v in relationships.values())
                except Exception as e:
                    logger.error(f"❌ Extraction failed: {e}")
                    result['stats']['entities_count'] = 0
                    result['stats']['relationships_count'] = 0
            
            # 4. Knowledge Graph (60%)
            if enable_graph and ADVANCED_FEATURES and result.get('entities'):
                update_progress("Building knowledge graph...", 60)
                try:
                    kg = build_knowledge_graph(
                        result['entities'], 
                        result['relationships'],
                        enable_summarization=False  # Disable for speed
                    )
                    
                    result['graph'] = kg.to_dict()
                    result['stats']['graph_nodes'] = kg.G.number_of_nodes()
                    result['stats']['graph_edges'] = kg.G.number_of_edges()
                except Exception as e:
                    logger.error(f"❌ Graph building failed: {e}")
                    result['stats']['graph_nodes'] = 0
                    result['stats']['graph_edges'] = 0
            
            # 5. Generate Embeddings (80%)
            if enable_embedding and ADVANCED_FEATURES:
                update_progress(f"Generating embeddings (batch={self.config['embedding_batch_size']})...", 80)
                try:
                    embeddings = []
                    
                    # Chunk embeddings
                    chunk_embeds = generate_embeddings(
                        chunks, 
                        batch_size=self.config['embedding_batch_size'],
                        use_cache=False
                    )
                    embeddings.extend(chunk_embeds)
                    
                    # Entity embeddings (if graph exists)
                    if result.get('entities') and result.get('graph'):
                        from backend.core.graph_builder import KnowledgeGraph
                        kg = KnowledgeGraph()
                        
                        for node in result['graph']['nodes']:
                            kg.add_entity(
                                entity_name=node['id'],
                                entity_type=node.get('type', 'UNKNOWN'),
                                description=node.get('description', ''),
                                source_id='',
                                source_document=''
                            )
                        
                        entity_embeds = generate_entity_embeddings(
                            result['entities'], 
                            kg,
                            batch_size=self.config['embedding_batch_size']
                        )
                        embeddings.extend(entity_embeds)
                    
                    result['embeddings'] = embeddings
                    result['stats']['embeddings_count'] = len(embeddings)
                except Exception as e:
                    logger.error(f"❌ Embedding generation failed: {e}")
                    result['stats']['embeddings_count'] = 0
            
            # 6. Complete (100%)
            update_progress("Processing complete!", 100)
            result['success'] = True
            logger.info(f"✅ Processed: {uploaded_file.name} - {result['stats']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {uploaded_file.name}: {e}", exc_info=True)
            result['error'] = str(e)
        
        return result
    
    def process_multiple_files_parallel(self, 
                                       uploaded_files: List,
                                       max_parallel: int = 3,
                                       **kwargs) -> List[Dict[str, Any]]:
        """
        ✅ NEW: Process multiple files in parallel
        
        Args:
            uploaded_files: List of uploaded files
            max_parallel: Max files to process concurrently
            **kwargs: Arguments for process_file
        
        Returns:
            List of results
        """
        results = []
        
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = []
            for file in uploaded_files:
                future = executor.submit(self.process_file, file, **kwargs)
                futures.append((file.name, future))
            
            for filename, future in futures:
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"✅ Completed: {filename}")
                except Exception as e:
                    logger.error(f"❌ Failed: {filename}: {e}")
                    results.append({
                        'success': False,
                        'filename': filename,
                        'error': str(e)
                    })
        
        success_count = sum(1 for r in results if r['success'])
        logger.info(f"✅ Parallel processing complete: {success_count}/{len(results)} successful")
        
        return results
    
    def process_multiple_files_sequential(self, 
                                         uploaded_files: List, 
                                         **kwargs) -> List[Dict[str, Any]]:
        """Process multiple files sequentially (safer for memory)"""
        results = []
        
        for i, file in enumerate(uploaded_files, 1):
            logger.info(f"Processing [{i}/{len(uploaded_files)}]: {file.name}")
            result = self.process_file(file, **kwargs)
            results.append(result)
        
        success_count = sum(1 for r in results if r['success'])
        logger.info(f"Completed: {success_count}/{len(results)} successful")
        
        return results


# ✅ Standalone function for backward compatibility
def process_document(filepath: str, 
                    config: Optional[DocChunkConfig] = None,
                    enable_extraction: bool = False,
                    enable_graph: bool = False,
                    enable_embedding: bool = False) -> Dict[str, Any]:
    """
    Process document from filepath (for testing/scripts)
    """
    try:
        # Chunking
        cfg = config or DocChunkConfig(max_tokens=300, overlap_tokens=50)
        chunks = process_document_to_chunks(filepath, config=cfg, use_cache=False)
        
        result = {
            'success': True,
            'filepath': filepath,
            'chunks': chunks,
            'stats': {
                'chunks_count': len(chunks),
                'total_tokens': sum(c['tokens'] for c in chunks)
            }
        }
        
        # Extraction
        if enable_extraction and ADVANCED_FEATURES:
            entities, relationships = extract_entities_relations(chunks, {
                'entity_types': ['PERSON', 'ORGANIZATION', 'LOCATION', 'EVENT', 
                               'PRODUCT', 'CONCEPT', 'TECHNOLOGY']
            })
            result['entities'] = entities
            result['relationships'] = relationships
        
        # Graph
        if enable_graph and ADVANCED_FEATURES and result.get('entities'):
            kg = build_knowledge_graph(result['entities'], result['relationships'])
            result['graph'] = kg.to_dict()
        
        # Embeddings
        if enable_embedding and ADVANCED_FEATURES:
            result['embeddings'] = generate_embeddings(chunks, use_cache=False)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to process {filepath}: {e}")
        return {
            'success': False,
            'filepath': filepath,
            'error': str(e)
        }