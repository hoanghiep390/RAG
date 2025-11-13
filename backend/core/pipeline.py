# backend/core/pipeline.py

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys
import os

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
    """Document processing pipeline"""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        
        # Upload directory
        self.upload_dir = Path("backend/data") / user_id / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.config = {
            'entity_types': ['PERSON', 'ORGANIZATION', 'LOCATION', 'EVENT', 
                           'PRODUCT', 'CONCEPT', 'TECHNOLOGY'],
            'max_concurrent_llm': int(os.getenv('MAX_CONCURRENT_LLM_CALLS', 8)),
            'extraction_batch_size': int(os.getenv('EXTRACTION_BATCH_SIZE', 10)),
            'embedding_batch_size': int(os.getenv('EMBEDDING_BATCH_SIZE', 64)),
        }
        
        logger.info(f"✅ Pipeline initialized for user: {user_id}")
    
    def process_file(self, uploaded_file, 
                    chunk_config: Optional[DocChunkConfig] = None,
                    enable_extraction: bool = True,
                    enable_graph: bool = True,
                    enable_embedding: bool = True) -> Dict[str, Any]:
        """
        Process uploaded file through complete pipeline
        
        Returns result dict WITHOUT saving to disk (caller handles MongoDB/FAISS)
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
        
        try:
            # 1. Save uploaded file
            filepath = save_uploaded_file(uploaded_file, user_id=self.user_id)
            result['filepath'] = filepath
            
            # 2. Chunking
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
            
            # 3. Entity Extraction (optional)
            if enable_extraction and ADVANCED_FEATURES:
                try:
                    entities, relationships = extract_entities_relations(chunks, self.config)
                    
                    result['entities'] = entities
                    result['relationships'] = relationships
                    result['stats']['entities_count'] = sum(len(v) for v in entities.values())
                    result['stats']['relationships_count'] = sum(len(v) for v in relationships.values())
                except Exception as e:
                    logger.error(f"❌ Extraction failed: {e}")
                    result['stats']['entities_count'] = 0
                    result['stats']['relationships_count'] = 0
            
            # 4. Knowledge Graph (optional)
            if enable_graph and ADVANCED_FEATURES and result.get('entities'):
                try:
                    kg = build_knowledge_graph(
                        result['entities'], 
                        result['relationships'],
                        enable_summarization=False
                    )
                    
                    result['graph'] = kg.to_dict()
                    result['stats']['graph_nodes'] = kg.G.number_of_nodes()
                    result['stats']['graph_edges'] = kg.G.number_of_edges()
                except Exception as e:
                    logger.error(f"❌ Graph building failed: {e}")
                    result['stats']['graph_nodes'] = 0
                    result['stats']['graph_edges'] = 0
            
            # 5. Generate Embeddings (optional)
            if enable_embedding and ADVANCED_FEATURES:
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
            
            result['success'] = True
            logger.info(f"✅ Processed: {uploaded_file.name} - {result['stats']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {uploaded_file.name}: {e}", exc_info=True)
            result['error'] = str(e)
        
        return result
    
    def process_multiple_files(self, uploaded_files: List, **kwargs) -> List[Dict[str, Any]]:
        """Process multiple files sequentially"""
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