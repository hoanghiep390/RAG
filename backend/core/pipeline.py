# backend/core/pipeline.py
"""
Pipeline xử lý tài liệu: Upload → Chunking → Extraction → Graph Building → Embedding → Storage
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import sys
import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.core.chunking import process_document_to_chunks, DocChunkConfig
from backend.utils.file_utils import save_uploaded_file

# Import các modules mới
try:
    from backend.core.extraction import extract_entities_relations
    from backend.core.graph_builder import build_knowledge_graph, KnowledgeGraph
    from backend.core.embedding import (
        VectorDatabase,
        generate_embeddings,
        generate_entity_embeddings,
        generate_relationship_embeddings,
        search_similar
    )
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Advanced features not available: {e}")
    ADVANCED_FEATURES_AVAILABLE = False

logger = logging.getLogger(__name__)

class DocumentPipeline:
    """Pipeline với optimizations"""

    def __init__(self, user_id: str = "default", enable_advanced: bool = True):
        self.user_id = user_id
        self.base_dir = Path("backend/data").absolute() / user_id
        self.chunks_dir = self.base_dir / "chunks"
        self.graphs_dir = self.base_dir / "graphs"
        self.vectors_dir = self.base_dir / "vectors"
        self.extractions_dir = self.base_dir / "extractions"
        
        for directory in [self.chunks_dir, self.graphs_dir, self.vectors_dir, self.extractions_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.enable_advanced = enable_advanced and ADVANCED_FEATURES_AVAILABLE
        
        self.global_config = {
            'entity_types': ['PERSON', 'ORGANIZATION', 'LOCATION', 'EVENT', 'PRODUCT', 'CONCEPT', 'TECHNOLOGY'],
            'chunk_size': 300,
            'chunk_overlap': 50,
            'enable_gleaning': False,
            'max_gleaning_iterations': 2,
            'enable_graph': True,
            'enable_embedding': True
        }
        self.max_workers = int(os.getenv('MAX_WORKERS', max(1, mp.cpu_count() - 1)))
        self.batch_size = int(os.getenv('EXTRACTION_BATCH_SIZE', 10))
        self.embedding_batch_size = int(os.getenv('EMBEDDING_BATCH_SIZE', 64))
        self.use_hnsw = os.getenv('USE_HNSW_INDEX', 'true').lower() == 'true'
        
        logger.info(f"Pipeline initialized:")
        logger.info(f"   User: {user_id}")
        logger.info(f"   Max workers: {self.max_workers}")
        logger.info(f"   Extraction batch: {self.batch_size}")
        logger.info(f"   Embedding batch: {self.embedding_batch_size}")
        logger.info(f"   HNSW index: {self.use_hnsw}")
        self.current_doc_id = None
        self.knowledge_graph = None
        self.vector_db = None
    
    def process_multiple_files(self, 
                              uploaded_files,
                              chunk_config: Optional[DocChunkConfig] = None,
                              **kwargs) -> List[Dict]:
        """
        Usage:
            results = pipeline.process_multiple_files(
                uploaded_files=[file1, file2, file3],
                chunk_config=config,
                enable_extraction=True,
                enable_graph=True,
                enable_embedding=True
            )
        """
        logger.info(f"Processing {len(uploaded_files)} files in parallel")
        
        results = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all files
            future_to_file = {
                executor.submit(
                    self.process_uploaded_file,
                    f, chunk_config, **kwargs
                ): f for f in uploaded_files
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_file):
                file = future_to_file[future]
                try:
                    result = future.result(timeout=int(os.getenv('FILE_PROCESSING_TIMEOUT', 300)))
                    results.append(result)
                    
                    if result.get('success'):
                        logger.info(f"[{len(results)}/{len(uploaded_files)}] {file.name}")
                    else:
                        logger.error(f"[{len(results)}/{len(uploaded_files)}] {file.name}: {result.get('error')}")
                
                except Exception as e:
                    logger.error(f"Failed: {file.name} - {e}")
                    results.append({
                        'success': False,
                        'filename': file.name,
                        'error': str(e)
                    })
        
        return results
    
    def process_uploaded_file(self, uploaded_file, chunk_config: Optional[DocChunkConfig] = None, **kwargs):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in uploaded_file.name if c.isalnum() or c in "._- ")
        doc_id = f"{timestamp}_{safe_name}"

        file_path_str = save_uploaded_file(uploaded_file, user_id=self.user_id)
        file_path = Path(file_path_str)  

        config = chunk_config or DocChunkConfig(max_tokens=300, overlap_tokens=50)
        
        try:
            chunks = process_document_to_chunks(file_path_str, config=config)
        except Exception as e:
            logger.error(f"Failed to chunk {file_path_str}: {e}")
            return {'success': False, 'error': str(e), 'filename': uploaded_file.name}

        if not chunks:
            return {'success': False, 'error': 'No chunks generated', 'filename': uploaded_file.name}

        chunk_filename = self._save_chunks(file_path, chunks, doc_id, uploaded_file.name)

        result = {
            'success': True,
            'filename': uploaded_file.name,
            'filepath': file_path_str,
            'chunks_count': len(chunks),
            'chunks_file': chunk_filename,
            'total_tokens': sum(c['tokens'] for c in chunks),
        }

        if self.enable_advanced:
            advanced_result = self._process_advanced_pipeline(
                chunks=chunks,
                doc_id=doc_id,
                **kwargs
            )
            result.update(advanced_result)

        return result

    def _save_chunks(self, filepath: Path, chunks: List[Dict], doc_id: str, original_name: str) -> str:
        chunk_file = self.chunks_dir / f"{doc_id}_chunks.json"
        data = {
            'source_file': original_name,
            'source_path': str(filepath),
            'chunk_count': len(chunks),
            'total_tokens': sum(c['tokens'] for c in chunks),
            'chunks': chunks
        }
        with open(chunk_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return str(chunk_file)

    def _process_advanced_pipeline(self, chunks, doc_id, enable_extraction, enable_graph, 
                                   enable_embedding, enable_gleaning):
        """
        OPTIMIZED: Use optimized batch sizes and HNSW
        """
        result = {}
        
        # Step 2: Extraction with batch processing
        entities_dict = {}
        relationships_dict = {}
        
        if enable_extraction:
            logger.info(f"[Pipeline] Step 2: Extraction (batch_size={self.batch_size})...")
            try:
                # Use batch extraction
                entities_dict, relationships_dict = extract_entities_relations(chunks, self.global_config)
                
                # === FIX: Convert tuple keys to string for JSON serialization ===
                safe_relationships = {
                    f"{src}|||{tgt}": rels 
                    for (src, tgt), rels in relationships_dict.items()
                }
                
                # Save results
                extraction_file = self.extractions_dir / f"{doc_id}_extraction.json"
                with open(extraction_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'entities': entities_dict,
                        'relationships': safe_relationships,
                        'total_entities': sum(len(v) for v in entities_dict.values()),
                        'total_relationships': sum(len(v) for v in relationships_dict.values())
                    }, f, ensure_ascii=False, indent=2)
                
                result.update({
                    'entities_count': sum(len(v) for v in entities_dict.values()),
                    'relationships_count': sum(len(v) for v in relationships_dict.values()),
                    'extraction_file': str(extraction_file)
                })
                
                logger.info(f"[Pipeline] Extracted {result['entities_count']} entities, "
                          f"{result['relationships_count']} relationships")
                
            except Exception as e:
                logger.error(f"[Pipeline] Extraction failed: {str(e)}")
                result['extraction_error'] = str(e)
        
        # Step 2.5: Gleaning (skip if not needed)
        if enable_gleaning and self.global_config.get('enable_gleaning', False):
            logger.info("[Pipeline] Step 2.5: Gleaning...")
            try:
                from backend.core.graph_builder import gleaning_process
                import asyncio
                
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        try:
                            import nest_asyncio
                            nest_asyncio.apply()
                        except ImportError:
                            logger.warning("[Pipeline] nest_asyncio not available, skipping gleaning")
                            enable_gleaning = False
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                if enable_gleaning:
                    entities_dict, relationships_dict = loop.run_until_complete(
                        gleaning_process(
                            entities_dict, 
                            relationships_dict, 
                            chunks, 
                            KnowledgeGraph(), 
                            self.global_config['max_gleaning_iterations']
                        )
                    )
                    
                    result['gleaning_applied'] = True
                    logger.info("[Pipeline] Gleaning completed")
                    
            except Exception as e:
                logger.error(f"[Pipeline] Gleaning failed: {str(e)}")
                result['gleaning_error'] = str(e)
        
        # Step 3: Graph Building
        kg = None  # Khởi tạo trước để tránh UnboundLocalError
        
        if enable_graph:
            logger.info("[Pipeline] Step 3: Graph Building...")
            try:
                # Use optimized graph building
                enable_summarization = os.getenv('ENABLE_GRAPH_SUMMARIZATION', 'false').lower() == 'true'
                kg = build_knowledge_graph(entities_dict, relationships_dict, 
                                          enable_summarization=enable_summarization)
                
                # Save graph
                graph_file = self.graphs_dir / f"{doc_id}_graph.json"
                with open(graph_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'graph': kg.to_dict(),
                        'statistics': kg.get_statistics(),
                        'metadata': {'source_file': doc_id}
                    }, f, ensure_ascii=False, indent=2)
                
                result.update({
                    'graph_nodes': kg.G.number_of_nodes(),
                    'graph_edges': kg.G.number_of_edges(),
                    'graph_file': str(graph_file)
                })
                
                logger.info(f"[Pipeline] Built graph: {result['graph_nodes']} nodes, "
                          f"{result['graph_edges']} edges")
                
            except Exception as e:
                logger.error(f"[Pipeline] Graph building failed: {str(e)}")
                result['graph_error'] = str(e)

        # Step 4: Embedding
        if enable_embedding:
            logger.info("[Pipeline] Step 4: Embedding...")
            self.vectors_dir.mkdir(parents=True, exist_ok=True) 
            logger.debug(f"Ensured vectors dir: {self.vectors_dir}")
            try:
                chunk_embeds = generate_embeddings(chunks)
                entity_embeds = generate_entity_embeddings(entities_dict, kg) if kg else []
                rel_embeds = generate_relationship_embeddings(relationships_dict)
                
                vector_db = VectorDatabase(
                    db_path=str(self.vectors_dir / f"{doc_id}.index"),
                    metadata_path=str(self.vectors_dir / f"{doc_id}_meta.json"),
                    dim=384
                )
                
                # Add all embeddings
                vector_db.add_embeddings(chunk_embeds)
                if entity_embeds:
                    vector_db.add_embeddings(entity_embeds)
                if rel_embeds:
                    vector_db.add_embeddings(rel_embeds)
                
                vector_db.save()
                
                result.update({
                    'total_embeddings': len(chunk_embeds) + len(entity_embeds) + len(rel_embeds),
                    'chunk_embeddings': len(chunk_embeds),
                    'entity_embeddings': len(entity_embeds),
                    'relationship_embeddings': len(rel_embeds),
                    'vector_db_path': str(self.vectors_dir / f"{doc_id}.index")
                })
                
                logger.info(f"[Pipeline] Generated {result['total_embeddings']} embeddings")
                
            except Exception as e:
                logger.error(f"[Pipeline] Embedding failed: {str(e)}")
                result['embedding_error'] = str(e)

        return result

    def load_chunks(self, chunk_file: str) -> Dict:
        """
        Load chunks từ file JSON
        
        Args:
            chunk_file: Đường dẫn file chunks
            
        Returns:
            Dict chứa chunks và metadata
        """
        with open(chunk_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_processed_docs(self) -> List[Dict]:
        """
        Lấy danh sách tài liệu đã xử lý
        
        Returns:
            List các dict thông tin document
        """
        docs = []
        
        for chunk_file in self.chunks_dir.glob("*_chunks.json"):
            try:
                data = self.load_chunks(str(chunk_file))
                doc_name = data.get('source_file', chunk_file.stem)
                
                doc_info = {
                    'file': doc_name,
                    'chunks': data.get('chunk_count', 0),
                    'tokens': data.get('total_tokens', 0),
                    'time': datetime.fromtimestamp(chunk_file.stat().st_mtime).strftime("%m/%d %H:%M"),
                    'has_graph': False,
                    'has_embeddings': False
                }
                
                # Check for graph
                doc_id = Path(doc_name).stem
                if (self.graphs_dir / f"{doc_id}_graph.json").exists():
                    doc_info['has_graph'] = True
                
                # Check for embeddings
                if (self.vectors_dir / f"{doc_id}.index").exists():
                    doc_info['has_embeddings'] = True
                
                docs.append(doc_info)
                
            except Exception as e:
                logger.error(f"Error reading chunk file {chunk_file}: {e}")
                continue
        
        return docs

    def delete_document(self, doc_id: str) -> bool:
        """
        Xóa tất cả dữ liệu liên quan đến document
        
        Args:
            doc_id: Document ID cần xóa
            
        Returns:
            True nếu xóa thành công
        """
        try:
            deleted_files = []
            
            # Delete chunks
            for chunk_file in self.chunks_dir.glob(f"{doc_id}*_chunks.json"):
                chunk_file.unlink()
                deleted_files.append(str(chunk_file))
            
            # Delete graph
            graph_file = self.graphs_dir / f"{doc_id}_graph.json"
            if graph_file.exists():
                graph_file.unlink()
                deleted_files.append(str(graph_file))
            
            # Delete extraction
            extraction_file = self.extractions_dir / f"{doc_id}_extraction.json"
            if extraction_file.exists():
                extraction_file.unlink()
                deleted_files.append(str(extraction_file))
            
            # Delete vector DB
            for pattern in [f"{doc_id}.index", f"{doc_id}_meta.json"]:
                file_path = self.vectors_dir / pattern
                if file_path.exists():
                    file_path.unlink()
                    deleted_files.append(str(file_path))
            
            logger.info(f"[Pipeline] Deleted document: {doc_id} ({len(deleted_files)} files)")
            return True
            
        except Exception as e:
            logger.error(f"[Pipeline] Failed to delete document {doc_id}: {str(e)}")
            return False

    def get_statistics(self, doc_id: str) -> Dict[str, Any]:
        """
        Lấy thống kê chi tiết về document
        
        Args:
            doc_id: Document ID
            
        Returns:
            Dict thống kê
        """
        stats = {'doc_id': doc_id, 'exists': False}
        
        try:
            # Check chunks
            chunk_files = list(self.chunks_dir.glob(f"{doc_id}*_chunks.json"))
            if chunk_files:
                data = self.load_chunks(str(chunk_files[0]))
                stats['chunks'] = {
                    'count': data.get('chunk_count', 0),
                    'total_tokens': data.get('total_tokens', 0)
                }
                stats['exists'] = True
            
            # Check extraction
            extraction_file = self.extractions_dir / f"{doc_id}_extraction.json"
            if extraction_file.exists():
                with open(extraction_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                stats['extraction'] = {
                    'entities_count': data.get('total_entities', 0),
                    'relationships_count': data.get('total_relationships', 0)
                }
            
            # Check graph
            if (self.graphs_dir / f"{doc_id}_graph.json").exists():
                stats['has_graph'] = True
            
            # Check embeddings
            if (self.vectors_dir / f"{doc_id}.index").exists():
                stats['has_embeddings'] = True
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics for {doc_id}: {e}")
            return stats


def process_document(filepath: str, config: Optional[DocChunkConfig] = None, 
                    user_id: str = "default",
                    enable_advanced: bool = False) -> Dict[str, Any]:
    """
    Process document (helper function cho command line usage)
    
    Args:
        filepath: Đường dẫn file
        config: Chunk config
        user_id: User ID
        enable_advanced: Bật advanced features
        
    Returns:
        Dict kết quả
    """
    try:
        cfg = config or DocChunkConfig(max_tokens=500, overlap_tokens=50)
        chunks = process_document_to_chunks(filepath, config=cfg)
        
        pipeline = DocumentPipeline(user_id=user_id, enable_advanced=enable_advanced)
        chunk_filename = pipeline._save_chunks(filepath, chunks, Path(filepath).stem, Path(filepath).name)
        
        result = {
            'success': True,
            'filepath': filepath,
            'chunks_count': len(chunks),
            'chunks_file': chunk_filename,
            'total_tokens': sum(c['tokens'] for c in chunks),
            'chunks': chunks
        }
        
        if enable_advanced and ADVANCED_FEATURES_AVAILABLE:
            doc_id = Path(filepath).stem
            advanced_result = pipeline._process_advanced_pipeline(
                chunks=chunks,
                doc_id=doc_id,
                enable_extraction=True,
                enable_graph=True,
                enable_embedding=True,
                enable_gleaning=False
            )
            result.update(advanced_result)
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'filepath': filepath
        }