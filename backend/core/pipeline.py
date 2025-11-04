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
    """
    Pipeline xử lý tài liệu từ upload đến chunking, extraction, graph building và embedding
    """
    
    def __init__(self, user_id: str = "default", enable_advanced: bool = True):
        self.user_id = user_id
        
        # Setup directories
        self.base_dir = Path("backend/data") / user_id
        self.chunks_dir = self.base_dir / "chunks"
        self.graphs_dir = self.base_dir / "graphs"
        self.vectors_dir = self.base_dir / "vectors"
        self.extractions_dir = self.base_dir / "extractions"
        
        # Create directories
        for directory in [self.chunks_dir, self.graphs_dir, self.vectors_dir, self.extractions_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Advanced features flag
        self.enable_advanced = enable_advanced and ADVANCED_FEATURES_AVAILABLE
        
        # Global configuration
        self.global_config = {
            'entity_types': ['PERSON', 'ORGANIZATION', 'LOCATION', 'EVENT', 'PRODUCT', 'CONCEPT', 'TECHNOLOGY'],
            'chunk_size': 300,
            'chunk_overlap': 50,
            'enable_gleaning': False,           # ← THÊM
            'max_gleaning_iterations': 2,        # ← THÊM
            'enable_graph': True,
            'enable_embedding': True
        }
        
        # Current state
        self.current_doc_id = None
        self.knowledge_graph = None
        self.vector_db = None
        
    def process_uploaded_file(self, uploaded_file, chunk_config: Optional[DocChunkConfig] = None,
                             enable_extraction: bool = True,
                             enable_graph: bool = True,
                             enable_embedding: bool = True,
                             enable_gleaning: bool = False) -> Dict[str, Any]:
        try:
            # Step 1: Save uploaded file
            filepath = save_uploaded_file(uploaded_file, user_id=self.user_id)
            doc_id = Path(uploaded_file.name).stem
            self.current_doc_id = doc_id
            
            logger.info(f"[Pipeline] Processing document: {doc_id}")
            
            # Step 2: Chunking
            logger.info("[Pipeline] Step 1: Chunking...")
            config = chunk_config or DocChunkConfig(
                max_token_size=self.global_config['chunk_size'],
                overlap_token_size=self.global_config['chunk_overlap']
            )
            
            chunks = process_document_to_chunks(filepath, config=config)
            chunk_filename = self._save_chunks(filepath, chunks, doc_id, uploaded_file.name)  # ← truyền tên gốc
            
            result = {
                'success': True,
                'filepath': filepath,
                'filename': uploaded_file.name,
                'doc_id': doc_id,
                'chunks_count': len(chunks),
                'chunks_file': chunk_filename,
                'total_tokens': sum(c['tokens'] for c in chunks),
                'processed_at': datetime.now().isoformat()
            }
            
            # Step 3-6: Advanced processing nếu được bật
            if self.enable_advanced:
                advanced_result = self._process_advanced_pipeline(
                    chunks=chunks,
                    doc_id=doc_id,
                    enable_extraction=enable_extraction,
                    enable_graph=enable_graph,
                    enable_embedding=enable_embedding,
                    enable_gleaning=enable_gleaning
                )
                result.update(advanced_result)
            else:
                logger.warning("[Pipeline] Advanced features disabled")
                result['advanced_processing'] = False
            
            logger.info(f"[Pipeline] Completed processing: {doc_id}")
            return result
            
        except Exception as e:
            logger.error(f"[Pipeline] Error: {str(e)}", exc_info=True)
            return {
                'success': False,
                'error': str(e)
            }

    def _save_chunks(self, filepath: str, chunks: List[Dict], doc_id: str, original_filename: str) -> str:
        """Lưu chunks + metadata"""
        safe_name = doc_id
        chunk_file = self.chunks_dir / f"{safe_name}_chunks.json"
        
        data = {
            'source_file': original_filename,  # ← tên gốc
            'source_path': filepath,
            'chunk_count': len(chunks),
            'total_tokens': sum(c['tokens'] for c in chunks),
            'processed_at': datetime.now().isoformat(),
            'chunks': chunks
        }
        
        with open(chunk_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[Pipeline] Saved chunks: {chunk_file}")
        return str(chunk_file)

    def _process_advanced_pipeline(self, chunks, doc_id, enable_extraction, enable_graph, enable_embedding, enable_gleaning):
        result = {}
        
        if enable_extraction:
            logger.info("[Pipeline] Step 2: Extraction...")
            entities_dict, relationships_dict = extract_entities_relations(chunks, self.global_config)
            extraction_file = self.extractions_dir / f"{doc_id}_extraction.json"
            with open(extraction_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'entities': entities_dict,
                    'relationships': relationships_dict,
                    'total_entities': sum(len(v) for v in entities_dict.values()),
                    'total_relationships': sum(len(v) for v in relationships_dict.values())
                }, f, ensure_ascii=False, indent=2)
            result.update({
                'entities_count': sum(len(v) for v in entities_dict.values()),
                'relationships_count': sum(len(v) for v in relationships_dict.values()),
                'extraction_file': str(extraction_file)
            })
        else:
            entities_dict, relationships_dict = {}, {}

        # Gleaning
        if enable_gleaning and self.global_config.get('enable_gleaning', False):
            logger.info("[Pipeline] Step 2.5: Gleaning...")
            from backend.core.graph_builder import gleaning_process
            import asyncio
            entities_dict, relationships_dict = asyncio.get_event_loop().run_until_complete(
                gleaning_process(entities_dict, relationships_dict, chunks, KnowledgeGraph(), self.global_config['max_gleaning_iterations'])
            )

        # Graph
        if enable_graph:
            logger.info("[Pipeline] Step 3: Graph Building...")
            kg = build_knowledge_graph(entities_dict, relationships_dict)
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

        # Embedding
        if enable_embedding:
            logger.info("[Pipeline] Step 4: Embedding...")
            chunk_embeds = generate_embeddings(chunks)
            entity_embeds = generate_entity_embeddings(entities_dict, kg if enable_graph else None)
            rel_embeds = generate_relationship_embeddings(relationships_dict)
            
            vector_db = VectorDatabase(
                db_path=str(self.vectors_dir / f"{doc_id}.index"),
                metadata_path=str(self.vectors_dir / f"{doc_id}_meta.json"),
                dim=384
            )
            vector_db.add_embeddings(chunk_embeds + entity_embeds + rel_embeds)
            vector_db.save()
            
            result.update({
                'total_embeddings': len(chunk_embeds) + len(entity_embeds) + len(rel_embeds),
                'vector_db_path': str(self.vectors_dir / f"{doc_id}.index")
            })

        return result

    def load_chunks(self, chunk_file: str) -> Dict:
        with  open(chunk_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_processed_docs(self) -> List[Dict]:
        docs = []
        for chunk_file in self.chunks_dir.glob("*_chunks.json"):
            try:
                data = self.load_chunks(str(chunk_file))
                doc_name = data.get('source_file', chunk_file.stem)
                doc_info = {
                    'file': doc_name,
                    'chunks': data.get('chunk_count', 0),
                    'tokens': data.get('total_tokens', 0),
                    'time': datetime.fromtimestamp(chunk_file.stat().mtime).strftime("%m/%d %H:%M"),
                    'has_graph': False,
                    'has_embeddings': False
                }
                doc_id = Path(doc_name).stem
                if (self.graphs_dir / f"{doc_id}_graph.json").exists():
                    doc_info['has_graph'] = True
                if (self.vectors_dir / f"{doc_id}.index").exists():
                    doc_info['has_embeddings'] = True
                docs.append(doc_info)
            except Exception as e:
                logger.error(f"Error reading chunk file {chunk_file}: {e}")
                continue
        return docs

    def delete_document(self, doc_id: str) -> bool:
        try:
            deleted_files = []
            for chunk_file in self.chunks_dir.glob(f"{doc_id}*_chunks.json"):
                chunk_file.unlink()
                deleted_files.append(str(chunk_file))
            graph_file = self.graphs_dir / f"{doc_id}_graph.json"
            if graph_file.exists():
                graph_file.unlink()
                deleted_files.append(str(graph_file))
            extraction_file = self.extractions_dir / f"{doc_id}_extraction.json"
            if extraction_file.exists():
                extraction_file.unlink()
                deleted_files.append(str(extraction_file))
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
        stats = {'doc_id': doc_id, 'exists': False}
        try:
            chunk_files = list(self.chunks_dir.glob(f"{doc_id}*_chunks.json"))
            if chunk_files:
                data = self.load_chunks(str(chunk_files[0]))
                stats['chunks'] = {
                    'count': data.get('chunk_count', 0),
                    'total_tokens': data.get('total_tokens', 0)
                }
                stats['exists'] = True
            extraction_file = self.extractions_dir / f"{doc_id}_extraction.json"
            if extraction_file.exists():
                with open(extraction_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                stats['extraction'] = {
                    'entities_count': data.get('total_entities', 0),
                    'relationships_count': data.get('total_relationships', 0)
                }
            if (self.graphs_dir / f"{doc_id}_graph.json").exists():
                stats['has_graph'] = True
            if (self.vectors_dir / f"{doc_id}.index").exists():
                stats['has_embeddings'] = True
            return stats
        except Exception as e:
            logger.error(f"Error getting statistics for {doc_id}: {e}")
            return stats


def process_document(filepath: str, config: Optional[DocChunkConfig] = None, 
                    user_id: str = "default",
                    enable_advanced: bool = False) -> Dict[str, Any]:
    try:
        cfg = config or DocChunkConfig(max_token_size=300, overlap_token_size=50)
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