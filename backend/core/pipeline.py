# backend/core/pipeline.py 

from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from backend.core.chunking import process_document_to_chunks, ChunkConfig
from backend.core.extraction import extract_entities_relations
from backend.core.graph_builder import build_knowledge_graph
from backend.core.embedding import generate_embeddings, generate_entity_embeddings
from backend.utils.file_utils import save_uploaded_file

class DocumentPipeline:
    
    def __init__(self, user_id: str = "default", vector_db=None, mongo_storage=None):
        """
        Args:
            user_id: User ID for data isolation
            vector_db: VectorDatabase instance (optional, for auto-save)
            mongo_storage: MongoStorage instance (optional, for auto-save)
        """
        self.user_id = user_id
        self.upload_dir = Path("backend/data") / user_id / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        
        self.vector_db = vector_db
        self.mongo_storage = mongo_storage
    
    def process_file(
        self, 
        uploaded_file, 
        chunk_config: ChunkConfig = None,
        enable_extraction: bool = True, 
        enable_graph: bool = True,
        enable_embedding: bool = True,
        auto_save: bool = True,  
        progress_callback: Callable = None
    ) -> Dict[str, Any]:
        """
        Process 1 file and optionally auto-save to databases
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            chunk_config: Chunking configuration
            enable_extraction: Extract entities/relationships
            enable_graph: Build knowledge graph
            enable_embedding: Generate embeddings
            auto_save: Auto-save to MongoDB + VectorDB 
            progress_callback: Progress callback function
                    
        Returns:
            Result dict with success status and stats
        """
        
        def update(msg: str, pct: float):
            if progress_callback:
                progress_callback(msg, pct)
        
        result = {
            'success': False,
            'filename': uploaded_file.name,
            'doc_id': f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_file.name}"
        }
        
        try:
            # 1. Save file (5%)
            update("Saving file...", 5)
            filepath = save_uploaded_file(uploaded_file, self.user_id)
            result['filepath'] = str(filepath)

            # 2. Chunk (20%)
            update("Chunking...", 20)
            config = chunk_config or ChunkConfig()
            chunks = process_document_to_chunks(filepath, config)

            if not chunks:
                result['error'] = 'No chunks generated from document'
                return result

            result['chunks'] = chunks
            result['stats'] = {
                'chunks_count': len(chunks),
                'entities_count': 0,
                'relationships_count': 0,
                'graph_nodes': 0,
                'graph_edges': 0,
                'embeddings_count': 0
            }

            # 3. Extract entities & relations (40%)
            if enable_extraction:
                update("Extracting entities & relations...", 40)
                entities, relationships = extract_entities_relations(chunks, {})
                result['entities'] = entities
                result['relationships'] = relationships
                result['stats']['entities_count'] = sum(len(v) for v in entities.values())
                result['stats']['relationships_count'] = sum(len(v) for v in relationships.values())
            else:
                result['entities'] = {}
                result['relationships'] = {}

            # 4. Build knowledge graph (60%)
            if enable_graph and result['entities']:
                update("Building knowledge graph...", 60)
                kg = build_knowledge_graph(result['entities'], result['relationships'])
                result['graph'] = kg.to_dict()
                result['stats']['graph_nodes'] = kg.G.number_of_nodes()
                result['stats']['graph_edges'] = kg.G.number_of_edges()

            # 5. Generate embeddings (75%)
            if enable_embedding:
                update("Generating embeddings...", 75)
                
                # Chunk embeddings
                embeddings = generate_embeddings(chunks, batch_size=128)
                
                # Entity embeddings
                if result.get('entities'):
                    entity_embeds = generate_entity_embeddings(result['entities'], batch_size=128)
                    embeddings.extend(entity_embeds)
                
                # Relationship embeddings
                if result.get('relationships'):
                    from backend.core.embedding import generate_relationship_embeddings
                    rel_embeds = generate_relationship_embeddings(result['relationships'], batch_size=128)
                    embeddings.extend(rel_embeds)
                
                result['embeddings'] = embeddings
                result['stats']['embeddings_count'] = len(embeddings)
            
            #  6. AUTO-SAVE to databases (85-95%)
            if auto_save and self.mongo_storage and self.vector_db:
                doc_id = result['doc_id']
                
                # 6a. Save to MongoDB (85%)
                update("Saving to MongoDB...", 85)
                try:
                    mongo_success = self.mongo_storage.save_document_complete(
                        doc_id=doc_id,
                        filename=result['filename'],
                        filepath=result['filepath'],
                        chunks=result['chunks'],
                        entities=result.get('entities'),
                        relationships=result.get('relationships'),
                        graph=result.get('graph'),
                        stats=result['stats']
                    )
                    
                    if not mongo_success:
                        result['error'] = 'Failed to save to MongoDB'
                        result['success'] = False
                        return result
                    
                except Exception as e:
                    result['error'] = f'MongoDB save error: {str(e)}'
                    result['success'] = False
                    return result
                
                # 6b. Save embeddings to VectorDB (90%)
                update("Saving embeddings to VectorDB...", 90)
                try:
                    if result.get('embeddings'):
                        added_count = self.vector_db.add_document_embeddings_batch(
                            doc_id=doc_id,
                            filename=result['filename'],
                            embeddings=result['embeddings']
                        )
                        
                        if added_count == 0:
                            result['warning'] = 'No embeddings added to VectorDB'
                        else:
                            result['vectors_added'] = added_count
                    else:
                        result['warning'] = 'No embeddings to save'
                        
                except Exception as e:
                    result['error'] = f'VectorDB save error: {str(e)}'
                    result['success'] = False
                    return result

            update("Processing completed!", 100)
            result['success'] = True

        except Exception as e:
            import traceback
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"❌ Lỗi pipeline: {e}")
            traceback.print_exc()
            result['error'] = str(e)
            result['success'] = False

        return result
    