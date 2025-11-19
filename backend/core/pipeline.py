# ==========================================
# backend/core/pipeline.py 
# ==========================================
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime

from backend.core.chunking import process_document_to_chunks, ChunkConfig
from backend.core.extraction import extract_entities_relations
from backend.core.graph_builder import build_knowledge_graph
from backend.core.embedding import generate_embeddings, generate_entity_embeddings
from backend.utils.file_utils import save_uploaded_file

class DocumentPipeline:
    """Simple pipeline"""
    
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.upload_dir = Path("backend/data") / user_id / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def process_file(self, uploaded_file, chunk_config: ChunkConfig = None,
                    enable_extraction: bool = True, enable_graph: bool = True,
                    enable_embedding: bool = True,
                    progress_callback: Callable = None) -> Dict[str, Any]:
        """Process 1 file"""
        
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

            # 5. Generate embeddings (80%)
            if enable_embedding:
                update("Generating embeddings...", 80)
                embeddings = generate_embeddings(chunks, batch_size=128)

                if result.get('entities'):
                    entity_embeds = generate_entity_embeddings(result['entities'], batch_size=128)
                    embeddings.extend(entity_embeds)

                if result.get('relationships'):
                    from backend.core.embedding import generate_relationship_embeddings
                    rel_embeds = generate_relationship_embeddings(result['relationships'], batch_size=128)
                    embeddings.extend(rel_embeds)

                result['embeddings'] = embeddings
                result['stats']['embeddings_count'] = len(embeddings)

            # Final: Success
            update("Processing completed!", 100)
            result['success'] = True

        except Exception as e:
            import traceback
            print(f"Pipeline error: {e}")
            traceback.print_exc()
            result['error'] = str(e)
            result['success'] = False

        return result
    