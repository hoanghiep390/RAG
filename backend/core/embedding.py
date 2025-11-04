from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import numpy as np
from backend.utils.file_utils import save_to_json
from backend.core.chunking import process_document_to_chunks

model = SentenceTransformer('all-MiniLM-L6-v2')

class VectorDatabase:
    """
    Class quản lý vector database với FAISS
    """
    def __init__(self, db_path: str = "faiss.index", 
                 metadata_path: str = "faiss_meta.json", 
                 dim: int = 384):
        self.db_path = db_path
        self.metadata_path = metadata_path
        self.dim = dim
        self.metadata = {}
        self._load_or_create()
    
    def _load_or_create(self):
        """Load existing database or create new one"""
        try:
            import faiss
            import json
            import os
            
            if os.path.exists(self.db_path) and os.path.exists(self.metadata_path):
                self.index = faiss.read_index(self.db_path)
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            else:
                self.index = faiss.IndexFlatL2(self.dim)
                self.metadata = {}
        except ImportError:
            print("FAISS not installed. Using in-memory storage.")
            self.index = None
            self.vectors = []
            self.metadata = {}
    
    def add_embedding(self, id: str, text: str, embedding: List[float], 
                     entity_name: str = None, entity_type: str = None,
                     chunk_id: str = None, **metadata):
        """Thêm một embedding vào database"""
        embedding_array = np.array([embedding], dtype=np.float32)
        
        if self.index is not None:
            idx = self.index.ntotal
            self.index.add(embedding_array)
        else:
            idx = len(self.vectors)
            self.vectors.append(embedding_array[0])
        
        # Store metadata với tất cả thông tin
        self.metadata[str(idx)] = {
            'id': id,
            'text': text,
            'entity_name': entity_name,
            'entity_type': entity_type,
            'chunk_id': chunk_id,
            **metadata  # Include all additional metadata
        }
    
    def add_embeddings(self, embeddings: List[Dict[str, Any]]):
        """Thêm nhiều embeddings cùng lúc"""
        for emb in embeddings:
            # Extract metadata
            metadata = {k: v for k, v in emb.items() 
                       if k not in ['id', 'text', 'embedding', 'entity_name', 'entity_type', 'chunk_id']}
            
            self.add_embedding(
                id=emb['id'],
                text=emb['text'],
                embedding=emb['embedding'],
                entity_name=emb.get('entity_name'),
                entity_type=emb.get('entity_type'),
                chunk_id=emb.get('chunk_id'),
                **metadata
            )
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Tìm kiếm top-k embeddings gần nhất"""
        query_array = np.array([query_embedding], dtype=np.float32)
        
        if self.index is not None:
            distances, indices = self.index.search(query_array, top_k)
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if str(idx) in self.metadata:
                    result = self.metadata[str(idx)].copy()
                    result['distance'] = float(dist)
                    results.append(result)
        else:
            if not self.vectors:
                return []
            
            vectors_array = np.array(self.vectors)
            distances = np.linalg.norm(vectors_array - query_array, axis=1)
            top_indices = np.argsort(distances)[:top_k]
            
            results = []
            for idx in top_indices:
                if str(idx) in self.metadata:
                    result = self.metadata[str(idx)].copy()
                    result['distance'] = float(distances[idx])
                    results.append(result)
        
        return results
    
    def save(self):
        """Lưu database và metadata"""
        import json
        
        if self.index is not None:
            import faiss
            faiss.write_index(self.index, self.db_path)
        
        with open(self.metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Lấy thống kê về database"""
        if self.index is not None:
            total_vectors = self.index.ntotal
        else:
            total_vectors = len(self.vectors)
        
        type_counts = {}
        for meta in self.metadata.values():
            entity_type = meta.get('entity_type', 'CHUNK')
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1
        
        return {
            'total_vectors': total_vectors,
            'dimension': self.dim,
            'entity_type_counts': type_counts,
            'total_metadata_entries': len(self.metadata)
        }


def generate_embeddings(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    ✅ FIXED: Đồng bộ với chunking.py output
    
    Nhận đầu vào là list các chunk từ process_document_to_chunks()
    Mỗi chunk có: chunk_id, content, tokens, order, hierarchy, file_path, file_type
    
    Trả về list dict gồm tất cả thông tin + embedding vector
    """
    texts = [chunk['content'] for chunk in chunks]
    embeddings_array = model.encode(texts, show_progress_bar=True)
    
    out = []
    for chunk, emb in zip(chunks, embeddings_array):
        # ✅ FIXED: Sử dụng 'chunk_id' thay vì 'chunkid'
        out.append({
            'id': chunk.get('chunk_id', ''),           
            'text': chunk.get('content', ''),
            'embedding': emb.tolist(),
            'chunk_id': chunk.get('chunk_id', ''),     
            'tokens': chunk.get('tokens', 0),          
            'order': chunk.get('order', 0),
            'hierarchy': chunk.get('hierarchy', ''),
            'file_path': chunk.get('file_path', ''),
            'file_type': chunk.get('file_type', ''),
            'entity_type': 'CHUNK'                   
        })
    return out


def generate_entity_embeddings(entities_dict: Dict[str, List[Dict]], 
                              knowledge_graph=None) -> List[Dict[str, Any]]:
    """
    Sinh embeddings cho các entities từ knowledge graph
    
    entities_dict format: {chunk_id: [entity1, entity2, ...]}
    """
    embeddings = []
    texts = []
    entity_metadata = []
    
    for chunk_id, chunk_entities in entities_dict.items():
        for entity in chunk_entities:
            entity_name = entity['entity_name']
            entity_type = entity['entity_type']
            description = entity.get('description', '')
            
            # Enhanced với graph context nếu có
            if knowledge_graph and knowledge_graph.G.has_node(entity_name):
                node_data = knowledge_graph.G.nodes[entity_name]
                description = node_data.get('description', description)
            
            # Format: "EntityName (TYPE): description"
            text = f"{entity_name} ({entity_type}): {description}"
            texts.append(text)
            
            entity_metadata.append({
                'id': f"entity_{entity_name}",
                'entity_name': entity_name,
                'entity_type': entity_type,
                'chunk_id': chunk_id
            })
    
    if texts:
        embeddings_array = model.encode(texts, show_progress_bar=True)
        
        for i, emb in enumerate(embeddings_array):
            embeddings.append({
                'id': entity_metadata[i]['id'],
                'text': texts[i],
                'embedding': emb.tolist(),
                'entity_name': entity_metadata[i]['entity_name'],
                'entity_type': entity_metadata[i]['entity_type'],
                'chunk_id': entity_metadata[i]['chunk_id']
            })
    
    return embeddings


def generate_relationship_embeddings(relationships_dict: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
    """
    Sinh embeddings cho relationships
    
    relationships_dict format: {chunk_id: [rel1, rel2, ...]}
    """
    embeddings = []
    texts = []
    rel_metadata = []
    
    for chunk_id, chunk_rels in relationships_dict.items():
        for rel in chunk_rels:
            source = rel['source_id']
            target = rel['target_id']
            description = rel.get('description', '')
            
            # Format: "Source -> Target: description"
            text = f"{source} -> {target}: {description}"
            texts.append(text)
            
            rel_metadata.append({
                'id': f"rel_{source}_{target}_{chunk_id}",
                'source': source,
                'target': target,
                'chunk_id': chunk_id
            })
    
    if texts:
        embeddings_array = model.encode(texts, show_progress_bar=True)
        
        for i, emb in enumerate(embeddings_array):
            embeddings.append({
                'id': rel_metadata[i]['id'],
                'text': texts[i],
                'embedding': emb.tolist(),
                'source': rel_metadata[i]['source'],
                'target': rel_metadata[i]['target'],
                'chunk_id': rel_metadata[i]['chunk_id'],
                'entity_type': 'RELATIONSHIP'  # Đánh dấu là relationship
            })
    
    return embeddings


def search_similar(query: str, vector_db: VectorDatabase, 
                  top_k: int = 5, filter_type: str = None) -> List[Dict[str, Any]]:
    """
    Tìm kiếm các items tương tự với query
    """
    query_embedding = model.encode([query])[0]
    results = vector_db.search(query_embedding.tolist(), top_k=top_k * 2)
    
    # Filter by type if specified
    if filter_type:
        results = [r for r in results if r.get('entity_type') == filter_type]
    
    return results[:top_k]


def process_file(filepath: str, 
                entities_dict: Optional[Dict] = None,
                relationships_dict: Optional[Dict] = None,
                knowledge_graph=None) -> VectorDatabase:
    """
    ✅ FIXED: Đọc file, chunk rồi sinh embedding cho chunks, entities và relationships
    """
    # Generate chunk embeddings
    chunks = process_document_to_chunks(filepath)  
    chunk_embeddings = generate_embeddings(chunks)  
    
    # Initialize vector database
    dim = len(chunk_embeddings[0]["embedding"]) if chunk_embeddings else 384
    vector_db = VectorDatabase(
        db_path="faiss.index", 
        metadata_path="faiss_meta.json", 
        dim=dim
    )
    
    # Add chunk embeddings
    vector_db.add_embeddings(chunk_embeddings)
    
    # Add entity embeddings if provided
    if entities_dict:
        entity_embeddings = generate_entity_embeddings(entities_dict, knowledge_graph)
        vector_db.add_embeddings(entity_embeddings)
        save_to_json(entity_embeddings, "entity_embeddings.json")
    
    # Add relationship embeddings if provided
    if relationships_dict:
        rel_embeddings = generate_relationship_embeddings(relationships_dict)
        vector_db.add_embeddings(rel_embeddings)
        save_to_json(rel_embeddings, "relationship_embeddings.json")
    
    all_embeddings = {
        'chunks': chunk_embeddings,
        'entities': entity_embeddings if entities_dict else [],
        'relationships': rel_embeddings if relationships_dict else []
    }
    save_to_json(all_embeddings, "embedding_output.json")
    
    vector_db.save()
    
    return vector_db