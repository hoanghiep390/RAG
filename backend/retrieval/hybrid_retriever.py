# backend/retrieval/hybrid_retriever.py
"""
🚀 ENHANCED HYBRID RETRIEVAL 
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
import asyncio
import logging
import numpy as np

from backend.retrieval.query_analyzer import QueryAnalyzer, QueryAnalysis
from backend.retrieval.vector_retriever import VectorRetriever, ScoredChunk
from backend.retrieval.graph_retriever import GraphRetriever, GraphContext

logger = logging.getLogger(__name__)

#  RETRIEVAL MODES 

@dataclass
class RetrievalMode:
    """Retrieval configuration"""
    use_global: bool = True      
    use_local: bool = True       
    use_multi_hop: bool = False  
    expand_query: bool = False
    rerank: bool = True          

#  QUERY EXPANSION 

class QueryExpander:
    """Expand query with synonyms and related terms"""
    
    def __init__(self):
        # Common expansions (can be replaced with word2vec/bert)
        self.expansions = {
            'ai': ['artificial intelligence', 'machine learning', 'deep learning'],
            'ml': ['machine learning', 'ai', 'algorithm'],
            'ceo': ['chief executive officer', 'president', 'founder'],
            'develop': ['create', 'build', 'design', 'implement'],
            'company': ['organization', 'corporation', 'firm', 'enterprise']
        }
    
    def expand(self, query: str, max_terms: int = 3) -> List[str]:
        """Expand query with related terms"""
        expanded = [query]
        query_lower = query.lower()
        
        for term, synonyms in self.expansions.items():
            if term in query_lower:
                expanded.extend(synonyms[:max_terms])
                break
        
        return expanded[:3]  


#  GLOBAL RETRIEVAL

async def global_retrieval(
    query: str,
    vector_retriever: VectorRetriever,
    top_k: int = 10
) -> List[ScoredChunk]:
    """
    Global retrieval: Vector search across all documents
    Similar to LightRAG's global search
    """
    try:
        chunks = await asyncio.to_thread(
            vector_retriever.search,
            query,
            top_k=top_k
        )
        
        logger.info(f"🌐 Truy xuất toàn cục: {len(chunks)} chunks")
        return chunks
    
    except Exception as e:
        logger.error(f"❌ Truy xuất toàn cục thất bại: {e}")
        return []


#  LOCAL RETRIEVAL 

async def local_retrieval(
    entities: List[str],
    graph_retriever: GraphRetriever,
    k_hops: int = 2,
    max_neighbors: int = 10
) -> List[GraphContext]:
    """
    Local retrieval: Graph traversal from entities
    Similar to LightRAG's local search
    """
    try:
        contexts = await asyncio.to_thread(
            graph_retriever.search,
            entity_names=entities,
            k_hops=k_hops,
            max_neighbors=max_neighbors
        )
        
        logger.info(f"📍 Truy xuất cục bộ: {len(contexts)} entities")
        return contexts
    
    except Exception as e:
        logger.error(f"❌ Truy xuất cục bộ thất bại: {e}")
        return []


# MULTI-HOP REASONING 

def multi_hop_traversal(
    start_entities: List[str],
    graph_retriever: GraphRetriever,
    max_hops: int = 3,
    max_paths: int = 5
) -> List[Dict]:
    """
    Multi-hop graph traversal for complex reasoning
    Returns paths between entities
    """
    try:
        from backend.db.mongo_storage import MongoStorage
        
        # Get graph
        graph_data = graph_retriever.storage.get_graph()
        
        if not graph_data or not graph_data.get('nodes'):
            return []
        
        # Build adjacency list
        adj = {}
        for link in graph_data.get('links', []):
            src, tgt = link['source'], link['target']
            if src not in adj:
                adj[src] = []
            adj[src].append({
                'target': tgt,
                'type': link.get('relationship_type', 'RELATED_TO'),
                'strength': link.get('strength', 1.0)
            })
        
        # BFS for each start entity
        all_paths = []
        
        for start in start_entities:
            if start not in adj:
                continue
            
            # BFS
            queue = [(start, [start], 0)]
            visited = {start}
            paths = []
            
            while queue and len(paths) < max_paths:
                current, path, depth = queue.pop(0)
                
                if depth >= max_hops:
                    continue
                
                for neighbor in adj.get(current, []):
                    tgt = neighbor['target']
                    
                    if tgt not in visited:
                        visited.add(tgt)
                        new_path = path + [tgt]
                        
                        # Save path
                        if len(new_path) >= 2:
                            paths.append({
                                'path': new_path,
                                'length': len(new_path),
                                'relationships': [neighbor['type']]
                            })
                        
                        queue.append((tgt, new_path, depth + 1))
            
            all_paths.extend(paths[:max_paths])
        
        logger.info(f"🔀 Multi-hop: đã tìm thấy {len(all_paths)} đường đi")
        return all_paths[:max_paths]
    
    except Exception as e:
        logger.error(f"❌ Duyệt multi-hop thất bại: {e}")
        return []


# RESULT RERANKING 

class ResultReranker:
    """Rerank results based on multiple factors"""
    
    @staticmethod
    def rerank_chunks(
        chunks: List[ScoredChunk],
        query: str,
        entities: List[str]
    ) -> List[ScoredChunk]:
        """
        Rerank chunks based on:
        1. Vector similarity
        2. Entity mentions
        3. Content length
        """
        if not chunks:
            return []
        
        for chunk in chunks:
            score = chunk.score
            
            # Boost if contains entities
            entity_count = sum(1 for e in entities if e.lower() in chunk.content.lower())
            score += entity_count * 0.1
            
            # Boost longer content (more context)
            length_score = min(len(chunk.content) / 1000, 0.2)
            score += length_score
            
            # Update score
            chunk.score = min(score, 1.0)
        
        # Sort by new score
        chunks.sort(key=lambda x: x.score, reverse=True)
        
        return chunks
    
    @staticmethod
    def rerank_entities(
        entities: List[GraphContext],
        query: str
    ) -> List[GraphContext]:
        """
        Rerank entities based on:
        1. Graph score
        2. Query relevance
        3. Number of relationships
        """
        if not entities:
            return []
        
        query_lower = query.lower()
        
        for entity in entities:
            score = entity.score
            
            # Boost if name in query
            if entity.entity_name.lower() in query_lower:
                score += 0.3
            
            # Boost by relationships
            rel_score = min(len(entity.relationships) / 10, 0.2)
            score += rel_score
            
            entity.score = min(score, 1.0)
        
        entities.sort(key=lambda x: x.score, reverse=True)
        
        return entities


#  ENHANCED HYBRID RETRIEVER

@dataclass
class EnhancedRetrievalContext:
    """Enhanced context with dual-level retrieval"""
    query: str
    intent: str
    retrieval_mode: str
    
    # Global results
    global_chunks: List[ScoredChunk] = field(default_factory=list)
    
    # Local results
    local_entities: List[GraphContext] = field(default_factory=list)
    
    # Multi-hop results
    reasoning_paths: List[Dict] = field(default_factory=list)
    
    # Formatted output
    formatted_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class EnhancedHybridRetriever:
    """
    Enhanced retriever with LightRAG-inspired features
    Drop-in replacement for HybridRetriever
    """
    
    def __init__(self, vector_db, mongo_storage):
        # ✅ ENHANCED: Pass mongo_storage to QueryAnalyzer for semantic entity recognition
        self.query_analyzer = QueryAnalyzer(mongo_storage=mongo_storage)
        self.vector_retriever = VectorRetriever(vector_db)
        self.graph_retriever = GraphRetriever(mongo_storage)
        self.query_expander = QueryExpander()
        self.reranker = ResultReranker()
    
    def retrieve(
        self,
        query: str,
        mode: Optional[RetrievalMode] = None,
        force_mode: Optional[str] = None,
        top_k: int = 10,
        expand_query: bool = True,
        rerank: bool = True
    ) -> EnhancedRetrievalContext:
        """
        Enhanced retrieval with dual-level search
        
        Args:
            query: User query
            mode: Retrieval mode configuration (RetrievalMode object)
            force_mode: Force specific mode ('vector', 'graph', 'hybrid') - for backward compatibility
            top_k: Number of results
            expand_query: Enable query expansion
            rerank: Enable result reranking
        """
        import time
        start = time.time()
        
        # Handle force_mode for backward compatibility
        if force_mode is not None and mode is None:
            if force_mode == 'vector':
                mode = RetrievalMode(use_global=True, use_local=False, use_multi_hop=False)
            elif force_mode == 'graph':
                mode = RetrievalMode(use_global=False, use_local=True, use_multi_hop=False)
            elif force_mode == 'hybrid':
                mode = RetrievalMode(use_global=True, use_local=True, use_multi_hop=False)
        
        # Default mode
        if mode is None:
            mode = RetrievalMode()
        
        # Analyze query
        analysis = self.query_analyzer.analyze(query)
        logger.info(f"📊 Ý định: {analysis.intent}, Entities: {analysis.entities}")
        
        # Query expansion (optimized: only when explicitly enabled)
        queries = [query]
        if expand_query and mode.expand_query:
            from backend.config import Config
            if Config.ENABLE_QUERY_EXPANSION:
                queries = self.query_expander.expand(query)
                logger.info(f"🔍 Đã mở rộng thành {len(queries)} queries")
        
        # Execute retrieval
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        global_chunks, local_entities, paths = loop.run_until_complete(
            self._parallel_retrieval(
                queries, analysis.entities, mode, top_k
            )
        )
        
        # Reranking
        if rerank and mode.rerank:
            global_chunks = self.reranker.rerank_chunks(
                global_chunks, query, analysis.entities
            )
            local_entities = self.reranker.rerank_entities(
                local_entities, query
            )
        
        # Format context
        formatted = self._format_enhanced_context(
            query, global_chunks, local_entities, paths, analysis
        )
        
        elapsed = time.time() - start
        
        return EnhancedRetrievalContext(
            query=query,
            intent=analysis.intent,
            retrieval_mode='enhanced_hybrid',
            global_chunks=global_chunks[:top_k],
            local_entities=local_entities[:top_k//2],
            reasoning_paths=paths,
            formatted_text=formatted,
            metadata={
                'retrieval_time_ms': int(elapsed * 1000),
                'num_chunks': len(global_chunks),
                'num_entities': len(local_entities),
                'num_paths': len(paths),
                'expanded_queries': len(queries),
                'intent': analysis.intent
            }
        )
    
    async def _parallel_retrieval(
        self,
        queries: List[str],
        entities: List[str],
        mode: RetrievalMode,
        top_k: int
    ) -> Tuple[List[ScoredChunk], List[GraphContext], List[Dict]]:
        """Execute parallel retrieval - ✅ OPTIMIZED: Early returns for speed"""
        
        tasks = []
        
        # Global retrieval (for all queries)
        if mode.use_global:
            for q in queries:
                tasks.append(
                    global_retrieval(q, self.vector_retriever, top_k)
                )
        
        # Local retrieval - ✅ OPTIMIZED: Skip if no entities
        if mode.use_local and entities:
            tasks.append(
                local_retrieval(entities, self.graph_retriever, k_hops=2, max_neighbors=top_k)
            )
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Merge global chunks
        global_chunks = []
        for r in results:
            if isinstance(r, list) and r and isinstance(r[0], ScoredChunk):
                global_chunks.extend(r)
        
        # Deduplicate chunks by chunk_id
        seen = set()
        unique_chunks = []
        for chunk in global_chunks:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                unique_chunks.append(chunk)
        
        # Get local entities
        local_entities = []
        for r in results:
            if isinstance(r, list) and r and isinstance(r[0], GraphContext):
                local_entities = r
                break
        
        # Multi-hop reasoning - ✅ OPTIMIZED: Skip if no entities (saves time)
        paths = []
        if mode.use_multi_hop and entities and len(entities) > 0:
            paths = await asyncio.to_thread(
                multi_hop_traversal,
                entities,
                self.graph_retriever,
                max_hops=3,
                max_paths=5
            )
        
        return unique_chunks, local_entities, paths
    
    def _format_enhanced_context(
        self,
        query: str,
        global_chunks: List[ScoredChunk],
        local_entities: List[GraphContext],
        paths: List[Dict],
        analysis: QueryAnalysis
    ) -> str:
        """Format enhanced context for LLM"""
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append(f"📋 ENHANCED RETRIEVAL CONTEXT (DUAL-LEVEL)")
        lines.append(f"Query: {query}")
        lines.append(f"Intent: {analysis.intent} | Entities: {', '.join(analysis.entities[:5])}")
        lines.append("=" * 80)
        lines.append("")
        
        # Global context (documents) - ✅ OPTIMIZED: Reduced from 5 to 3 chunks, 400 to 200 chars
        if global_chunks:
            from backend.config import Config
            max_chunks = Config.MAX_CONTEXT_CHUNKS
            lines.append("## 🌐 GLOBAL CONTEXT (Documents)")
            lines.append("")
            
            for i, chunk in enumerate(global_chunks[:max_chunks], 1):
                lines.append(f"[{i}] **Score: {chunk.score:.3f}** | {chunk.filename}")
                lines.append(f"{chunk.content[:200]}...")
                lines.append("")
        
        # Local context (graph) - ✅ OPTIMIZED: Reduced from 5 to 3 entities
        if local_entities:
            from backend.config import Config
            max_chunks = Config.MAX_CONTEXT_CHUNKS
            lines.append("## 📍 LOCAL CONTEXT (Knowledge Graph)")
            lines.append("")
            
            for i, entity in enumerate(local_entities[:max_chunks], 1):
                lines.append(f"[{i}] **{entity.entity_name}** ({entity.entity_type})")
                
                if entity.description:
                    lines.append(f"    📝 {entity.description[:150]}")
                
                if entity.relationships:
                    lines.append(f"    🔗 Relationships:")
                    for rel in entity.relationships[:3]:
                        rel_type = rel.get('relationship_type', rel.get('keywords', 'RELATED_TO'))
                        category = rel.get('category', rel.get('keywords', ''))
                        lines.append(
                            f"       • {rel_type} → {rel['target']} "
                            f"[{category}]"
                        )
                
                lines.append("")
        
        # Multi-hop reasoning paths
        if paths:
            lines.append("## 🔀 REASONING PATHS (Multi-hop)")
            lines.append("")
            
            for i, path_info in enumerate(paths[:3], 1):
                path = path_info['path']
                lines.append(f"[{i}] {' → '.join(path)}")
                lines.append("")
        
        # Instructions
        lines.append("=" * 80)
        lines.append("## 📖 INSTRUCTIONS")
        lines.append("")
        lines.append("1. **Global Context**: Use document chunks for detailed information")
        lines.append("2. **Local Context**: Use graph for entity relationships")
        lines.append("3. **Reasoning Paths**: Use for multi-step reasoning")
        lines.append("4. **Citation**: [1], [2] for docs; mention entities for graph")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    # Backward compatibility
    def retrieve_with_rerank(
        self,
        query: str,
        top_k: int = 5,
        rerank_method: str = 'enhanced'
    ):
        """Backward compatible method"""
        return self.retrieve(query, top_k=top_k, rerank=True)


#  CONVENIENCE FUNCTIONS 

def retrieve_enhanced(
    query: str,
    vector_db,
    mongo_storage,
    **kwargs
) -> EnhancedRetrievalContext:
    """Quick enhanced retrieval"""
    retriever = EnhancedHybridRetriever(vector_db, mongo_storage)
    return retriever.retrieve(query, **kwargs)


#  EXPORT 

__all__ = [
    'EnhancedHybridRetriever',
    'EnhancedRetrievalContext',
    'RetrievalMode',
    'retrieve_enhanced',
    'QueryExpander',
    'ResultReranker'
]