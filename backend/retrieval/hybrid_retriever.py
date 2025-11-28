# backend/retrieval/hybrid_retriever.py
"""
 Hybrid Retriever - Orchestrator
Kết hợp vector search + graph traversal để tạo context
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import asyncio
import logging

from backend.retrieval.query_analyzer import QueryAnalyzer, QueryAnalysis
from backend.retrieval.vector_retriever import VectorRetriever, ScoredChunk
from backend.retrieval.graph_retriever import GraphRetriever, GraphContext
    
logger = logging.getLogger(__name__)

# ================= Data Classes =================
@dataclass
class RetrievalContext:
    """Final context for LLM"""
    query: str
    intent: str
    retrieval_mode: str
    
    # Results
    chunks: List[ScoredChunk] = field(default_factory=list)
    entities: List[GraphContext] = field(default_factory=list)
    
    # Formatted output
    formatted_text: str = ""
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"RetrievalContext(mode={self.retrieval_mode}, chunks={len(self.chunks)}, entities={len(self.entities)})"

#  Main Retriever
class HybridRetriever:
    """
    Main orchestrator - Kết hợp tất cả retrievers
    
    Usage:
        retriever = HybridRetriever(vector_db, mongo_storage)
        context = retriever.retrieve("What is GPT-4?")
        print(context.formatted_text)  # Ready for LLM
    """
    
    def __init__(self, vector_db, mongo_storage):
        """
        Args:
            vector_db: VectorDatabase instance
            mongo_storage: MongoStorage instance
        """
        self.query_analyzer = QueryAnalyzer()
        self.vector_retriever = VectorRetriever(vector_db)
        self.graph_retriever = GraphRetriever(mongo_storage)
    
    def retrieve(
        self,
        query: str,
        force_mode: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> RetrievalContext:
        """
        Main retrieval function
        
        Args:
            query: User query
            force_mode: Override auto mode ('vector', 'graph', 'hybrid')
            top_k: Override auto top_k
        
        Returns:
            RetrievalContext with formatted text for LLM
        """
        try:
            import time
            start_time = time.time()
            
            # Step 1: Analyze query
            analysis = self.query_analyzer.analyze(query)
            logger.info(f"Query analyzed: intent={analysis.intent}, mode={analysis.retrieval_mode}")
            
            # Override if needed
            mode = force_mode or analysis.retrieval_mode
            k = top_k or analysis.top_k
            
            # Step 2: Route to appropriate retrieval
            if mode == 'vector':
                chunks, entities = self._vector_only(query, k)
            elif mode == 'graph':
                chunks, entities = self._graph_only(analysis.entities, k)
            else:  # hybrid
                chunks, entities = self._hybrid_search(query, analysis.entities, k)
            
            # Step 3: Format context
            formatted = self._format_context(query, chunks, entities, analysis)
            
            # Step 4: Build metadata
            elapsed = time.time() - start_time
            metadata = {
                'retrieval_time_ms': int(elapsed * 1000),
                'retrieval_mode': mode,
                'num_chunks': len(chunks),
                'num_entities': len(entities),
                'intent': analysis.intent,
                'keywords': analysis.keywords[:5]
            }
            
            return RetrievalContext(
                query=query,
                intent=analysis.intent,
                retrieval_mode=mode,
                chunks=chunks,
                entities=entities,
                formatted_text=formatted,
                metadata=metadata
            )
        
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            # Return empty context on error
            return RetrievalContext(
                query=query,
                intent='unknown',
                retrieval_mode='error',
                formatted_text=f"Error during retrieval: {str(e)}"
            )
    
    def _vector_only(self, query: str, top_k: int) -> tuple:
        """Vector search only"""
        logger.info(f"Vector-only retrieval (top_k={top_k})")
        
        chunks = self.vector_retriever.search(query, top_k=top_k)
        entities = []
        
        return chunks, entities
    
    def _graph_only(self, entity_names: List[str], top_k: int) -> tuple:
        """Graph search only"""
        logger.info(f"Graph-only retrieval (entities={entity_names})")
        
        if not entity_names:
            logger.warning("No entities for graph search, returning empty")
            return [], []
        
        entities = self.graph_retriever.search(
            entity_names=entity_names,
            k_hops=2,
            max_neighbors=top_k
        )
        chunks = []
        
        return chunks, entities
    
    def _hybrid_search(self, query: str, entity_names: List[str], top_k: int) -> tuple:
        """
        Hybrid search - Parallel vector + graph
        
        Returns more chunks from vector, fewer entities from graph for balance
        """
        logger.info(f"Hybrid retrieval (top_k={top_k}, entities={entity_names})")
        
        # Parallel execution (if possible)
        try:
            # Try async parallel
            chunks, entities = asyncio.run(self._parallel_search(query, entity_names, top_k))
        except:
            # Fallback to sequential
            logger.warning("Async failed, using sequential search")
            chunks = self.vector_retriever.search(query, top_k=int(top_k * 0.7))
            entities = []
            
            if entity_names:
                entities = self.graph_retriever.search(
                    entity_names=entity_names,
                    k_hops=1,
                    max_neighbors=int(top_k * 0.3)
                )
        
        return chunks, entities
    
    async def _parallel_search(self, query: str, entity_names: List[str], top_k: int) -> tuple:
        """Execute vector and graph search in parallel"""
        
        async def vector_task():
            return self.vector_retriever.search(query, top_k=int(top_k * 0.7))
        
        async def graph_task():
            if not entity_names:
                return []
            return self.graph_retriever.search(
                entity_names=entity_names,
                k_hops=1,
                max_neighbors=int(top_k * 0.3)
            )
        
        # Run in parallel
        chunks, entities = await asyncio.gather(
            asyncio.to_thread(vector_task),
            asyncio.to_thread(graph_task)
        )
        
        return chunks, entities
    
    def _format_context(
        self,
        query: str,
        chunks: List[ScoredChunk],
        entities: List[GraphContext],
        analysis: QueryAnalysis
    ) -> str:
        """
        Format final context for LLM prompt
        
        Structure:
        1. Query info
        2. Vector search results (chunks)
        3. Graph context (entities + relationships)
        4. Instructions for LLM
        """
        lines = []
        
        # Header
        lines.append("=" * 70)
        lines.append(f"RETRIEVAL CONTEXT FOR QUERY: {query}")
        lines.append(f"Intent: {analysis.intent} | Mode: {analysis.retrieval_mode}")
        lines.append("=" * 70)
        lines.append("")
        
        # Section 1: Chunks (if any)
        if chunks:
            lines.append("## 📄 RELEVANT DOCUMENTS")
            lines.append("")
            
            for i, chunk in enumerate(chunks, 1):
                lines.append(f"[{i}] Score: {chunk.score:.3f} | Source: {chunk.filename}")
                lines.append(f"{chunk.content}")
                lines.append("")
        
        # Section 2: Entities (if any)
        if entities:
            lines.append("## 🕸️ KNOWLEDGE GRAPH CONTEXT")
            lines.append("")
            
            for i, entity in enumerate(entities, 1):
                lines.append(f"[{i}] **{entity.entity_name}** ({entity.entity_type})")
                
                if entity.description:
                    lines.append(f"    Description: {entity.description}")
                
                if entity.neighbors:
                    lines.append(f"    Related to: {', '.join(entity.neighbors[:5])}")
                
                if entity.relationships:
                    lines.append(f"    Key relationships:")
                    for rel in entity.relationships[:3]:
                        lines.append(f"      → {rel['target']}: {rel['description']}")
                
                lines.append("")
        
        # Footer: Instructions
        lines.append("=" * 70)
        lines.append("INSTRUCTIONS:")
        lines.append("1. Use the above context to answer the query accurately")
        lines.append("2. Cite sources using [1], [2], etc.")
        lines.append("3. If context is insufficient, acknowledge limitations")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def retrieve_with_rerank(
        self,
        query: str,
        top_k: int = 5,
        rerank_method: str = 'rrf'
    ) -> RetrievalContext:
        """
        Advanced retrieval with reranking
        
        Args:
            query: User query
            top_k: Final number of results
            rerank_method: 'rrf' (Reciprocal Rank Fusion) or 'score'
        
        Note: This is a placeholder for future reranker integration
        """
        # For now, just use regular retrieve
        context = self.retrieve(query, top_k=top_k * 2)
        
        # Simple reranking by score
        if rerank_method == 'score':
            context.chunks.sort(key=lambda x: x.score, reverse=True)
            context.chunks = context.chunks[:top_k]
            
            context.entities.sort(key=lambda x: x.score, reverse=True)
            context.entities = context.entities[:top_k // 2]
        
        # Reformat after reranking
        analysis = self.query_analyzer.analyze(query)
        context.formatted_text = self._format_context(
            query, context.chunks, context.entities, analysis
        )
        
        return context

# ================= Convenience Function =================
def retrieve(
    query: str,
    vector_db,
    mongo_storage,
    **kwargs
) -> RetrievalContext:
    """
    Quick retrieval function
    
    Usage:
        from backend.db.vector_db import VectorDatabase
        from backend.db.mongo_storage import MongoStorage
        from backend.retrieval.hybrid_retriever import retrieve
        
        vector_db = VectorDatabase(user_id='admin_00000000')
        storage = MongoStorage(user_id='admin_00000000')
        
        context = retrieve("What is AI?", vector_db, storage)
        print(context.formatted_text)
    """
    retriever = HybridRetriever(vector_db, mongo_storage)
    return retriever.retrieve(query, **kwargs)

