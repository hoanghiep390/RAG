# backend/retrieval/hybrid_retriever.py
"""
🔍 Hybrid Retriever - Enhanced for relationship-aware Q&A
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
import asyncio
import logging

from backend.retrieval.query_analyzer import QueryAnalyzer, QueryAnalysis
from backend.retrieval.vector_retriever import VectorRetriever, ScoredChunk
from backend.retrieval.graph_retriever import GraphRetriever, GraphContext
    
logger = logging.getLogger(__name__)

@dataclass
class RetrievalContext:
    """Enhanced context with relationship awareness"""
    query: str
    intent: str
    retrieval_mode: str
    
    chunks: List[ScoredChunk] = field(default_factory=list)
    entities: List[GraphContext] = field(default_factory=list)
    
    formatted_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self):
        return f"RetrievalContext(mode={self.retrieval_mode}, chunks={len(self.chunks)}, entities={len(self.entities)})"

class HybridRetriever:
    """Enhanced orchestrator with relationship-aware context"""
    
    def __init__(self, vector_db, mongo_storage):
        self.query_analyzer = QueryAnalyzer()
        self.vector_retriever = VectorRetriever(vector_db)
        self.graph_retriever = GraphRetriever(mongo_storage)
    
    def retrieve(
        self,
        query: str,
        force_mode: Optional[str] = None,
        top_k: Optional[int] = None,
        filter_category: Optional[List[str]] = None,
        filter_rel_type: Optional[List[str]] = None
    ) -> RetrievalContext:
        """
        Enhanced retrieval with relationship filtering
        
        Args:
            query: User query
            force_mode: Override mode
            top_k: Number of results
            filter_category: Graph categories to include
            filter_rel_type: Relationship types to include
        """
        try:
            import time
            start = time.time()
            
            analysis = self.query_analyzer.analyze(query)
            logger.info(f"📊 Intent: {analysis.intent}, Mode: {analysis.retrieval_mode}")
            
            mode = force_mode or analysis.retrieval_mode
            k = top_k or analysis.top_k
            
            # Retrieve
            if mode == 'vector':
                chunks, entities = self._vector_only(query, k)
            elif mode == 'graph':
                chunks, entities = self._graph_only(
                    analysis.entities, k, 
                    filter_category, filter_rel_type
                )
            else:  # hybrid
                chunks, entities = self._hybrid_search(
                    query, analysis.entities, k,
                    filter_category, filter_rel_type
                )
            
            # ✅ ENHANCED: Relationship-aware formatting
            formatted = self._format_context(query, chunks, entities, analysis)
            
            elapsed = time.time() - start
            metadata = {
                'retrieval_time_ms': int(elapsed * 1000),
                'retrieval_mode': mode,
                'num_chunks': len(chunks),
                'num_entities': len(entities),
                'intent': analysis.intent,
                'keywords': analysis.keywords[:5],
                'has_relationships': any(len(e.relationships) > 0 for e in entities)
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
            logger.error(f"❌ Retrieval failed: {e}")
            return RetrievalContext(
                query=query,
                intent='unknown',
                retrieval_mode='error',
                formatted_text=f"Error: {str(e)}"
            )
    
    def _vector_only(self, query: str, top_k: int) -> tuple:
        logger.info(f"🔍 Vector search (k={top_k})")
        chunks = self.vector_retriever.search(query, top_k=top_k)
        return chunks, []
    
    def _graph_only(
        self, 
        entity_names: List[str], 
        top_k: int,
        filter_category: Optional[List[str]] = None,
        filter_rel_type: Optional[List[str]] = None
    ) -> tuple:
        logger.info(f"🕸️ Graph search (entities={entity_names})")
        
        if not entity_names:
            return [], []
        
        entities = self.graph_retriever.search(
            entity_names=entity_names,
            k_hops=2,
            max_neighbors=top_k,
            filter_category=filter_category,
            filter_rel_type=filter_rel_type
        )
        return [], entities
    
    def _hybrid_search(
        self, 
        query: str, 
        entity_names: List[str], 
        top_k: int,
        filter_category: Optional[List[str]] = None,
        filter_rel_type: Optional[List[str]] = None
    ) -> tuple:
        logger.info(f"🔀 Hybrid search (k={top_k})")
        
        try:
            chunks, entities = asyncio.run(
                self._parallel_search(
                    query, entity_names, top_k,
                    filter_category, filter_rel_type
                )
            )
        except:
            logger.warning("⚠️ Async failed, using sequential")
            chunks = self.vector_retriever.search(query, top_k=int(top_k * 0.7))
            entities = []
            
            if entity_names:
                entities = self.graph_retriever.search(
                    entity_names=entity_names,
                    k_hops=1,
                    max_neighbors=int(top_k * 0.3),
                    filter_category=filter_category,
                    filter_rel_type=filter_rel_type
                )
        
        return chunks, entities
    
    async def _parallel_search(
        self, 
        query: str, 
        entity_names: List[str], 
        top_k: int,
        filter_category: Optional[List[str]] = None,
        filter_rel_type: Optional[List[str]] = None
    ) -> tuple:
        """Parallel execution"""
        
        async def vector_task():
            return self.vector_retriever.search(query, top_k=int(top_k * 0.7))
        
        async def graph_task():
            if not entity_names:
                return []
            return self.graph_retriever.search(
                entity_names=entity_names,
                k_hops=1,
                max_neighbors=int(top_k * 0.3),
                filter_category=filter_category,
                filter_rel_type=filter_rel_type
            )
        
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
        ✅ ENHANCED: Relationship-aware context formatting
        """
        lines = []
        
        # Header
        lines.append("=" * 80)
        lines.append(f"📋 RETRIEVAL CONTEXT FOR: {query}")
        lines.append(f"🎯 Intent: {analysis.intent} | Mode: {analysis.retrieval_mode}")
        lines.append("=" * 80)
        lines.append("")
        
        # Vector chunks
        if chunks:
            lines.append("## 📄 RELEVANT DOCUMENTS")
            lines.append("")
            
            for i, chunk in enumerate(chunks, 1):
                lines.append(f"[{i}] **Score: {chunk.score:.3f}** | Source: {chunk.filename}")
                lines.append(f"{chunk.content[:500]}...")
                lines.append("")
        
        # ✅ ENHANCED: Graph entities with relationship details
        if entities:
            lines.append("## 🕸️ KNOWLEDGE GRAPH - ENTITIES & RELATIONSHIPS")
            lines.append("")
            
            for i, entity in enumerate(entities, 1):
                lines.append(f"[{i}] **{entity.entity_name}** ({entity.entity_type})")
                
                if entity.description:
                    lines.append(f"    📝 Description: {entity.description[:200]}")
                
                # ✅ DETAILED RELATIONSHIPS
                if entity.relationships:
                    lines.append(f"    🔗 Relationships ({len(entity.relationships)}):")
                    
                    for rel in entity.relationships[:5]:
                        rel_type = rel['relationship_type']
                        verb = rel['verb_phrase']
                        category = rel['category']
                        target = rel['target']
                        desc = rel.get('description', '')
                        strength = rel.get('strength', 1.0)
                        
                        # Format: "→ OpenAI DEVELOPS (FUNCTIONAL) [S:0.95] GPT-4"
                        lines.append(
                            f"       → **{rel_type}** ({category}) [S:{strength:.2f}] **{target}**"
                        )
                        lines.append(f"         \"{verb}\" - {desc[:100]}")
                    
                    if len(entity.relationships) > 5:
                        lines.append(f"       ... and {len(entity.relationships) - 5} more")
                
                lines.append("")
        
        # ✅ ENHANCED: Instructions for relationship-aware Q&A
        lines.append("=" * 80)
        lines.append("## 📖 INSTRUCTIONS FOR ASSISTANT")
        lines.append("")
        lines.append("1. **Relationship Analysis**: Pay attention to:")
        lines.append("   - Relationship TYPES (e.g., DEVELOPS, MANAGES, OWNS)")
        lines.append("   - Verb phrases (natural language descriptions)")
        lines.append("   - Categories (FUNCTIONAL, HIERARCHICAL, TEMPORAL, etc.)")
        lines.append("   - Strength scores (higher = more important)")
        lines.append("")
        lines.append("2. **Answer Strategy**:")
        lines.append("   - Use BOTH documents and graph relationships")
        lines.append("   - Cite sources: [1], [2] for docs; mention entities for graph")
        lines.append("   - Explain relationships clearly with their types/categories")
        lines.append("   - If asked about connections, describe the relationship path")
        lines.append("")
        lines.append("3. **Examples**:")
        lines.append("   - 'What does OpenAI develop?' → Use DEVELOPS (FUNCTIONAL) relationships")
        lines.append("   - 'Who manages X?' → Use MANAGES (HIERARCHICAL) relationships")
        lines.append("   - 'How are X and Y connected?' → Describe relationship path with types")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def retrieve_with_rerank(
        self,
        query: str,
        top_k: int = 5,
        rerank_method: str = 'rrf'
    ) -> RetrievalContext:
        """Advanced retrieval with reranking"""
        context = self.retrieve(query, top_k=top_k * 2)
        
        if rerank_method == 'score':
            context.chunks.sort(key=lambda x: x.score, reverse=True)
            context.chunks = context.chunks[:top_k]
            
            context.entities.sort(key=lambda x: x.score, reverse=True)
            context.entities = context.entities[:top_k // 2]
        
        analysis = self.query_analyzer.analyze(query)
        context.formatted_text = self._format_context(
            query, context.chunks, context.entities, analysis
        )
        
        return context

# Convenience function
def retrieve(
    query: str,
    vector_db,
    mongo_storage,
    **kwargs
) -> RetrievalContext:
    """Quick retrieval"""
    retriever = HybridRetriever(vector_db, mongo_storage)
    return retriever.retrieve(query, **kwargs)