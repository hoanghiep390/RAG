# backend/core/graph_builder.py
import networkx as nx
import logging
from typing import Tuple, Dict, List, Any, Optional
from pathlib import Path

from backend.utils.file_utils import save_to_json, load_from_json
from backend.core.chunking import process_document_to_chunks
from backend.core.extraction import extract_entities_relations

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Quản lý Knowledge Graph với NetworkX, hỗ trợ source_document"""
    
    def __init__(self):
        self.G = nx.DiGraph()

    def add_entity(
        self,
        entity_name: str,
        entity_type: str,
        description: str,
        source_id: str,
        source_document: str,
        **kwargs
    ):
        if self.G.has_node(entity_name):
            node = self.G.nodes[entity_name]
            if description and description not in node.get('description', ''):
                node['description'] = f"{node['description']}; {description}".strip('; ')
            sources = node.get('sources', set())
            sources.add(source_id)
            node['sources'] = sources
            docs = node.get('source_documents', set())
            docs.add(source_document)
            node['source_documents'] = docs
            for k, v in kwargs.items():
                if k not in node:
                    node[k] = v
        else:
            self.G.add_node(
                entity_name,
                type=entity_type,
                description=description,
                sources={source_id},
                source_documents={source_document},
                **kwargs
            )

    def add_relationship(
        self,
        source_entity: str,
        target_entity: str,
        description: str,
        strength: float = 1.0,
        chunk_id: Optional[str] = None,
        source_document: Optional[str] = None,
        **kwargs
    ):
        if self.G.has_edge(source_entity, target_entity):
            edge = self.G.edges[source_entity, target_entity]
            if description and description not in edge.get('description', ''):
                edge['description'] = f"{edge['description']}; {description}".strip('; ')
            edge['strength'] = max(edge.get('strength', 1.0), strength)
            chunks = edge.get('chunks', set())
            if chunk_id:
                chunks.add(chunk_id)
            edge['chunks'] = chunks
            docs = edge.get('source_documents', set())
            if source_document:
                docs.add(source_document)
            edge['source_documents'] = docs
        else:
            self.G.add_edge(
                source_entity,
                target_entity,
                description=description,
                strength=strength,
                chunks={chunk_id} if chunk_id else set(),
                source_documents={source_document} if source_document else set(),
                **kwargs
            )

    def to_dict(self) -> Dict[str, Any]:
        data = nx.node_link_data(self.G)
        for node in data['nodes']:
            for field in ['sources', 'source_documents']:
                if field in node and isinstance(node[field], set):
                    node[field] = list(node[field])
        for link in data['links']:
            for field in ['chunks', 'source_documents']:
                if field in link and isinstance(link[field], set):
                    link[field] = list(link[field])
        return data

    def get_statistics(self) -> Dict[str, Any]:
        return {
            'num_entities': self.G.number_of_nodes(),
            'num_relationships': self.G.number_of_edges(),
            'entity_types': self._count_entity_types(),
            'avg_degree': sum(dict(self.G.degree()).values()) / max(self.G.number_of_nodes(), 1),
            'density': nx.density(self.G)
        }

    def _count_entity_types(self) -> Dict[str, int]:
        counts = {}
        for _, data in self.G.nodes(data=True):
            t = data.get('type', 'UNKNOWN')
            counts[t] = counts.get(t, 0) + 1
        return counts


def build_knowledge_graph(
    entities_dict: Dict[str, List[Dict]],
    relationships_dict: Dict[str, List[Dict]],
    vector_db=None
) -> KnowledgeGraph:
    kg = KnowledgeGraph()
    for chunk_id, ents in entities_dict.items():
        doc = ents[0].get('source_document', Path(chunk_id).stem) if ents else 'unknown'
        for e in ents:
            e['source_document'] = e.get('source_document', doc)
            kg.add_entity(
                entity_name=e['entity_name'],
                entity_type=e['entity_type'],
                description=e['description'],
                source_id=e.get('source_id', chunk_id),
                source_document=e['source_document']
            )
    for chunk_id, rels in relationships_dict.items():
        doc = rels[0].get('source_document', Path(chunk_id).stem) if rels else 'unknown'
        for r in rels:
            r['source_document'] = r.get('source_document', doc)
            if not kg.G.has_node(r['source_id']):
                kg.add_entity(r['source_id'], 'UNKNOWN', '', chunk_id, doc)
            if not kg.G.has_node(r['target_id']):
                kg.add_entity(r['target_id'], 'UNKNOWN', '', chunk_id, doc)
            kg.add_relationship(
                source_entity=r['source_id'],
                target_entity=r['target_id'],
                description=r['description'],
                strength=r.get('strength', 1.0),
                chunk_id=chunk_id,
                source_document=doc
            )
    return kg


def process_file(filepath: str, global_config: Optional[Dict] = None, vector_db=None) -> Dict:
    logger.info(f"Processing: {filepath}")
    chunks = process_document_to_chunks(filepath)
    entities, relationships = extract_entities_relations(chunks, global_config)
    kg = build_knowledge_graph(entities, relationships, vector_db)
    result = {
        'graph': kg.to_dict(),
        'statistics': kg.get_statistics(),
        'metadata': {
            'source_file': filepath,
            'num_chunks': len(chunks),
            'num_entity_chunks': len(entities),
            'num_relationship_chunks': len(relationships)
        }
    }
    safe_name = Path(filepath).stem
    output = f"backend/data/{Path(filepath).parent.name}/graphs/{safe_name}_graph.json"
    save_to_json(result, output)
    return result


def merge_admin_graphs(user_id: str) -> Optional[KnowledgeGraph]:
    graphs_dir = Path(f"backend/data/{user_id}/graphs")
    if not graphs_dir.exists():
        logger.warning(f"No graphs dir for {user_id}")
        return None

    files = [f for f in graphs_dir.glob("*_graph.json") if f.name != "COMBINED_graph.json"]
    if not files:
        return None

    kg = KnowledgeGraph()
    logger.info(f"Merging {len(files)} graphs for {user_id}")

    for f in files:
        try:
            data = load_from_json(str(f))
            graph_data = data.get('graph', data)
            doc_name = f.stem.replace("_graph", "")
            for node in graph_data.get('nodes', []):
                src = list(node.get('sources', [])) or [f.stem]
                docs = list(node.get('source_documents', [])) or [doc_name]
                kg.add_entity(
                    entity_name=node['id'],
                    entity_type=node.get('type', 'UNKNOWN'),
                    description=node.get('description', ''),
                    source_id=src[0],
                    source_document=docs[0]
                )
            for link in graph_data.get('links', []):
                chunks = list(link.get('chunks', [])) or []
                docs = list(link.get('source_documents', [])) or [doc_name]
                kg.add_relationship(
                    source_entity=link['source'],
                    target_entity=link['target'],
                    description=link.get('description', ''),
                    strength=link.get('strength', 1.0),
                    chunk_id=chunks[0] if chunks else None,
                    source_document=docs[0]
                )
        except Exception as e:
            logger.error(f"Failed to load {f}: {e}")

    out_path = graphs_dir / "COMBINED_graph.json"
    save_to_json(kg.to_dict(), str(out_path))
    logger.info(f"COMBINED_graph.json saved: {kg.G.number_of_nodes()} nodes, {kg.G.number_of_edges()} edges")
    return kg


# === THÊM HÀM GLEANING_PROCESS ===
async def gleaning_process(
    entities_dict: Dict[str, List[Dict]],
    relationships_dict: Dict[str, List[Dict]],
    chunks: List[Dict],
    knowledge_graph: KnowledgeGraph,
    max_iterations: int = 2
) -> Tuple[Dict, Dict]:
    """
    Refinement bằng LLM: sửa lỗi, bổ sung, chuẩn hóa entities/relations
    """
    from backend.utils.llm_utils import call_llm_async
    from backend.core.extraction import parse_extraction_result, process_extraction_result

    current_entities = entities_dict.copy()
    current_relations = relationships_dict.copy()

    for it in range(max_iterations):
        logger.info(f"[Gleaning] Iteration {it+1}/{max_iterations}")

        entities_text = "\n".join([
            f"- {e['entity_name']} ({e['entity_type']}): {e.get('description', '')}"
            for ents in current_entities.values() for e in ents
        ]) or "Không có"

        relations_text = "\n".join([
            f"- {r['source_id']} → {r['target_id']}: {r.get('description', '')} (strength: {r['strength']})"
            for rels in current_relations.values() for r in rels
        ]) or "Không có"

        prompt = f"""
Bạn là chuyên gia tinh chỉnh Knowledge Graph.
Sửa lỗi, bổ sung thiếu sót, chuẩn hóa tên entity.

Entities hiện tại:
{entities_text}

Relations hiện tại:
{relations_text}

Yêu cầu:
- Sửa entity sai
- Bổ sung entity/relation mới
- Chuẩn hóa tên (ví dụ: Apple Inc. → Apple)
- Output định dạng: entity<|>name<|>type<|>desc<|>
  hoặc relationship<|>source<|>target<|>desc<|>strength<|>

Output:
"""

        try:
            response = await call_llm_async(prompt, temperature=0.0, max_tokens=1500)
            records = parse_extraction_result(response)
            new_ents, new_rels = process_extraction_result(records, chunk_id=f"gleaning_{it}")

            for chunk_id, ents in new_ents.items():
                current_entities.setdefault(chunk_id, []).extend(ents)
            for chunk_id, rels in new_rels.items():
                current_relations.setdefault(chunk_id, []).extend(rels)

            if len(new_ents) + len(new_rels) == 0:
                logger.info("[Gleaning] No improvement. Stopping.")
                break
        except Exception as e:
            logger.error(f"[Gleaning] Error in iteration {it+1}: {e}")
            break

    return current_entities, current_relations