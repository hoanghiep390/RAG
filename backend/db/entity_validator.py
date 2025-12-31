# backend/db/entity_validator.py
"""
Bộ Kiểm Tra Entity - Xác thực ràng buộc khóa ngoại
Đảm bảo tính toàn vẹn dữ liệu cho relationships
"""

from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class EntityValidator:
    """
    Validator để đảm bảo foreign key constraints
    
    Features:
    - xác nhận relationships có entities tồn tại
    - Cache entities để tăng performance
    - Cleanup orphaned relationships
    """
    
    def __init__(self, mongo_storage):
        """
        Tham số:
            mongo_storage 
        """
        self.storage = mongo_storage
        self.user_id = mongo_storage.user_id
        self._entity_cache = {}  
        self._cache_loaded = False
    
    def _load_entity_cache(self, force_reload: bool = False):
        """Tải tất cả entities vào cache"""
        if self._cache_loaded and not force_reload:
            return
        
        try:
            # Query all entities for user
            entities = self.storage.entities.find(
                {'user_id': self.user_id},
                {'entity_name': 1}
            )
            
            # Build cache
            self._entity_cache = {
                e['entity_name']: True 
                for e in entities
            }
            
            self._cache_loaded = True
            logger.info(f"✅ Loaded {len(self._entity_cache)} entities into cache")
            
        except Exception as e:
            logger.error(f"❌ Failed to load entity cache: {e}")
            self._entity_cache = {}
    
    def entity_exists(self, entity_name: str, use_cache: bool = True) -> bool:
        """
        Kiểm tra nếu entity tồn tại
        
        Tham số:
            entity_name: Tên entity cần kiểm tra
            use_cache: Dùng cache (nhanh hơn) hoặc query DB (chính xác hơn)
        
        Trả về:
            True nếu entity tồn tại
        """
        if not entity_name:
            return False
        
        # Use cache
        if use_cache:
            if not self._cache_loaded:
                self._load_entity_cache()
            return entity_name in self._entity_cache
        
        # Query DB directly
        try:
            result = self.storage.entities.find_one({
                'user_id': self.user_id,
                'entity_name': entity_name
            })
            return result is not None
        except Exception as e:
            logger.error(f"❌ Failed to check entity existence: {e}")
            return False
    
    def validate_relationship(
        self, 
        source_id: str, 
        target_id: str,
        use_cache: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Xác thực một relationship
        
        Tham số:
            source_id: Tên entity nguồn
            target_id: Tên entity đích
            use_cache: Dùng cache cho validation
        
        Trả về:
            (is_valid, error_message)
        """
        # Check source exists
        if not self.entity_exists(source_id, use_cache):
            return False, f"Source entity not found: {source_id}"
        
        # Check target exists
        if not self.entity_exists(target_id, use_cache):
            return False, f"Target entity not found: {target_id}"
        
        # Check not self-loop
        if source_id == target_id:
            return False, f"Self-loop not allowed: {source_id}"
        
        return True, None
    
    def validate_relationships_bulk(
        self,
        relationships_dict: Dict[Tuple[str, str], List[Dict]],
        use_cache: bool = True
    ) -> Tuple[Dict, List[Dict]]:
        """
        Xác thực batch relationships
        
        Tham số:
            relationships_dict: Dict của {(source, target): [relationship_dicts]}
            use_cache: Dùng cache cho validation
        
        Trả về:
            (valid_relationships_dict, invalid_relationships_list)
        """
        if not relationships_dict:
            return {}, []
        
        # Load cache once
        if use_cache:
            self._load_entity_cache(force_reload=True)
        
        valid_rels = {}
        invalid_rels = []
        
        for (source, target), rels in relationships_dict.items():
            is_valid, error = self.validate_relationship(source, target, use_cache)
            
            if is_valid:
                valid_rels[(source, target)] = rels
            else:
                # Log invalid relationships
                for rel in rels:
                    invalid_rel = rel.copy()
                    invalid_rel['validation_error'] = error
                    invalid_rels.append(invalid_rel)
                
                logger.warning(f"⚠️ Invalid relationship: {source} → {target} ({error})")
        
        logger.info(
            f"✅ Validation: {len(valid_rels)} valid, "
            f"{len(invalid_rels)} invalid relationships"
        )
        
        return valid_rels, invalid_rels
    
    def get_orphaned_relationships(self) -> List[Dict]:
        """
        Tìm relationships không có entities
        
        Trả về:
            Danh sách orphaned relationship documents
        """
        try:
            # Get all entity names
            self._load_entity_cache(force_reload=True)
            valid_entities = set(self._entity_cache.keys())
            
            # Find relationships with invalid source or target
            orphaned = []
            
            relationships = self.storage.relationships.find({
                'user_id': self.user_id
            })
            
            for rel in relationships:
                source = rel.get('source_id')
                target = rel.get('target_id')
                
                if source not in valid_entities or target not in valid_entities:
                    orphaned.append(rel)
            
            logger.info(f"🔍 Found {len(orphaned)} orphaned relationships")
            return orphaned
            
        except Exception as e:
            logger.error(f"❌ Failed to find orphaned relationships: {e}")
            return []
    
    def cleanup_orphaned_relationships(self, dry_run: bool = True) -> Dict:
        """
        Dọn dẹp orphaned relationships
        
        Tham số:
            dry_run: Nếu True, chỉ báo cáo những gì sẽ bị xóa
        
        Trả về:
            Dict thống kê
        """
        stats = {
            'orphaned_count': 0,
            'deleted_count': 0,
            'errors': []
        }
        
        try:
            orphaned = self.get_orphaned_relationships()
            stats['orphaned_count'] = len(orphaned)
            
            if not orphaned:
                logger.info("✅ No orphaned relationships found")
                return stats
            
            if dry_run:
                logger.info(f"🔍 DRY RUN: Would delete {len(orphaned)} relationships")
                return stats
            
            # Delete orphaned relationships
            orphaned_ids = [rel['_id'] for rel in orphaned]
            
            result = self.storage.relationships.delete_many({
                '_id': {'$in': orphaned_ids}
            })
            
            stats['deleted_count'] = result.deleted_count
            
            logger.info(f"✅ Deleted {stats['deleted_count']} orphaned relationships")
            
        except Exception as e:
            error_msg = f"Failed to cleanup orphaned relationships: {e}"
            logger.error(f"❌ {error_msg}")
            stats['errors'].append(error_msg)
        
        return stats
    
    def validate_graph_edges(
        self,
        graph_data: Dict,
        use_cache: bool = True
    ) -> Tuple[Dict, List[Dict]]:
        """
        Validate graph edges có nodes tồn tại
        
        Args:
            graph_data: Graph dict với 'nodes' và 'links'
            use_cache: Use cache for validation
        
        Returns:
            (valid_graph_data, invalid_edges)
        """
        if not graph_data or not graph_data.get('links'):
            return graph_data, []
        
        # Build node set
        node_ids = {node['id'] for node in graph_data.get('nodes', [])}
        
        valid_links = []
        invalid_links = []
        
        for link in graph_data['links']:
            source = link.get('source')
            target = link.get('target')
            
            if source in node_ids and target in node_ids:
                valid_links.append(link)
            else:
                invalid_link = link.copy()
                invalid_link['validation_error'] = (
                    f"Node not found: "
                    f"{'source' if source not in node_ids else 'target'}"
                )
                invalid_links.append(invalid_link)
        
        if invalid_links:
            logger.warning(f"⚠️ {len(invalid_links)} invalid graph edges filtered")
        
        valid_graph = {
            'nodes': graph_data.get('nodes', []),
            'links': valid_links
        }
        
        return valid_graph, invalid_links
    
    def refresh_cache(self):
        """Refresh entity cache"""
        self._load_entity_cache(force_reload=True)
    
    def clear_cache(self):
        """Clear entity cache"""
        self._entity_cache = {}
        self._cache_loaded = False


# ================= Convenience Functions =================

def validate_relationships(
    relationships_dict: Dict,
    mongo_storage,
    use_cache: bool = True
) -> Tuple[Dict, List[Dict]]:
    """
    Quick validation function
    
    Usage:
        valid, invalid = validate_relationships(rels_dict, storage)
    """
    validator = EntityValidator(mongo_storage)
    return validator.validate_relationships_bulk(relationships_dict, use_cache)


def cleanup_orphaned_data(
    mongo_storage,
    dry_run: bool = True
) -> Dict:
    """
    Quick cleanup function
    
    Usage:
        stats = cleanup_orphaned_data(storage, dry_run=False)
    """
    validator = EntityValidator(mongo_storage)
    return validator.cleanup_orphaned_relationships(dry_run)


# ================= Export =================

__all__ = [
    'EntityValidator',
    'validate_relationships',
    'cleanup_orphaned_data'
]
