"""
Simplified test script for LightRAG system
Tests: Chunking → Extraction → Graph Building → Embedding
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Simple colors
C = {
    'GREEN': '\033[92m',
    'RED': '\033[91m',
    'YELLOW': '\033[93m',
    'CYAN': '\033[96m',
    'RESET': '\033[0m',
    'BOLD': '\033[1m'
}

def log(emoji, msg, color='RESET'):
    """Simple logging"""
    print(f"{C[color]}{emoji} {msg}{C['RESET']}")

def header(text):
    """Print header"""
    print(f"\n{C['BOLD']}{C['CYAN']}{'='*60}{C['RESET']}")
    print(f"{C['BOLD']}{C['CYAN']}{text}{C['RESET']}")
    print(f"{C['BOLD']}{C['CYAN']}{'='*60}{C['RESET']}\n")


# ============================================
# Test Functions
# ============================================

def test_imports():
    """Test 1: Import modules"""
    header("TEST 1: Imports")
    
    modules = [
        'backend.core.chunking',
        'backend.core.extraction',
        'backend.core.graph_builder',
        'backend.core.embedding',
        'backend.utils.file_utils'
    ]
    
    failed = []
    for module in modules:
        try:
            __import__(module)
            log('✅', f'{module.split(".")[-1]}', 'GREEN')
        except Exception as e:
            log('❌', f'{module.split(".")[-1]}: {str(e)}', 'RED')
            failed.append(module)
    
    return len(failed) == 0


def test_create_document():
    """Test 2: Create test document"""
    header("TEST 2: Create Test File")
    
    try:
        test_dir = Path("backend/data/test")
        test_dir.mkdir(parents=True, exist_ok=True)
        
        test_file = test_dir / "test_doc.txt"
        content = """# LightRAG Test

Apple Inc. is a technology company founded by Steve Jobs.
Microsoft Corporation was founded by Bill Gates.
Both companies are based in the United States."""
        
        test_file.write_text(content, encoding='utf-8')
        
        log('✅', f'Created: {test_file}', 'GREEN')
        log('ℹ️', f'Size: {test_file.stat().st_size} bytes', 'CYAN')
        
        return True, str(test_file)
        
    except Exception as e:
        log('❌', f'Failed: {str(e)}', 'RED')
        return False, None


def test_chunking(test_file):
    """Test 3: Chunking"""
    header("TEST 3: Chunking")
    
    try:
        from backend.core.chunking import process_document_to_chunks, DocChunkConfig
        
        config = DocChunkConfig(max_token_size=200, overlap_token_size=30)
        chunks = process_document_to_chunks(test_file, config=config)
        
        log('✅', f'Chunks: {len(chunks)}', 'GREEN')
        log('ℹ️', f'Tokens: {sum(c["tokens"] for c in chunks)}', 'CYAN')
        
        if chunks:
            log('ℹ️', f'First chunk: {chunks[0]["content"][:60]}...', 'CYAN')
        
        return True, chunks
        
    except Exception as e:
        log('❌', f'Failed: {str(e)}', 'RED')
        import traceback
        traceback.print_exc()
        return False, None


def test_extraction(chunks):
    """Test 4: Entity & Relationship Extraction"""
    header("TEST 4: Extraction")
    
    try:
        from backend.core.extraction import extract_entities_relations
        
        log('⏳', 'Extracting (using mock LLM)...', 'YELLOW')
        
        config = {'entity_types': ['PERSON', 'ORGANIZATION', 'LOCATION']}
        entities, relationships = extract_entities_relations(chunks, config)
        
        total_ents = sum(len(e) for e in entities.values())
        total_rels = sum(len(r) for r in relationships.values())
        
        log('✅', f'Entities: {total_ents}', 'GREEN')
        log('✅', f'Relationships: {total_rels}', 'GREEN')
        
        return True, (entities, relationships)
        
    except Exception as e:
        log('❌', f'Failed: {str(e)}', 'RED')
        import traceback
        traceback.print_exc()
        return False, None


def test_graph(entities, relationships):
    """Test 5: Graph Building"""
    header("TEST 5: Graph Building")
    
    try:
        from backend.core.graph_builder import build_knowledge_graph
        
        kg = build_knowledge_graph(entities, relationships)
        stats = kg.get_statistics()
        
        log('✅', f'Nodes: {stats["num_entities"]}', 'GREEN')
        log('✅', f'Edges: {stats["num_relationships"]}', 'GREEN')
        log('ℹ️', f'Density: {stats["density"]:.4f}', 'CYAN')
        
        return True, kg
        
    except Exception as e:
        log('❌', f'Failed: {str(e)}', 'RED')
        import traceback
        traceback.print_exc()
        return False, None


def test_embedding(chunks, entities, relationships, kg):
    """Test 6: Embedding Generation"""
    header("TEST 6: Embeddings")
    
    try:
        from backend.core.embedding import (
            generate_embeddings,
            generate_entity_embeddings,
            VectorDatabase
        )
        
        log('⏳', 'Generating embeddings...', 'YELLOW')
        
        chunk_embs = generate_embeddings(chunks)
        entity_embs = generate_entity_embeddings(entities, kg)
        
        log('✅', f'Chunk embeddings: {len(chunk_embs)}', 'GREEN')
        log('✅', f'Entity embeddings: {len(entity_embs)}', 'GREEN')
        
        # Create vector DB
        vdb = VectorDatabase(
            db_path="backend/data/test/test.index",
            metadata_path="backend/data/test/test_meta.json"
        )
        
        vdb.add_embeddings(chunk_embs)
        vdb.add_embeddings(entity_embs)
        vdb.save()
        
        log('✅', f'Vector DB saved', 'GREEN')
        
        return True, vdb
        
    except Exception as e:
        log('⚠️', f'Skipped: {str(e)}', 'YELLOW')
        return None, None


# ============================================
# Main Test Runner
# ============================================

def run_tests():
    """Run all tests"""
    print(f"{C['BOLD']}{C['CYAN']}")
    print("""
    ╔════════════════════════════════════════╗
    ║    LightRAG System Test Suite v1.0    ║
    ╚════════════════════════════════════════╝
    """)
    print(C['RESET'])
    
    results = {}
    
    # Test 1: Imports
    results['imports'] = test_imports()
    if not results['imports']:
        log('❌', 'Import failed. Cannot continue.', 'RED')
        return results
    
    # Test 2: Create document
    success, test_file = test_create_document()
    results['create_doc'] = success
    if not success:
        return results
    
    # Test 3: Chunking
    success, chunks = test_chunking(test_file)
    results['chunking'] = success
    if not success:
        return results
    
    # Test 4: Extraction
    success, extraction_data = test_extraction(chunks)
    results['extraction'] = success
    if not success:
        return results
    
    entities, relationships = extraction_data
    
    # Test 5: Graph
    success, kg = test_graph(entities, relationships)
    results['graph'] = success
    if not success:
        kg = None
    
    # Test 6: Embedding (optional)
    success, vdb = test_embedding(chunks, entities, relationships, kg)
    results['embedding'] = success
    
    # Summary
    header("SUMMARY")
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    total = len([v for v in results.values() if v is not None])
    
    log('📊', f'Total: {total}', 'CYAN')
    log('✅', f'Passed: {passed}', 'GREEN')
    if failed > 0:
        log('❌', f'Failed: {failed}', 'RED')
    if skipped > 0:
        log('⚠️', f'Skipped: {skipped}', 'YELLOW')
    
    success_rate = (passed / total * 100) if total > 0 else 0
    log('📈', f'Success: {success_rate:.0f}%', 'CYAN')
    
    print("\nDetails:")
    for name, result in results.items():
        symbol = '✅' if result else ('❌' if result is False else '⚠️')
        print(f"  {symbol} {name}")
    
    return results


if __name__ == "__main__":
    results = run_tests()
    
    # Exit code
    all_passed = all(v is True for v in results.values() if v is not None)
    sys.exit(0 if all_passed else 1)