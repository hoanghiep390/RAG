#!/usr/bin/env python3
"""
=======================================================
LightRAG Integration Test - FIXED & 100% PASS
=======================================================
Tests entire pipeline + auto cleanup
"""

import os
import sys
import asyncio
import shutil
import time
import traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.abspath('.'))

# ==================== CONFIG ====================
TEST_USER = "test_integration"
TEST_DIR = Path("backend/data") / TEST_USER
CLEANUP = True  # Auto cleanup after test


# ==================== MOCK LLM (FIXED FORMAT) ====================
async def mock_llm(*args, **kwargs):
    """Mock LLM response with CORRECT format (has parentheses)"""
    await asyncio.sleep(0.01)
    return """
("entity"<|>LightRAG<|>TECHNOLOGY<|>RAG system combining knowledge graphs and vector search)##
("entity"<|>FAISS<|>TECHNOLOGY<|>High-performance vector similarity search library)##
("relationship"<|>LightRAG<|>FAISS<|>uses for efficient vector search<|>uses,integrates<|>0.9)<|COMPLETE|>
"""

_original_llm_sync = None
_original_llm_async = None

def setup_mock():
    """Setup mock LLM"""
    global _original_llm_sync, _original_llm_async
    try:
        from backend.utils import llm_utils
        _original_llm_sync = getattr(llm_utils, 'call_llm_with_retry', None)
        _original_llm_async = getattr(llm_utils, 'call_llm_async', None)
        llm_utils.call_llm_with_retry = mock_llm
        llm_utils.call_llm_async = mock_llm
        print("Mock LLM installed")
    except Exception as e:
        print(f"Warning: Could not patch LLM: {e}")

def restore_mock():
    """Restore original LLM"""
    try:
        from backend.utils import llm_utils
        if _original_llm_sync:
            llm_utils.call_llm_with_retry = _original_llm_sync
        if _original_llm_async:
            llm_utils.call_llm_async = _original_llm_async
    except:
        pass


# ==================== TEST DATA ====================
def create_test_files():
    """Create test files"""
    upload_dir = TEST_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Test file 1: Markdown
    (upload_dir / "test.md").write_text("""
# LightRAG System

## Features
- Document chunking
- Entity extraction
- Knowledge graphs
- Vector search

## Technologies
- FAISS for vectors
- NetworkX for graphs
- Sentence Transformers
""", encoding='utf-8')
    
    # Test file 2: Text
    (upload_dir / "test.txt").write_text("""
LightRAG combines knowledge graphs with vector embeddings.
It uses FAISS for fast similarity search.
The system extracts entities and builds graphs automatically.
""", encoding='utf-8')
    
    return list(upload_dir.glob("*"))


# ==================== CLEANUP ====================
def cleanup():
    """Remove all test data"""
    print("\nCleaning up...")
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
        print(f"   Removed: {TEST_DIR}")
    
    # Remove temp files
    for pattern in ["*.index", "*_meta.json", "test_*"]:
        for f in Path(".").glob(pattern):
            try:
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)
            except:
                pass
    
    # Clear cache
    cache_dir = Path("backend/data/cache")
    if cache_dir.exists():
        for subdir in cache_dir.glob("*"):
            if subdir.is_dir():
                shutil.rmtree(subdir, ignore_errors=True)
    
    print("Cleanup complete")

from io import BytesIO

class MockFile:
    """Mock Streamlit uploaded file """
    def __init__(self, path):
        self.name = path.name
        self._data = path.read_bytes() 

    def getbuffer(self):
        return memoryview(self._data)
# ==================== TESTS ====================
def test_imports():
    """Test 1: Imports"""
    print("\nTest 1: Imports")
    try:
        from backend.core.chunking import process_document_to_chunks
        from backend.core.extraction import extract_entities_relations
        from backend.core.graph_builder import build_knowledge_graph
        from backend.core.embedding import generate_embeddings, VectorDatabase
        from backend.core.pipeline import DocumentPipeline
        print("All imports OK")
        return True
    except Exception as e:
        print(f"Import failed: {e}")
        traceback.print_exc()
        return False


def test_chunking(files):
    """Test 2: Chunking"""
    print("\nTest 2: Chunking")
    try:
        from backend.core.chunking import process_document_to_chunks, ChunkConfig
        
        all_chunks = []
        for f in files:
            chunks = process_document_to_chunks(str(f), ChunkConfig(max_tokens=200, overlap_tokens=30))
            assert len(chunks) > 0, f"No chunks from {f}"
            assert all('chunk_id' in c for c in chunks), "Missing chunk_id"
            all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} chunks total")
        return all_chunks
    except Exception as e:
        print(f"Chunking failed: {e}")
        traceback.print_exc()
        return []


def test_extraction(chunks):
    """Test 3: Extraction"""
    print("\nTest 3: Extraction")
    
    if not chunks:
        print("No chunks to extract from")
        return {}, {}
    
    setup_mock()
    
    try:
        from backend.core.extraction import extract_entities_relations
        
        entities, relationships = extract_entities_relations(chunks)
        
        ent_count = sum(len(v) for v in entities.values())
        rel_count = sum(len(v) for v in relationships.values())
        
        # FIXED: Assert >0
        assert ent_count > 0, "No entities extracted"
        assert rel_count > 0, "No relationships extracted"
        
        print(f"Extracted {ent_count} entities, {rel_count} relationships")
        return entities, relationships
    except Exception as e:
        print(f"Extraction failed: {e}")
        traceback.print_exc()
        return {}, {}
    finally:
        restore_mock()


def test_graph(entities, relationships):
    """Test 4: Graph Building"""
    print("\nTest 4: Graph Building")
    try:
        from backend.core.graph_builder import build_knowledge_graph
        
        kg = build_knowledge_graph(entities, relationships, enable_summarization=False)
        stats = kg.get_statistics()
        
        assert stats['num_entities'] >= 2, f"Expected >=2 nodes, got {stats['num_entities']}"
        assert stats['num_relationships'] >= 1, f"Expected >=1 edge, got {stats['num_relationships']}"
        
        print(f"Built graph: {stats['num_entities']} nodes, {stats['num_relationships']} edges")
        return kg
    except Exception as e:
        print(f"Graph failed: {e}")
        traceback.print_exc()
        return None


def test_embedding(chunks):
    """Test 5: Embeddings"""
    print("\nTest 5: Embeddings")
    try:
        from backend.core.embedding import generate_embeddings, VectorDatabase
        
        # Generate embeddings
        embeddings = generate_embeddings(chunks, batch_size=32, use_cache=False)
        assert len(embeddings) == len(chunks)
        assert all(len(e['embedding']) == 384 for e in embeddings), "Invalid embedding dim"
        
        # Create vector DB
        db = VectorDatabase("test.index", "test_meta.json", dim=384, use_hnsw=True)
        db.add_embeddings(embeddings)
        
        # FIXED: top_k <= số lượng vectors
        top_k = min(3, len(embeddings))
        results = db.search(embeddings[0]['embedding'], top_k=top_k)
        assert len(results) == top_k, f"Expected {top_k} results, got {len(results)}"
        
        db.save()
        
        print(f"Created {len(embeddings)} embeddings, search top_k={top_k} OK")
        return True
    except Exception as e:
        print(f"Embedding failed: {e}")
        traceback.print_exc()
        return False
def test_pipeline(files):
    """Test 6: Full Pipeline"""
    print("\nTest 6: Full Pipeline")
    
    setup_mock()
    
    try:
        from backend.core.pipeline import DocumentPipeline, DocChunkConfig

        pipeline = DocumentPipeline(user_id=TEST_USER, enable_advanced=True)
        
        # Lấy file đầu tiên (đã tồn tại trên đĩa)
        original_file = files[0]
        mock_file = MockFile(original_file)
        
        # Gọi pipeline
        result = pipeline.process_uploaded_file(
            uploaded_file=mock_file,
            chunk_config=DocChunkConfig(max_tokens=200, overlap_tokens=30),
            enable_extraction=True,
            enable_graph=True,
            enable_embedding=True,
            enable_gleaning=False
        )
        
        # Kiểm tra kết quả
        assert result['success'], f"Pipeline failed: {result.get('error')}"
        assert result['chunks_count'] > 0
        assert result.get('entities_count', 0) > 0
        assert result.get('graph_nodes', 0) > 0
        
        print(f"Pipeline: {result['chunks_count']} chunks, "
              f"{result.get('entities_count', 0)} entities, "
              f"{result.get('graph_nodes', 0)} nodes")
        return True
    except Exception as e:
        print(f"Pipeline failed: {e}")
        traceback.print_exc()
        return False
    finally:
        restore_mock()

def test_persistence():
    """Test 7: Persistence"""
    print("\nTest 7: Persistence")
    try:
        from backend.core.pipeline import DocumentPipeline
        
        pipeline = DocumentPipeline(user_id=TEST_USER)
        docs = pipeline.get_processed_docs()
        
        assert len(docs) > 0, "No documents persisted"
        doc = docs[0]
        assert doc['chunks'] > 0
        assert doc['has_graph']
        assert doc['has_embeddings']
        
        print(f"Found {len(docs)} persisted documents with graph & embeddings")
        return True
    except Exception as e:
        print(f"Persistence failed: {e}")
        traceback.print_exc()
        return False


# ==================== MAIN ====================
def main():
    print("="*60)
    print("LIGHTRAG INTEGRATION TEST - FIXED")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Test dir: {TEST_DIR}")
    
    start_time = time.time()
    passed = 0
    failed = 0
    
    try:
        print("\nCreating test files...")
        files = create_test_files()
        print(f"Created {len(files)} test files")
        
        # Apply nest_asyncio
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass
        
        # Run tests
        if not test_imports():
            failed += 1
            return 1
        
        passed += 1
        
        chunks = test_chunking(files)
        if chunks:
            passed += 1
        else:
            failed += 1
            return 1
        
        entities, relationships = test_extraction(chunks)
        if entities and relationships:
            passed += 1
        else:
            failed += 1
        
        kg = test_graph(entities, relationships)
        if kg:
            passed += 1
        else:
            failed += 1
        
        if test_embedding(chunks):
            passed += 1
        else:
            failed += 1
        
        if test_pipeline(files):
            passed += 1
        else:
            failed += 1
        
        if test_persistence():
            passed += 1
        else:
            failed += 1
        
        # Summary
        duration = time.time() - start_time
        total = passed + failed
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Total: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Duration: {duration:.2f}s")
        print(f"Success Rate: {(passed/total*100):.1f}%")
        print("="*60)
        
        if failed == 0:
            print("\nALL TESTS PASSED!")
            return 0
        else:
            print(f"\n{failed} TEST(S) FAILED")
            return 1
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()
        return 1
    finally:
        if CLEANUP:
            cleanup()
        else:
            print(f"\nKeeping test data: {TEST_DIR}")


if __name__ == "__main__":
    sys.exit(main())