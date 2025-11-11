#!/usr/bin/env python3
"""
=======================================================
LightRAG Integration Test - Minimal Version
=======================================================
Tests entire pipeline + auto cleanup
"""

import os
import sys
import asyncio
import shutil
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.abspath('.'))

# ==================== CONFIG ====================
TEST_USER = "test_integration"
TEST_DIR = Path("backend/data") / TEST_USER
CLEANUP = True  # Auto cleanup after test


# ==================== MOCK LLM ====================
async def mock_llm(*args, **kwargs):
    """Mock LLM response"""
    await asyncio.sleep(0.01)
    return """
entity<|>LightRAG<|>TECHNOLOGY<|>RAG system with graphs and vectors
entity<|>FAISS<|>TECHNOLOGY<|>Vector search library##
relationship<|>LightRAG<|>FAISS<|>uses for search<|>uses<|>0.9<|COMPLETE|>
"""

def setup_mock():
    """Setup mock LLM"""
    global _original
    try:
        from backend.utils import llm_utils
        _original = llm_utils.call_llm_with_retry
        llm_utils.call_llm_with_retry = mock_llm
        llm_utils.call_llm_async = mock_llm
    except:
        pass

def restore_mock():
    """Restore original LLM"""
    try:
        from backend.utils import llm_utils
        llm_utils.call_llm_with_retry = _original
        llm_utils.call_llm_async = _original
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
    print("\n🗑️ Cleaning up...")
    
    # Remove test user directory
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
        print(f"   ✅ Removed: {TEST_DIR}")
    
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
    
    print("✅ Cleanup complete")


# ==================== TESTS ====================
class MockFile:
    """Mock uploaded file"""
    def __init__(self, path):
        self.name = path.name
        self.size = path.stat().st_size
        self._path = path
    
    def getbuffer(self):
        return self._path.read_bytes()


def test_imports():
    """Test 1: Imports"""
    print("\n🧪 Test 1: Imports")
    try:
        from backend.core.chunking import process_document_to_chunks
        from backend.core.extraction import extract_entities_relations
        from backend.core.graph_builder import build_knowledge_graph
        from backend.core.embedding import generate_embeddings, VectorDatabase
        from backend.core.pipeline import DocumentPipeline
        print("✅ All imports OK")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_chunking(files):
    """Test 2: Chunking"""
    print("\n🧪 Test 2: Chunking")
    try:
        from backend.core.chunking import process_document_to_chunks, ChunkConfig
        
        all_chunks = []
        for f in files:
            chunks = process_document_to_chunks(str(f), ChunkConfig(max_tokens=200, overlap_tokens=30))
            assert len(chunks) > 0
            assert all('chunk_id' in c for c in chunks)
            all_chunks.extend(chunks)
        
        print(f"✅ Created {len(all_chunks)} chunks")
        return all_chunks
    except Exception as e:
        print(f"❌ Chunking failed: {e}")
        return []


def test_extraction(chunks):
    """Test 3: Extraction"""
    print("\n🧪 Test 3: Extraction")
    
    setup_mock()
    
    try:
        from backend.core.extraction import extract_entities_relations
        
        entities, relationships = extract_entities_relations(chunks)
        
        ent_count = sum(len(v) for v in entities.values())
        rel_count = sum(len(v) for v in relationships.values())
        
        print(f"✅ Extracted {ent_count} entities, {rel_count} relationships")
        return entities, relationships
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return {}, {}
    finally:
        restore_mock()


def test_graph(entities, relationships):
    """Test 4: Graph Building"""
    print("\n🧪 Test 4: Graph Building")
    try:
        from backend.core.graph_builder import build_knowledge_graph
        
        kg = build_knowledge_graph(entities, relationships, enable_summarization=False)
        stats = kg.get_statistics()
        
        assert stats['num_entities'] > 0
        
        print(f"✅ Built graph: {stats['num_entities']} nodes, {stats['num_relationships']} edges")
        return kg
    except Exception as e:
        print(f"❌ Graph failed: {e}")
        return None


def test_embedding(chunks):
    """Test 5: Embeddings"""
    print("\n🧪 Test 5: Embeddings")
    try:
        from backend.core.embedding import generate_embeddings, VectorDatabase
        
        # Generate embeddings
        embeddings = generate_embeddings(chunks, batch_size=32, use_cache=False)
        assert len(embeddings) == len(chunks)
        assert all(len(e['embedding']) == 384 for e in embeddings)
        
        # Create vector DB
        db = VectorDatabase("test.index", "test_meta.json", dim=384, use_hnsw=True)
        db.add_embeddings(embeddings)
        
        # Test search
        results = db.search(embeddings[0]['embedding'], top_k=3)
        assert len(results) == 3
        
        db.save()
        
        print(f"✅ Created {len(embeddings)} embeddings, HNSW index working")
        return True
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        return False


def test_pipeline(files):
    """Test 6: Full Pipeline"""
    print("\n🧪 Test 6: Full Pipeline")
    
    setup_mock()
    
    try:
        from backend.core.pipeline import DocumentPipeline, DocChunkConfig
        
        pipeline = DocumentPipeline(user_id=TEST_USER, enable_advanced=True)
        mock_file = MockFile(files[0])
        
        result = pipeline.process_uploaded_file(
            uploaded_file=mock_file,
            chunk_config=DocChunkConfig(max_tokens=200, overlap_tokens=30),
            enable_extraction=True,
            enable_graph=True,
            enable_embedding=True,
            enable_gleaning=False
        )
        
        assert result['success']
        assert result['chunks_count'] > 0
        
        print(f"✅ Pipeline: {result['chunks_count']} chunks, "
              f"{result.get('entities_count', 0)} entities, "
              f"{result.get('graph_nodes', 0)} nodes")
        return True
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        restore_mock()


def test_persistence():
    """Test 7: Persistence"""
    print("\n🧪 Test 7: Persistence")
    try:
        from backend.core.pipeline import DocumentPipeline
        
        pipeline = DocumentPipeline(user_id=TEST_USER)
        docs = pipeline.get_processed_docs()
        
        assert len(docs) > 0
        
        print(f"✅ Found {len(docs)} persisted documents")
        return True
    except Exception as e:
        print(f"❌ Persistence failed: {e}")
        return False


# ==================== MAIN ====================
def main():
    """Run all tests"""
    
    print("="*60)
    print("🚀 LIGHTRAG INTEGRATION TEST")
    print("="*60)
    print(f"⏰ Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"📁 Test dir: {TEST_DIR}")
    
    start_time = time.time()
    passed = 0
    failed = 0
    
    try:
        # Setup
        print("\n📁 Creating test files...")
        files = create_test_files()
        print(f"✅ Created {len(files)} test files")
        
        # Apply nest_asyncio
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass
        
        # Run tests
        tests = []
        
        # Test 1: Imports
        if test_imports():
            passed += 1
        else:
            failed += 1
            print("\n❌ Cannot continue without imports")
            return 1
        
        # Test 2: Chunking
        chunks = test_chunking(files)
        if chunks:
            passed += 1
        else:
            failed += 1
        
        # Test 3: Extraction
        if chunks:
            entities, relationships = test_extraction(chunks)
            if entities or relationships:
                passed += 1
            else:
                failed += 1
        
        # Test 4: Graph
        if chunks:
            kg = test_graph(entities if entities else {}, relationships if relationships else {})
            if kg:
                passed += 1
            else:
                failed += 1
        
        # Test 5: Embedding
        if chunks:
            if test_embedding(chunks):
                passed += 1
            else:
                failed += 1
        
        # Test 6: Full Pipeline
        if test_pipeline(files):
            passed += 1
        else:
            failed += 1
        
        # Test 7: Persistence
        if test_persistence():
            passed += 1
        else:
            failed += 1
        
        # Summary
        duration = time.time() - start_time
        total = passed + failed
        
        print("\n" + "="*60)
        print("📊 SUMMARY")
        print("="*60)
        print(f"Total: {total}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"⏱️ Duration: {duration:.2f}s")
        print(f"📈 Success Rate: {(passed/total*100):.1f}%")
        print("="*60)
        
        success = failed == 0
        
        if success:
            print("\n🎉 ALL TESTS PASSED!")
        else:
            print(f"\n⚠️ {failed} TEST(S) FAILED")
        
        return 0 if success else 1
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        return 130
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Cleanup
        if CLEANUP:
            cleanup()
        else:
            print(f"\n⏭️ Keeping test data: {TEST_DIR}")


if __name__ == "__main__":
    sys.exit(main())