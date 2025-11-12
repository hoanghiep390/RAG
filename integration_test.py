"""
✅ FIXED: Integration Test with better error handling
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

TEST_USER = "test_integration"
TEST_DIR = Path("backend/data") / TEST_USER
CLEANUP = True

async def mock_llm(*args, **kwargs):
    """Mock LLM with CORRECT format"""
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
        print("✅ Mock LLM installed")
    except Exception as e:
        print(f"⚠️ Could not patch LLM: {e}")

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

def create_test_files():
    """Create test files"""
    upload_dir = TEST_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    (upload_dir / "test.txt").write_text("""
LightRAG combines knowledge graphs with vector embeddings.
It uses FAISS for fast similarity search.
The system extracts entities and builds graphs automatically.
""", encoding='utf-8')
    
    return list(upload_dir.glob("*"))

def cleanup():
    """Remove all test data"""
    print("\n🗑️ Cleaning up...")
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
        print(f"   ✅ Removed: {TEST_DIR}")
    
    for pattern in ["*.index", "*_meta.json", "test_*"]:
        for f in Path(".").glob(pattern):
            try:
                if f.is_file():
                    f.unlink()
                elif f.is_dir():
                    shutil.rmtree(f)
            except:
                pass
    
    cache_dir = Path("backend/data/cache")
    if cache_dir.exists():
        for subdir in cache_dir.glob("*"):
            if subdir.is_dir():
                shutil.rmtree(subdir, ignore_errors=True)
    
    print("✅ Cleanup complete")

from io import BytesIO

class MockFile:
    """Mock Streamlit uploaded file"""
    def __init__(self, path):
        self.name = path.name
        self._data = path.read_bytes()

    def getbuffer(self):
        return memoryview(self._data)

def test_imports():
    """Test 1: Imports"""
    print("\n📦 Test 1: Imports")
    try:
        from backend.core.chunking import process_document_to_chunks
        from backend.core.extraction import extract_entities_relations
        from backend.core.graph_builder import build_knowledge_graph
        from backend.core.embedding import generate_embeddings, VectorDatabase
        from backend.core.pipeline import DocumentPipeline
        print("   ✅ All imports OK")
        return True
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_chunking(files):
    """Test 2: Chunking"""
    print("\n📄 Test 2: Chunking")
    try:
        from backend.core.chunking import process_document_to_chunks, ChunkConfig
        
        all_chunks = []
        for f in files:
            chunks = process_document_to_chunks(str(f), ChunkConfig(max_tokens=200, overlap_tokens=30))
            assert len(chunks) > 0, f"No chunks from {f}"
            assert all('chunk_id' in c for c in chunks), "Missing chunk_id"
            all_chunks.extend(chunks)
        
        print(f"   ✅ Created {len(all_chunks)} chunks total")
        return all_chunks
    except Exception as e:
        print(f"   ❌ Chunking failed: {e}")
        traceback.print_exc()
        return []

def test_extraction(chunks):
    """Test 3: Extraction"""
    print("\n🔍 Test 3: Extraction")
    
    if not chunks:
        print("   ❌ No chunks to extract from")
        return {}, {}
    
    setup_mock()
    
    try:
        from backend.core.extraction import extract_entities_relations
        
        entities, relationships = extract_entities_relations(chunks)
        
        ent_count = sum(len(v) for v in entities.values())
        rel_count = sum(len(v) for v in relationships.values())
        
        assert ent_count > 0, "No entities extracted"
        assert rel_count > 0, "No relationships extracted"
        
        print(f"   ✅ Extracted {ent_count} entities, {rel_count} relationships")
        return entities, relationships
    except Exception as e:
        print(f"   ❌ Extraction failed: {e}")
        traceback.print_exc()
        return {}, {}
    finally:
        restore_mock()

def test_graph(entities, relationships):
    """Test 4: Graph Building"""
    print("\n🕸️ Test 4: Graph Building")
    try:
        from backend.core.graph_builder import build_knowledge_graph
        
        kg = build_knowledge_graph(entities, relationships, enable_summarization=False)
        stats = kg.get_statistics()
        
        assert stats['num_entities'] >= 2, f"Expected >=2 nodes, got {stats['num_entities']}"
        assert stats['num_relationships'] >= 1, f"Expected >=1 edge, got {stats['num_relationships']}"
        
        print(f"   ✅ Built graph: {stats['num_entities']} nodes, {stats['num_relationships']} edges")
        return kg
    except Exception as e:
        print(f"   ❌ Graph failed: {e}")
        traceback.print_exc()
        return None

def test_embedding(chunks):
    """Test 5: Embeddings"""
    print("\n🧮 Test 5: Embeddings")
    try:
        from backend.core.embedding import generate_embeddings, VectorDatabase
        
        embeddings = generate_embeddings(chunks, batch_size=32, use_cache=False)
        assert len(embeddings) == len(chunks)
        assert all(len(e['embedding']) == 384 for e in embeddings), "Invalid embedding dim"
        
        db = VectorDatabase("test.index", "test_meta.json", dim=384, use_hnsw=True)
        db.add_embeddings(embeddings)
        
        top_k = min(3, len(embeddings))
        results = db.search(embeddings[0]['embedding'], top_k=top_k)
        assert len(results) == top_k, f"Expected {top_k} results, got {len(results)}"
        
        db.save()
        
        print(f"   ✅ Created {len(embeddings)} embeddings, search top_k={top_k} OK")
        return True
    except Exception as e:
        print(f"   ❌ Embedding failed: {e}")
        traceback.print_exc()
        return False

def test_pipeline(files):
    """Test 6: Full Pipeline"""
    print("\n🔄 Test 6: Full Pipeline")
    
    setup_mock()
    
    try:
        from backend.core.pipeline import DocumentPipeline, DocChunkConfig

        pipeline = DocumentPipeline(user_id=TEST_USER, enable_advanced=True)
        
        original_file = files[0]
        mock_file = MockFile(original_file)
        
        result = pipeline.process_uploaded_file(
            uploaded_file=mock_file,
            chunk_config=DocChunkConfig(max_tokens=200, overlap_tokens=30),
            enable_extraction=True,
            enable_graph=True,
            enable_embedding=True,
            enable_gleaning=False
        )
        
        assert result['success'], f"Pipeline failed: {result.get('error')}"
        assert result['chunks_count'] > 0
        assert result.get('entities_count', 0) > 0
        assert result.get('graph_nodes', 0) > 0
        
        print(f"   ✅ Pipeline: {result['chunks_count']} chunks, "
              f"{result.get('entities_count', 0)} entities, "
              f"{result.get('graph_nodes', 0)} nodes")
        return True
    except Exception as e:
        print(f"   ❌ Pipeline failed: {e}")
        traceback.print_exc()
        return False
    finally:
        restore_mock()

def test_persistence():
    """✅ FIXED: Test 7 with better debugging"""
    print("\n💾 Test 7: Persistence")
    try:
        from backend.core.pipeline import DocumentPipeline
        
        pipeline = DocumentPipeline(user_id=TEST_USER)
        docs = pipeline.get_processed_docs()
        
        assert len(docs) > 0, "No documents persisted"
        
        doc = docs[0]
        
        # Debug info
        print(f"   📄 Doc: {doc['file']}")
        print(f"   📊 Chunks: {doc['chunks']}")
        print(f"   🕸️ Has graph: {doc['has_graph']}")
        print(f"   🧮 Has embeddings: {doc['has_embeddings']}")
        
        # ✅ FIX: Check actual files
        chunks_dir = TEST_DIR / "chunks"
        graphs_dir = TEST_DIR / "graphs"
        vectors_dir = TEST_DIR / "vectors"
        
        print(f"   📂 Chunks dir: {chunks_dir.exists()}")
        print(f"   📂 Graphs dir: {graphs_dir.exists()}")
        print(f"   📂 Vectors dir: {vectors_dir.exists()}")
        
        if graphs_dir.exists():
            graph_files = list(graphs_dir.glob("*_graph.json"))
            print(f"   📝 Graph files: {[f.name for f in graph_files]}")
        
        if vectors_dir.exists():
            vector_files = list(vectors_dir.glob("*.index"))
            print(f"   📝 Vector files: {[f.name for f in vector_files]}")
        
        assert doc['chunks'] > 0, "No chunks"
        assert doc['has_graph'], "Graph file not found"
        assert doc['has_embeddings'], "Embedding files not found"
        
        print(f"   ✅ Found {len(docs)} persisted documents with graph & embeddings")
        return True
    except Exception as e:
        print(f"   ❌ Persistence failed: {e}")
        traceback.print_exc()
        return False

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
        print("\n📁 Creating test files...")
        files = create_test_files()
        print(f"✅ Created {len(files)} test files")
        
        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass
        
        tests = [
            ("Imports", lambda: test_imports()),
            ("Chunking", lambda: test_chunking(files)),
            ("Extraction", lambda: test_extraction(test_chunking(files))),
            ("Graph", lambda: test_graph(*test_extraction(test_chunking(files)))),
            ("Embedding", lambda: test_embedding(test_chunking(files))),
            ("Pipeline", lambda: test_pipeline(files)),
            ("Persistence", lambda: test_persistence()),
        ]
        
        results = []
        for name, test_func in tests:
            result = test_func()
            if result:
                passed += 1
                results.append(f"✅ {name}")
            else:
                failed += 1
                results.append(f"❌ {name}")
        
        duration = time.time() - start_time
        total = passed + failed
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        for r in results:
            print(r)
        print("-"*60)
        print(f"Total: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Duration: {duration:.2f}s")
        print(f"Success Rate: {(passed/total*100):.1f}%")
        print("="*60)
        
        if failed == 0:
            print("\n🎉 ALL TESTS PASSED!")
            return 0
        else:
            print(f"\n❌ {failed} TEST(S) FAILED")
            return 1
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        traceback.print_exc()
        return 1
    finally:
        if CLEANUP:
            cleanup()
        else:
            print(f"\n📂 Keeping test data: {TEST_DIR}")

if __name__ == "__main__":
    sys.exit(main())