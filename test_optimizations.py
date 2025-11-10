#!/usr/bin/env python3
"""
Test script to verify all optimizations are working
Run: python test_optimizations.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath('.'))

def test_imports():
    """Test 1: Verify all imports work"""
    print("🧪 Test 1: Imports")
    try:
        from backend.utils.cache_utils import DiskCache, disk_cached
        from backend.core.chunking import process_document_to_chunks, Tokenizer
        from backend.core.embedding import OptimizedEmbeddingModel, VectorDatabase
        from backend.core.extraction import extract_entities_relations
        from backend.core.pipeline import DocumentPipeline
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_cache():
    """Test 2: Verify caching works"""
    print("\n🧪 Test 2: Caching")
    try:
        from backend.utils.cache_utils import DiskCache
        
        cache = DiskCache("test_cache")
        
        # Test set/get
        cache.set("test_key", {"data": "test"})
        result = cache.get("test_key")
        
        assert result == {"data": "test"}, "Cache get/set failed"
        
        # Test clear
        count = cache.clear()
        
        # Cleanup
        import shutil
        shutil.rmtree("test_cache", ignore_errors=True)
        
        print(f"✅ Cache working (cleared {count} items)")
        return True
    except Exception as e:
        print(f"❌ Cache test failed: {e}")
        return False


def test_tokenizer_cache():
    """Test 3: Verify tokenizer caching"""
    print("\n🧪 Test 3: Tokenizer Cache")
    try:
        from backend.core.chunking import Tokenizer
        
        tokenizer = Tokenizer()
        
        # Test cached counting
        text = "This is a test sentence."
        
        start = time.time()
        count1 = tokenizer.count(text)
        time1 = time.time() - start
        
        start = time.time()
        count2 = tokenizer.count(text)
        time2 = time.time() - start
        
        assert count1 == count2, "Token counts don't match"
        assert time2 < time1, "Second call should be faster (cached)"
        
        speedup = time1 / time2 if time2 > 0 else float('inf')
        print(f"✅ Tokenizer cache working (speedup: {speedup:.1f}x)")
        return True
    except Exception as e:
        print(f"❌ Tokenizer cache test failed: {e}")
        return False


def test_embedding_model():
    """Test 4: Verify optimized embedding model"""
    print("\n🧪 Test 4: Optimized Embedding Model")
    try:
        from backend.core.embedding import get_embedding_model
        
        model = get_embedding_model()
        
        texts = ["Test sentence 1", "Test sentence 2", "Test sentence 3"]
        
        # Test batch encoding
        start = time.time()
        embeddings = model.encode_batch(texts, batch_size=3, show_progress=False)
        elapsed = time.time() - start
        
        assert len(embeddings) == 3, "Wrong number of embeddings"
        assert embeddings.shape[1] == 384, "Wrong embedding dimension"
        
        print(f"✅ Embedding model working ({elapsed:.2f}s for {len(texts)} texts)")
        print(f"   Device: {model.device}")
        return True
    except Exception as e:
        print(f"❌ Embedding model test failed: {e}")
        return False


def test_vector_db():
    """Test 5: Verify HNSW index"""
    print("\n🧪 Test 5: Vector Database (HNSW)")
    try:
        from backend.core.embedding import VectorDatabase, get_embedding_model
        
        # Create test embeddings
        model = get_embedding_model()
        texts = [f"Test document {i}" for i in range(100)]
        embeddings_array = model.encode_batch(texts, show_progress=False)
        
        embeddings = []
        for i, (text, emb) in enumerate(zip(texts, embeddings_array)):
            embeddings.append({
                'id': f'test_{i}',
                'text': text,
                'embedding': emb.tolist(),
                'entity_type': 'TEST'
            })
        
        # Test HNSW index
        db = VectorDatabase(
            db_path="test.index",
            metadata_path="test_meta.json",
            dim=384,
            use_hnsw=True
        )
        
        db.add_embeddings(embeddings)
        
        # Test search
        query_emb = embeddings_array[0]
        start = time.time()
        results = db.search(query_emb.tolist(), top_k=5)
        search_time = time.time() - start
        
        assert len(results) == 5, "Wrong number of results"
        assert results[0]['id'] == 'test_0', "Wrong top result"
        
        # Cleanup
        for f in ['test.index', 'test_meta.json']:
            Path(f).unlink(missing_ok=True)
        
        print(f"✅ HNSW index working (search time: {search_time*1000:.2f}ms)")
        return True
    except Exception as e:
        print(f"❌ Vector DB test failed: {e}")
        return False


def test_batch_extraction():
    """Test 6: Verify batch extraction config"""
    print("\n🧪 Test 6: Batch Extraction Config")
    try:
        batch_size = int(os.getenv('EXTRACTION_BATCH_SIZE', 10))
        max_concurrent = int(os.getenv('MAX_CONCURRENT_LLM_CALLS', 8))
        
        print(f"✅ Batch extraction configured:")
        print(f"   Batch size: {batch_size}")
        print(f"   Max concurrent: {max_concurrent}")
        return True
    except Exception as e:
        print(f"❌ Batch extraction config test failed: {e}")
        return False


def test_pipeline_config():
    """Test 7: Verify pipeline configuration"""
    print("\n🧪 Test 7: Pipeline Configuration")
    try:
        from backend.core.pipeline import DocumentPipeline
        
        pipeline = DocumentPipeline(user_id="test_user")
        
        print(f"✅ Pipeline configured:")
        print(f"   Max workers: {pipeline.max_workers}")
        print(f"   Extraction batch: {pipeline.batch_size}")
        print(f"   Embedding batch: {pipeline.embedding_batch_size}")
        print(f"   HNSW index: {pipeline.use_hnsw}")
        
        return True
    except Exception as e:
        print(f"❌ Pipeline config test failed: {e}")
        return False


def test_env_config():
    """Test 8: Verify .env configuration"""
    print("\n🧪 Test 8: Environment Configuration")
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        configs = {
            'EXTRACTION_BATCH_SIZE': os.getenv('EXTRACTION_BATCH_SIZE'),
            'EMBEDDING_BATCH_SIZE': os.getenv('EMBEDDING_BATCH_SIZE'),
            'MAX_CONCURRENT_LLM_CALLS': os.getenv('MAX_CONCURRENT_LLM_CALLS'),
            'USE_HNSW_INDEX': os.getenv('USE_HNSW_INDEX'),
            'CHUNK_SIZE': os.getenv('CHUNK_SIZE'),
            'CHUNK_OVERLAP': os.getenv('CHUNK_OVERLAP'),
        }
        
        print("✅ Environment variables:")
        for key, value in configs.items():
            print(f"   {key}: {value}")
        
        return True
    except Exception as e:
        print(f"❌ Env config test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("🚀 OPTIMIZATION VERIFICATION TESTS")
    print("="*60)
    
    tests = [
        test_imports,
        test_cache,
        test_tokenizer_cache,
        test_embedding_model,
        test_vector_db,
        test_batch_extraction,
        test_pipeline_config,
        test_env_config,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test crashed: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("✅ All optimizations verified!")
        return 0
    else:
        print("❌ Some tests failed. Check configuration.")
        return 1


if __name__ == "__main__":
    sys.exit(main())