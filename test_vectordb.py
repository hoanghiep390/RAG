#!/usr/bin/env python3
# test_vectordb.py - Comprehensive VectorDB diagnostic

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.abspath('.'))

print("=" * 70)
print("🔍 VECTORDB DIAGNOSTIC TEST")
print("=" * 70)

# Test 1: Check FAISS availability
print("\n1️⃣ Checking FAISS availability...")
try:
    import faiss
    print(f"   ✅ FAISS version: {faiss.__version__}")
    print(f"   ✅ FAISS path: {faiss.__file__}")
    
    # Test basic functionality
    test_index = faiss.IndexFlatL2(384)
    import numpy as np
    test_vec = np.random.rand(10, 384).astype('float32')
    test_index.add(test_vec)
    print(f"   ✅ FAISS works: added {test_index.ntotal} vectors")
    
except ImportError as e:
    print(f"   ❌ FAISS not available: {e}")
    print(f"   💡 Install: pip install faiss-cpu")
except Exception as e:
    print(f"   ⚠️ FAISS error: {e}")

# Test 2: Check Config
print("\n2️⃣ Checking Config...")
try:
    from backend.config import Config
    print(f"   ✅ EMBEDDING_DIM: {Config.EMBEDDING_DIM}")
    print(f"   ✅ USE_HNSW: {Config.USE_HNSW}")
    print(f"   ✅ HNSW_M: {Config.HNSW_M}")
    print(f"   ✅ DATA_DIR: {Config.DATA_DIR}")
    
    # Test user dir creation
    test_user_id = 'test_user'
    vector_dir = Config.get_user_vector_dir(test_user_id)
    print(f"   ✅ Vector dir: {vector_dir}")
    print(f"   ✅ Dir exists: {vector_dir.exists()}")
    
except Exception as e:
    print(f"   ❌ Config error: {e}")

# Test 3: Check VectorDatabase initialization
print("\n3️⃣ Testing VectorDatabase initialization...")
try:
    from backend.db.vector_db import VectorDatabase, FAISS_AVAILABLE
    
    print(f"   📊 FAISS_AVAILABLE: {FAISS_AVAILABLE}")
    
    test_user = 'admin_00000000'
    vector_db = VectorDatabase(test_user)
    
    print(f"   ✅ VectorDB created for: {test_user}")
    print(f"   ✅ Base dir: {vector_db.base_dir}")
    print(f"   ✅ Index path: {vector_db.index_path}")
    print(f"   ✅ Index path exists: {vector_db.index_path.exists()}")
    
    stats = vector_db.get_statistics()
    print(f"\n   📈 Statistics:")
    for key, value in stats.items():
        print(f"      • {key}: {value}")
    
except Exception as e:
    import traceback
    print(f"   ❌ VectorDB error: {e}")
    traceback.print_exc()

# Test 4: Check file structure
print("\n4️⃣ Checking file structure...")
try:
    base_path = Path('backend/data/admin_00000000/vectors')
    
    files_to_check = [
        'combined.index',
        'combined_metadata.json',
        'document_map.json',
        'combined_index_metadata.json'
    ]
    
    for fname in files_to_check:
        fpath = base_path / fname
        if fpath.exists():
            size = fpath.stat().st_size
            print(f"   ✅ {fname}: {size} bytes")
        else:
            print(f"   ⚠️ {fname}: NOT FOUND")
    
except Exception as e:
    print(f"   ❌ File check error: {e}")

# Test 5: Try add/search operation
print("\n5️⃣ Testing add/search operations...")
try:
    from backend.db.vector_db import VectorDatabase
    from backend.core.embedding import get_model
    
    vector_db = VectorDatabase('test_diagnostic', auto_save=True)
    model = get_model()
    
    # Create test embeddings
    test_texts = ["This is a test", "Another test document"]
    embeddings_raw = model.encode(test_texts, normalize_embeddings=True)
    
    embeddings = [
        {
            'id': f'test_{i}',
            'text': text,
            'embedding': emb.tolist(),
            'tokens': len(text.split()),
            'order': i,
            'file_path': 'test.txt',
            'file_type': 'TXT'
        }
        for i, (text, emb) in enumerate(zip(test_texts, embeddings_raw))
    ]
    
    # Add to VectorDB
    count = vector_db.add_document_embeddings_batch(
        doc_id='test_doc_123',
        filename='test.txt',
        embeddings=embeddings
    )
    
    print(f"   ✅ Added {count} embeddings")
    
    # Test search
    query_emb = model.encode(["test query"], normalize_embeddings=True)[0]
    results = vector_db.search(query_emb.tolist(), top_k=2)
    
    print(f"   ✅ Search returned {len(results)} results")
    for i, r in enumerate(results):
        print(f"      [{i+1}] Score: {r['similarity']:.3f} - {r['content'][:50]}")
    
    # Check save
    print(f"   💾 Index saved: {vector_db.index_path.exists()}")
    
    # Cleanup
    import shutil
    cleanup_dir = Path('backend/data/test_diagnostic')
    if cleanup_dir.exists():
        shutil.rmtree(cleanup_dir)
        print(f"   🗑️ Cleaned up test data")
    
except Exception as e:
    import traceback
    print(f"   ❌ Add/search error: {e}")
    traceback.print_exc()

# Test 6: Check MongoDB
print("\n6️⃣ Checking MongoDB...")
try:
    from backend.db.mongo_storage import MongoStorage
    
    storage = MongoStorage('admin_00000000')
    docs = storage.list_documents()
    
    print(f"   ✅ MongoDB connected")
    print(f"   ✅ Documents: {len(docs)}")
    
    if docs:
        print(f"   📄 Latest: {docs[0]['filename']}")
    
except Exception as e:
    print(f"   ❌ MongoDB error: {e}")

# Summary
print("\n" + "=" * 70)
print("✅ DIAGNOSTIC COMPLETE")
print("=" * 70)

print("\n💡 If you see errors:")
print("   1. Install FAISS: pip install faiss-cpu")
print("   2. Check MongoDB is running: mongod")
print("   3. Verify .env configuration")
print("   4. Check file permissions in backend/data/")
print("   5. Try deleting backend/data/*/vectors/ and re-upload")