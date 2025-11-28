# check_system.py
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

print("=" * 60)
print("🔍 SYSTEM CHECK")
print("=" * 60)

# 1. Check files
print("\n1️⃣ Checking files...")
files = ['frontend/login.py', 'frontend/pages/upload.py', 'frontend/pages/graph.py', 'frontend/pages/chat.py']
for f in files:
    exists = "✅" if os.path.exists(f) else "❌"
    print(f"   {exists} {f}")

# 2. Check VectorDB
print("\n2️⃣ Checking VectorDB...")
try:
    from backend.db.vector_db import VectorDatabase
    vector_db = VectorDatabase('admin_00000000')
    stats = vector_db.get_statistics()
    print(f"   ✅ Total vectors: {stats['total_vectors']}")
    print(f"   ✅ Active vectors: {stats['active_vectors']}")
    print(f"   ✅ Documents: {stats['total_documents']}")
    
    if stats['active_vectors'] == 0:
        print("   ⚠️  WARNING: No vectors! Please upload documents.")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 3. Check MongoDB
print("\n3️⃣ Checking MongoDB...")
try:
    from backend.db.mongo_storage import MongoStorage
    storage = MongoStorage('admin_00000000')
    docs = storage.list_documents()
    print(f"   ✅ Documents in MongoDB: {len(docs)}")
    
    if len(docs) == 0:
        print("   ⚠️  WARNING: No documents! Please upload.")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 4. Check Retriever
print("\n4️⃣ Checking Retriever...")
try:
    from backend.retrieval.hybrid_retriever import HybridRetriever
    retriever = HybridRetriever(vector_db, storage)
    context = retriever.retrieve("test")
    print(f"   ✅ Retrieval works: {len(context.chunks)} chunks, {len(context.entities)} entities")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 5. Check LLM
print("\n5️⃣ Checking LLM...")
try:
    from backend.utils.llm_utils import call_llm
    response = call_llm("Say hello", max_tokens=10)
    print(f"   ✅ LLM works: {response[:50]}...")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("✅ Check completed!")
print("=" * 60)