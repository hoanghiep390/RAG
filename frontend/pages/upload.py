# frontend/pages/upload.py
"""
Upload Page - MongoDB + FAISS Storage
- MongoDB: Documents, chunks, entities, relationships, graph
- FAISS: Vector embeddings for semantic search
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.core.pipeline import DocumentPipeline
from backend.core.chunking import DocChunkConfig
from backend.db.mongo_storage import MongoStorage
from backend.db.vector_db import VectorDatabase

# Auth check
if not st.session_state.get('authenticated', False):
    st.switch_page("login.py")

if st.session_state.get('role') != 'admin':
    st.error("⛔ Chỉ **Admin** được phép truy cập trang này.")
    if st.button("🏠 Quay lại Login", use_container_width=True): 
        st.switch_page("login.py")
    st.stop()

user_id = st.session_state.get('user_id', 'admin_00000000')
username = st.session_state.get('username', 'Admin')
st.set_page_config(page_title="LightRAG | Upload", page_icon="📤", layout="wide")

# CSS
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .header-container { 
        background: linear-gradient(90deg, #1e1e1e 0%, #2d2d2d 100%); 
        padding: 1.5rem; 
        border-radius: 10px; 
        margin-bottom: 2rem; 
        border-left: 5px solid #10b981; 
    }
    .header-title { color: #10b981; font-size: 2rem; font-weight: 700; margin: 0; }
    .badge { 
        padding: 0.3rem 0.8rem; 
        border-radius: 12px; 
        font-size: 0.8rem; 
        font-weight: 600; 
    }
    .badge-mongo { background: #dc2626; color: white; }
    .badge-faiss { background: #10b981; color: white; }
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown(f"""
<div class="header-container">
    <div class="header-title">
        📤 Upload Document 
        <span class="badge badge-mongo">MONGODB</span>
        <span class="badge badge-faiss">FAISS</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 👤 Admin Info")
    st.markdown(f"**User**: {username}<br>**ID**: `{user_id}`", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 🧭 Navigation")
    if st.button("🕸️ Knowledge Graph", use_container_width=True):
        st.switch_page("pages/graph.py")
    
    st.markdown("---")
    
    if st.button("🚪 Logout", use_container_width=True, type="secondary"):
        for k in ['authenticated', 'user_id', 'username', 'role']:
            st.session_state.pop(k, None)
        st.switch_page("login.py")

# Initialize storage
try:
    mongo_storage = MongoStorage(user_id)
    vector_db = VectorDatabase(user_id, dim=384, use_hnsw=True)
    pipeline = DocumentPipeline(user_id)
except Exception as e:
    st.error(f"❌ Failed to initialize storage: {e}")
    st.info("💡 Make sure MongoDB is running: `mongod`")
    st.stop()

# Upload section
st.markdown("### 📁 Upload Tài liệu")

st.markdown("""
<div class="info-card">
    <strong>📋 Lưu trữ:</strong><br>
    • 🗄️ <strong>MongoDB</strong>: Documents, chunks, entities, relationships, graph<br>
    • 🚀 <strong>FAISS</strong>: Vector embeddings (semantic search)<br>
    • 📄 Định dạng: PDF, DOCX, TXT, MD, CSV, JSON, XML<br>
    • 📏 Max size: 50MB per file
</div>
""", unsafe_allow_html=True)

# File uploader
uploaded_files = st.file_uploader(
    "Chọn file để upload",
    type=['pdf', 'docx', 'txt', 'md', 'csv', 'xlsx', 'html', 'py', 'js', 'java', 'cpp', 'json', 'xml'],
    accept_multiple_files=True,
    help="Có thể chọn nhiều file cùng lúc"
)

if uploaded_files:
    st.markdown(f"**Đã chọn {len(uploaded_files)} file:**")
    for f in uploaded_files:
        file_size = f.size / 1024 / 1024
        st.markdown(f"- 📄 {f.name} ({file_size:.2f} MB)")

# Configuration
st.markdown("---")
st.markdown("### ⚙️ Cấu hình xử lý")

col1, col2 = st.columns(2)

with col1:
    chunk_size = st.slider("📏 Chunk Size (tokens)", 100, 1000, 300, 50)

with col2:
    chunk_overlap = st.slider("🔄 Overlap (tokens)", 0, 200, 50, 10)

with st.expander("🔧 Tùy chọn nâng cao", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        enable_extraction = st.checkbox("📊 Entity Extraction", value=True)
    
    with col2:
        enable_graph = st.checkbox("🕸️ Knowledge Graph", value=True)
    
    with col3:
        enable_embedding = st.checkbox("🧮 Vector Embedding", value=True)

# Process button
st.markdown("---")
if uploaded_files:
    if st.button("🚀 Bắt đầu xử lý", type="primary", use_container_width=True):
        MAX_FILE_SIZE = 50 * 1024 * 1024
        
        # Validate
        invalid_files = [f"{f.name} ({f.size/1024/1024:.1f}MB)" 
                        for f in uploaded_files if f.size > MAX_FILE_SIZE]
        
        if invalid_files:
            st.error("❌ **File quá lớn!**")
            for fname in invalid_files:
                st.markdown(f"- {fname}")
            st.stop()
        
        # Process files
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        success_count = 0
        failed_count = 0
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"⏳ Đang xử lý [{i+1}/{len(uploaded_files)}]: {file.name}")
            
            try:
                # Process with pipeline
                result = pipeline.process_file(
                    file,
                    chunk_config=DocChunkConfig(
                        max_tokens=chunk_size,
                        overlap_tokens=chunk_overlap
                    ),
                    enable_extraction=enable_extraction,
                    enable_graph=enable_graph,
                    enable_embedding=enable_embedding
                )
                
                if result['success']:
                    doc_id = result['doc_id']
                    
                    # ✅ Save to MongoDB
                    mongo_storage.save_document(
                        doc_id=doc_id,
                        filename=result['filename'],
                        filepath=result['filepath'],
                        metadata={'original_size': file.size}
                    )
                    
                    mongo_storage.save_chunks(doc_id, result['chunks'])
                    
                    if result.get('entities'):
                        mongo_storage.save_entities(doc_id, result['entities'])
                        mongo_storage.save_relationships(doc_id, result['relationships'])
                    
                    if result.get('graph'):
                        mongo_storage.save_graph(result['graph'])
                    
                    # ✅ Save embeddings to FAISS
                    if result.get('embeddings'):
                        vector_db.add_document_embeddings(
                            doc_id=doc_id,
                            filename=result['filename'],
                            embeddings=result['embeddings']
                        )
                        vector_db.save()
                    
                    # Update status
                    mongo_storage.update_document_status(doc_id, 'completed', result['stats'])
                    
                    success_count += 1
                    
                    with st.expander(f"✅ {file.name} - Thành công"):
                        st.json({
                            'Chunks': result['stats'].get('chunks_count', 0),
                            'Entities': result['stats'].get('entities_count', 0),
                            'Graph Nodes': result['stats'].get('graph_nodes', 0),
                            'Embeddings': result['stats'].get('embeddings_count', 0)
                        })
                else:
                    failed_count += 1
                    st.error(f"❌ {file.name}: {result.get('error')}")
                
            except Exception as e:
                failed_count += 1
                st.error(f"❌ {file.name}: {str(e)}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        progress_bar.empty()
        status_text.empty()
        
        # Summary
        if success_count > 0:
            st.markdown(f"""
            <div class="info-card">
                <strong>🎉 Hoàn thành!</strong><br>
                ✅ Thành công: {success_count} file<br>
                ❌ Thất bại: {failed_count} file<br>
                💾 MongoDB: Chunks, entities, graph<br>
                🚀 FAISS: Vector embeddings
            </div>
            """, unsafe_allow_html=True)
            
            st.rerun()

else:
    st.info("👆 Vui lòng chọn file để upload")

# Quick actions
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🕸️ Xem Graph", type="primary", use_container_width=True):
        st.switch_page("pages/graph.py")

with col2:
    if st.button("📊 Statistics", use_container_width=True):
        mongo_stats = mongo_storage.get_user_statistics()
        vector_stats = vector_db.get_statistics()
        
        st.markdown(f"""
        <div class="info-card">
            <strong>📈 Storage Statistics:</strong><br>
            <br><strong>MongoDB:</strong><br>
            • Documents: {mongo_stats['total_documents']}<br>
            • Chunks: {mongo_stats['total_chunks']}<br>
            • Entities: {mongo_stats['total_entities']}<br>
            • Relationships: {mongo_stats['total_relationships']}<br>
            • Graph: {mongo_stats['graph_nodes']} nodes, {mongo_stats['graph_edges']} edges<br>
            <br><strong>FAISS:</strong><br>
            • Vectors: {vector_stats['index_size']}<br>
            • Documents: {vector_stats['total_documents']}<br>
            • Type: {vector_stats['index_type']}
        </div>
        """, unsafe_allow_html=True)

with col3:
    if st.button("🔍 Test Search", use_container_width=True):
        if vector_db.get_statistics()['index_size'] > 0:
            query = st.text_input("Enter search query:", key="test_search")
            if query:
                results = vector_db.search_by_text(query, top_k=3)
                for r in results:
                    st.markdown(f"**Similarity: {r['similarity']:.3f}**")
                    st.text(r['content'][:200] + "...")
        else:
            st.info("No embeddings yet. Upload documents first.")

# Document list
st.markdown("---")
st.markdown("### 📚 Tài liệu đã xử lý")

try:
    docs = mongo_storage.list_documents()
    
    if docs:
        df_data = []
        for doc in docs:
            stats = doc.get('stats', {})
            df_data.append({
                'File': doc['filename'],
                'Status': doc.get('status', 'unknown'),
                'Chunks': stats.get('chunks_count', 0),
                'Entities': stats.get('entities_count', 0),
                'Graph Nodes': stats.get('graph_nodes', 0),
                'Embeddings': stats.get('embeddings_count', 0),
                'Uploaded': doc['uploaded_at'].strftime("%m/%d %H:%M")
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, height=400)
        
        # Delete section
        with st.expander("🗑️ Xóa tài liệu", expanded=False):
            doc_to_delete = st.selectbox(
                "Chọn tài liệu cần xóa",
                options=[doc['doc_id'] for doc in docs],
                format_func=lambda x: next((d['filename'] for d in docs if d['doc_id'] == x), x)
            )
            
            if st.button("🗑️ Xóa", type="secondary"):
                # Delete from MongoDB
                if mongo_storage.delete_document(doc_to_delete):
                    # Delete from FAISS
                    vector_db.delete_document(doc_to_delete)
                    vector_db.save()
                    
                    st.success(f"✅ Đã xóa tài liệu khỏi MongoDB & FAISS")
                    st.info("💡 Run rebuild_index() periodically to clean up FAISS")
                    st.rerun()
                else:
                    st.error(f"❌ Không thể xóa tài liệu")
    else:
        st.info("📭 Chưa có tài liệu nào")

except Exception as e:
    st.error(f"❌ Error: {e}")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#6b7280;'>"
    "📤 Upload <strong>MongoDB + FAISS</strong> – Đại học Thủy lợi"
    "</p>",
    unsafe_allow_html=True
)