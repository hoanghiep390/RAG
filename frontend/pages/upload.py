# frontend/pages/upload.py 

import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.core.pipeline import DocumentPipeline
from backend.core.chunking import ChunkConfig
from backend.db.mongo_storage import MongoStorage
from backend.db.vector_db import VectorDatabase
from backend.config import Config

# Auth check
if not st.session_state.get('authenticated', False):
    st.switch_page("login.py")

if st.session_state.get('role') != 'admin':
    st.error("⛔ Chỉ **Admin** được phép truy cập trang này.")
    if st.button("🏠 Quay lại Login"): 
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
    .badge-config { background: #3b82f6; color: white; }
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    .error-card {
        background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
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
        <span class="badge badge-config">AUTO-SAVE</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 👤 Admin Info")
    st.markdown(f"**User**: {username}<br>**ID**: `{user_id}`", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 🧭 Navigation")
    if st.button("💬 Chat"):
        st.switch_page("pages/chat.py")
    if st.button("🕸️ Knowledge Graph"):
        st.switch_page("pages/graph.py")
    
    st.markdown("---")
    
    # Config info
    with st.expander("⚙️ Current Config", expanded=False):
        st.markdown(f"""
        **LLM**: {Config.LLM_PROVIDER} / {Config.LLM_MODEL}  
        **Embedding**: {Config.EMBEDDING_MODEL} ({Config.EMBEDDING_DIM}D)  
        **FAISS**: {'HNSW' if Config.USE_HNSW else 'Flat'} (M={Config.HNSW_M})  
        **Batch**: LLM={Config.MAX_CONCURRENT_LLM_CALLS}, Embed={Config.EMBEDDING_BATCH_SIZE}  
        **Chunk**: {Config.DEFAULT_CHUNK_SIZE} tokens (overlap {Config.DEFAULT_CHUNK_OVERLAP})
        """)
    
    st.markdown("---")
    
    if st.button("🚪 Logout", type="secondary"):
        for k in ['authenticated', 'user_id', 'username', 'role']:
            st.session_state.pop(k, None)
        st.switch_page("login.py")

# Initialize storage with error handling
mongo_storage = None
vector_db = None
pipeline = None
init_errors = []

try:
    mongo_storage = MongoStorage(user_id)
    
    # Health check
    if not mongo_storage.health_check():
        init_errors.append("MongoDB connection unhealthy")
except Exception as e:
    init_errors.append(f"MongoDB init failed: {str(e)}")

try:
    vector_db = VectorDatabase(
        user_id, 
        dim=Config.EMBEDDING_DIM,
        use_hnsw=Config.USE_HNSW,
        auto_save=True
    )
except Exception as e:
    init_errors.append(f"FAISS init failed: {str(e)}")

# ✅ FIXED: Pass vector_db and mongo_storage to pipeline
if mongo_storage and vector_db:
    try:
        pipeline = DocumentPipeline(
            user_id=user_id,
            vector_db=vector_db,           # ✅ NEW: Pass VectorDB
            mongo_storage=mongo_storage    # ✅ NEW: Pass MongoDB
        )
    except Exception as e:
        init_errors.append(f"Pipeline init failed: {str(e)}")

# Display errors if any
if init_errors:
    st.markdown(f"""
    <div class="error-card">
        <strong>❌ Initialization Errors:</strong><br>
        {"<br>".join(f"• {err}" for err in init_errors)}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <strong>💡 Troubleshooting:</strong><br>
        1. Make sure MongoDB is running: <code>mongod</code><br>
        2. Check .env file configuration<br>
        3. Verify API keys are set correctly<br>
        4. Check logs in backend/data/logs/
    </div>
    """, unsafe_allow_html=True)
    
    st.stop()

# Upload section
st.markdown("### 📁 Upload Tài liệu")

st.markdown(f"""
<div class="info-card">
    <strong>⚡ Pipeline Features:</strong><br>
    • 🚀 <strong>{Config.MAX_CONCURRENT_LLM_CALLS} parallel LLM calls</strong><br>
    • 📊 <strong>Batch size {Config.EMBEDDING_BATCH_SIZE}</strong> for embeddings<br>
    • 💾 <strong>AUTO-SAVE: MongoDB + VectorDB</strong> (no manual save needed)<br>
    • 🗑️ <strong>Consistent delete</strong> (MongoDB + FAISS + Files)<br>
    • 📈 <strong>Real-time progress</strong> tracking<br>
    • 🔍 <strong>FAISS Index</strong>: {'HNSW (optimized)' if Config.USE_HNSW else 'Flat L2'}
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
    chunk_size = st.slider(
        "📏 Chunk Size (tokens)", 
        100, 1000, 
        Config.DEFAULT_CHUNK_SIZE, 
        50
    )

with col2:
    chunk_overlap = st.slider(
        "🔄 Overlap (tokens)", 
        0, 200, 
        Config.DEFAULT_CHUNK_OVERLAP, 
        10
    )

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
    if st.button("🚀 Bắt đầu xử lý", type="primary"):
        MAX_FILE_SIZE = Config.MAX_FILE_SIZE_MB * 1024 * 1024
        
        # Validate
        invalid_files = [
            f"{f.name} ({f.size/1024/1024:.1f}MB)" 
            for f in uploaded_files 
            if f.size > MAX_FILE_SIZE
        ]
        
        if invalid_files:
            st.markdown(f"""
            <div class="error-card">
                <strong>❌ File quá lớn (max {Config.MAX_FILE_SIZE_MB}MB):</strong><br>
                {"<br>".join(f"• {fname}" for fname in invalid_files)}
            </div>
            """, unsafe_allow_html=True)
            st.stop()
        
        # Main progress
        main_progress = st.progress(0)
        status_text = st.empty()
        
        success_count = 0
        failed_count = 0
        failed_files = []
        
        # Process each file
        for i, file in enumerate(uploaded_files):
            status_text.text(f"⏳ Đang xử lý [{i+1}/{len(uploaded_files)}]: {file.name}")
            
            # File progress container
            with st.expander(f"📄 {file.name}", expanded=True):
                file_progress = st.progress(0)
                file_status = st.empty()
                
                def progress_callback(msg: str, pct: float):
                    file_status.text(f"[{pct:.0f}%] {msg}")
                    file_progress.progress(pct / 100)
                
                try:
                    # ✅ Process with auto_save=True (default)
                    result = pipeline.process_file(
                        file,
                        chunk_config=ChunkConfig(
                            max_tokens=chunk_size,
                            overlap_tokens=chunk_overlap
                        ),
                        enable_extraction=enable_extraction,
                        enable_graph=enable_graph,
                        enable_embedding=enable_embedding,
                        auto_save=True,  # ✅ Auto-save to MongoDB + VectorDB
                        progress_callback=progress_callback
                    )
                    
                    if result['success']:
                        # ✅ No manual save needed - already done in pipeline!
                        
                        file_status.text("✅ Complete!")
                        file_progress.progress(1.0)
                        
                        success_count += 1
                        
                        st.success(f"✅ {file.name} - Success")
                        st.json({
                            'Chunks': result['stats'].get('chunks_count', 0),
                            'Entities': result['stats'].get('entities_count', 0),
                            'Graph Nodes': result['stats'].get('graph_nodes', 0),
                            'Embeddings': result['stats'].get('embeddings_count', 0),
                            'Vectors Added': result.get('vectors_added', 0)  # ✅ NEW
                        })
                        
                        # Show warning if any
                        if 'warning' in result:
                            st.warning(f"⚠️ {result['warning']}")
                    else:
                        failed_count += 1
                        failed_files.append((file.name, result.get('error')))
                        st.error(f"❌ {file.name}: {result.get('error')}")
                    
                except Exception as e:
                    failed_count += 1
                    failed_files.append((file.name, str(e)))
                    st.error(f"❌ {file.name}: {str(e)}")
            
            main_progress.progress((i + 1) / len(uploaded_files))
        
        main_progress.empty()
        status_text.empty()
        
        # Summary
        if success_count > 0:
            vector_stats = vector_db.get_statistics()
            
            st.markdown(f"""
            <div class="success-card">
                <strong>🎉 Hoàn thành!</strong><br>
                ✅ Thành công: {success_count} file<br>
                ❌ Thất bại: {failed_count} file<br>
                💾 MongoDB: Auto-saved (chunks, entities, graph)<br>
                🚀 VectorDB: {vector_stats['active_vectors']} vectors ({vector_stats['index_type']})<br>
                ⚡ Auto-save: Enabled (no manual save needed)
            </div>
            """, unsafe_allow_html=True)
            
            # Check if rebuild needed
            if vector_stats.get('needs_rebuild'):
                st.markdown(f"""
                <div class="warning-card">
                    ⚠️ <strong>FAISS rebuild recommended!</strong><br>
                    Deleted: {vector_stats['deleted_vectors']} / {vector_stats['total_vectors']} 
                    ({vector_stats['deletion_ratio']*100:.1f}%)<br>
                    Threshold: {Config.AUTO_REBUILD_THRESHOLD*100:.0f}%<br>
                    Run rebuild to optimize search performance.
                </div>
                """, unsafe_allow_html=True)
            
            # Show failed files if any
            if failed_files:
                st.markdown(f"""
                <div class="error-card">
                    <strong>❌ Failed files:</strong><br>
                    {"<br>".join(f"• {name}: {err}" for name, err in failed_files)}
                </div>
                """, unsafe_allow_html=True)
            
            st.rerun()

else:
    st.info("👆 Vui lòng chọn file để upload")

# Quick actions
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("💬 Chat", type="primary"):
        st.switch_page("pages/chat.py")

with col2:
    if st.button("🕸️ Xem Graph"):
        st.switch_page("pages/graph.py")

with col3:
    if st.button("📊 Statistics"):
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
            • Active Vectors: {vector_stats['active_vectors']} / {vector_stats['total_vectors']}<br>
            • Deleted: {vector_stats['deleted_vectors']} ({vector_stats['deletion_ratio']*100:.1f}%)<br>
            • Documents: {vector_stats['total_documents']}<br>
            • Type: {vector_stats['index_type']}<br>
            • Needs Rebuild: {'⚠️ YES' if vector_stats['needs_rebuild'] else '✅ NO'}<br>
            • FAISS: {'✅ Available' if vector_stats['faiss_available'] else '⚠️ NumPy Fallback'}
        </div>
        """, unsafe_allow_html=True)

# Document list with delete
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
                'Doc ID': doc['doc_id'],
                'Status': doc.get('status', 'unknown'),
                'Chunks': stats.get('chunks_count', 0),
                'Entities': stats.get('entities_count', 0),
                'Graph Nodes': stats.get('graph_nodes', 0),
                'Embeddings': stats.get('embeddings_count', 0),
                'Uploaded': doc['uploaded_at'].strftime("%m/%d %H:%M")
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, height=400)
        
        # Delete section
        st.markdown("---")
        st.markdown("### 🗑️ Xóa tài liệu")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            doc_to_delete = st.selectbox(
                "Chọn tài liệu cần xóa",
                options=[doc['doc_id'] for doc in docs],
                format_func=lambda x: next((d['filename'] for d in docs if d['doc_id'] == x), x)
            )
        
        with col2:
            if st.button("🗑️ Xóa hoàn toàn", type="secondary"):
                with st.spinner("Deleting from all storages..."):
                    try:
                        # Delete from MongoDB (cascade)
                        mongo_stats = mongo_storage.delete_document_cascade(doc_to_delete)
                        
                        # Delete from FAISS
                        faiss_stats = vector_db.delete_document(doc_to_delete)
                        
                        # Display results
                        error_msg = ""
                        if mongo_stats.get('errors'):
                            error_msg = f"<br><strong>⚠️ Warnings:</strong><br>{'<br>'.join(f'• {e}' for e in mongo_stats['errors'])}"
                        
                        st.markdown(f"""
                        <div class="success-card">
                            <strong>✅ Deleted successfully!</strong><br>
                            <br><strong>MongoDB:</strong><br>
                            • Documents: {mongo_stats['document']}<br>
                            • Chunks: {mongo_stats['chunks']}<br>
                            • Entities: {mongo_stats['entities']}<br>
                            • Relationships: {mongo_stats['relationships']}<br>
                            • Files: {len(mongo_stats['files_deleted'])}<br>
                            <br><strong>FAISS:</strong><br>
                            • Marked deleted: {faiss_stats['marked']}<br>
                            • Total deleted: {faiss_stats['total_deleted']} / {faiss_stats['total_vectors']}
                            {error_msg}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Rebuild recommendation
                        if faiss_stats['needs_rebuild']:
                            st.markdown(f"""
                            <div class="warning-card">
                                ⚠️ <strong>FAISS rebuild recommended!</strong><br>
                                Deleted ratio: {faiss_stats['total_deleted']/faiss_stats['total_vectors']*100:.1f}%<br>
                                Threshold: {Config.AUTO_REBUILD_THRESHOLD*100:.0f}%<br>
                                Click "Rebuild FAISS" below to optimize.
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.markdown(f"""
                        <div class="error-card">
                            <strong>❌ Delete failed:</strong><br>
                            {str(e)}
                        </div>
                        """, unsafe_allow_html=True)
        
        # FAISS rebuild option
        if vector_db.get_statistics()['needs_rebuild']:
            st.markdown("---")
            if st.button("🔨 Rebuild FAISS Index"):
                with st.spinner("Rebuilding FAISS index..."):
                    rebuild_stats = vector_db.rebuild_index()
                    st.markdown(f"""
                    <div class="success-card">
                        <strong>✅ Rebuild complete!</strong><br>
                        Before: {rebuild_stats['before']} vectors<br>
                        After: {rebuild_stats['after']} vectors<br>
                        Removed: {rebuild_stats['removed']} vectors
                    </div>
                    """, unsafe_allow_html=True)
                    st.rerun()
    else:
        st.info("📭 Chưa có tài liệu nào")

except Exception as e:
    st.markdown(f"""
    <div class="error-card">
        <strong>❌ Error loading documents:</strong><br>
        {str(e)}
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#6b7280;'>"
    "📤 Upload <strong>AUTO-SAVE FIXED</strong> – MongoDB + FAISS – Đại học Thủy lợi"
    "</p>",
    unsafe_allow_html=True
)