# frontend/pages/upload.py

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


if not st.session_state.get('authenticated', False):
    st.switch_page("login.py")

if st.session_state.get('role') != 'admin':
    st.error("⛔ Chỉ **Admin** được phép truy cập trang này.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🏠 Quay lại Login", use_container_width=True): 
            st.switch_page("login.py")
    with col2:
        if st.button("💬 Đi tới Chat", use_container_width=True): 
            st.info("Chat feature coming soon!")
    st.stop()

user_id = st.session_state.get('user_id', 'admin_00000000')
username = st.session_state.get('username', 'Admin')
st.set_page_config(page_title="LightRAG | Upload", page_icon="📤", layout="wide")


st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .header-container { 
        background: linear-gradient(90deg, #1e1e1e 0%, #2d2d2d 100%); 
        padding: 1.5rem; 
        border-radius: 10px; 
        margin-bottom: 2rem; 
        border-left: 5px solid #dc2626; 
    }
    .header-title { color: #dc2626; font-size: 2rem; font-weight: 700; margin: 0; }
    .admin-badge { 
        background: #dc2626; 
        color: white; 
        padding: 0.3rem 0.8rem; 
        border-radius: 12px; 
        font-size: 0.8rem; 
        font-weight: 600; 
    }
    .stButton > button { 
        width: 100%; 
        border-radius: 8px; 
        font-weight: 600; 
    }
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


st.markdown(f"""
<div class="header-container">
    <div class="header-title">📤 Upload Document <span class="admin-badge">MONGODB</span></div>
</div>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## 👤 Admin Info")
    st.markdown(f"**User**: {username}<br>**ID**: `{user_id}`", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 🧭 Navigation")
    if st.button("🕸️ Knowledge Graph", use_container_width=True):
        st.switch_page("pages/graph.py")
    
    if st.button("💬 Chat (Soon)", use_container_width=True, disabled=True):
        st.info("Chat feature coming soon!")
    
    st.markdown("---")
    
    if st.button("🚪 Logout", use_container_width=True, type="secondary"):
        for k in ['authenticated', 'user_id', 'username', 'role']:
            st.session_state.pop(k, None)
        st.switch_page("login.py")

try:
    storage = MongoStorage(user_id)
    pipeline = DocumentPipeline(user_id)
except Exception as e:
    st.error(f"❌ Failed to connect to MongoDB: {e}")
    st.info("💡 Make sure MongoDB is running: `mongod` or check MONGODB_URI in .env")
    st.stop()



st.markdown("### 📁 Upload Tài liệu")


st.markdown("""
<div class="info-card">
    <strong>📋 Định dạng hỗ trợ:</strong><br>
    • Documents: PDF, DOCX, TXT, MD<br>
    • Data: CSV, XLSX, JSON, XML<br>
    • Code: PY, JS, JAVA, CPP<br>
    • Max size: 50MB per file<br>
    • ✅ Dữ liệu lưu trong MongoDB
</div>
""", unsafe_allow_html=True)

# File uploader
uploaded_files = st.file_uploader(
    "Chọn file để upload",
    type=['pdf', 'docx', 'txt', 'md', 'csv', 'xlsx', 'html', 'py', 'js', 'java', 'cpp', 'json', 'xml'],
    accept_multiple_files=True,
    help="Có thể chọn nhiều file cùng lúc"
)

# Display selected files
if uploaded_files:
    st.markdown(f"**Đã chọn {len(uploaded_files)} file:**")
    for f in uploaded_files:
        file_size = f.size / 1024 / 1024  
        st.markdown(f"- 📄 {f.name} ({file_size:.2f} MB)")

# Processing configuration
st.markdown("---")
st.markdown("### ⚙️ Cấu hình xử lý")

col1, col2 = st.columns(2)

with col1:
    chunk_size = st.slider(
        "📏 Chunk Size (tokens)",
        min_value=100,
        max_value=1000,
        value=300,
        step=50,
        help="Kích thước mỗi chunk"
    )

with col2:
    chunk_overlap = st.slider(
        "🔄 Overlap (tokens)",
        min_value=0,
        max_value=200,
        value=50,
        step=10,
        help="Số tokens chồng lấn"
    )

# Advanced options
with st.expander("🔧 Tùy chọn nâng cao", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        enable_extraction = st.checkbox(
            "📊 Entity Extraction",
            value=True,
            help="Trích xuất entities và relationships"
        )
    
    with col2:
        enable_graph = st.checkbox(
            "🕸️ Knowledge Graph",
            value=True,
            help="Xây dựng knowledge graph"
        )
    
    with col3:
        enable_embedding = st.checkbox(
            "🧮 Vector Embedding",
            value=True,
            help="Tạo embeddings cho semantic search"
        )

# ================= PROCESS BUTTON =================
st.markdown("---")
if uploaded_files:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        process_btn = st.button("🚀 Bắt đầu xử lý", type="primary", use_container_width=True)
    
    with col2:
        use_parallel = st.checkbox("⚡ Parallel", value=True, 
                                   help="Xử lý nhiều file cùng lúc")
    
    if process_btn:
        MAX_FILE_SIZE = 50 * 1024 * 1024
        
        # Validate file sizes
        invalid_files = []
        for f in uploaded_files:
            if f.size > MAX_FILE_SIZE:
                invalid_files.append(f"{f.name} ({f.size / 1024 / 1024:.1f}MB > 50MB)")
        
        if invalid_files:
            st.error("❌ **File quá lớn!**")
            for fname in invalid_files:
                st.markdown(f"- {fname}")
            st.stop()
        
        # Process files
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        success_count = 0
        failed_count = 0
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"⏳ Đang xử lý [{i+1}/{len(uploaded_files)}]: {file.name}")
            
            try:
                # ✅ NEW: Process with pipeline (returns data)
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
                    # ✅ NEW: Save to MongoDB
                    doc_id = result['doc_id']
                    
                    # Save document metadata
                    storage.save_document(
                        doc_id=doc_id,
                        filename=result['filename'],
                        filepath=result['filepath'],
                        metadata={'original_size': file.size}
                    )
                    
                    # Save chunks
                    storage.save_chunks(doc_id, result['chunks'])
                    
                    # Save entities & relationships
                    if result.get('entities'):
                        storage.save_entities(doc_id, result['entities'])
                        storage.save_relationships(doc_id, result['relationships'])
                    
                    # Save graph
                    if result.get('graph'):
                        storage.save_graph(result['graph'])
                    
                    # Save embeddings
                    if result.get('embeddings'):
                        storage.save_embeddings(doc_id, result['embeddings'])
                    
                    # Update document status
                    storage.update_document_status(doc_id, 'completed', result['stats'])
                    
                    success_count += 1
                    
                    # Show results
                    with st.expander(f"✅ {file.name} - Thành công"):
                        st.json({
                            'Chunks': result['stats'].get('chunks_count', 0),
                            'Tokens': result['stats'].get('total_tokens', 0),
                            'Entities': result['stats'].get('entities_count', 0),
                            'Relationships': result['stats'].get('relationships_count', 0),
                            'Graph Nodes': result['stats'].get('graph_nodes', 0),
                            'Graph Edges': result['stats'].get('graph_edges', 0),
                            'Embeddings': result['stats'].get('embeddings_count', 0)
                        })
                else:
                    failed_count += 1
                    st.error(f"❌ {file.name}: {result.get('error')}")
                
                results.append(result)
                
            except Exception as e:
                failed_count += 1
                st.error(f"❌ {file.name}: {str(e)}")
                results.append({
                    'success': False,
                    'filename': file.name,
                    'error': str(e)
                })
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Clear status
        progress_bar.empty()
        status_text.empty()
        
        # Show summary
        if success_count > 0:
            st.markdown(f"""
            <div class="success-card">
                <strong>🎉 Hoàn thành!</strong><br>
                ✅ Thành công: {success_count} file<br>
                ❌ Thất bại: {failed_count} file<br>
                💾 Dữ liệu đã lưu vào MongoDB
            </div>
            """, unsafe_allow_html=True)
            
            # Reload page
            st.rerun()

else:
    st.info("👆 Vui lòng chọn file để upload")

# ================= QUICK ACTIONS =================
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    if st.button("🕸️ Xem Knowledge Graph", type="primary", use_container_width=True):
        st.switch_page("pages/graph.py")

with col2:
    if st.button("📊 Xem Statistics", use_container_width=True):
        # ✅ NEW: Get stats from MongoDB
        stats = storage.get_user_statistics()
        
        if stats['total_documents'] > 0:
            st.markdown(f"""
            <div class="info-card">
                <strong>📈 Thống kê MongoDB:</strong><br>
                • Documents: {stats['total_documents']}<br>
                • Chunks: {stats['total_chunks']}<br>
                • Entities: {stats['total_entities']}<br>
                • Relationships: {stats['total_relationships']}<br>
                • Graph Nodes: {stats['graph_nodes']}<br>
                • Graph Edges: {stats['graph_edges']}<br>
                • Embeddings: {stats['total_embeddings']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Chưa có dữ liệu")

# ================= DOCUMENT LIST =================
st.markdown("---")
st.markdown("### 📚 Tài liệu đã xử lý")

# ✅ NEW: Get documents from MongoDB
try:
    docs = storage.list_documents()
    
    if docs:
        # Create DataFrame
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
        
        # Display table
        st.dataframe(
            df,
            use_container_width=True,
            height=400
        )
        
        # Delete document section
        with st.expander("🗑️ Xóa tài liệu", expanded=False):
            doc_to_delete = st.selectbox(
                "Chọn tài liệu cần xóa",
                options=[doc['doc_id'] for doc in docs],
                format_func=lambda x: next((d['filename'] for d in docs if d['doc_id'] == x), x),
                help="⚠️ Hành động này không thể hoàn tác!"
            )
            
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("🗑️ Xóa", type="secondary"):
                    # ✅ NEW: Delete from MongoDB
                    if storage.delete_document(doc_to_delete):
                        st.success(f"✅ Đã xóa tài liệu")
                        st.rerun()
                    else:
                        st.error(f"❌ Không thể xóa tài liệu")
        
        # Export option
        with st.expander("💾 Export dữ liệu", expanded=False):
            export_format = st.radio(
                "Chọn định dạng export",
                options=["CSV", "JSON"],
                horizontal=True
            )
            
            if st.button("💾 Export", use_container_width=True):
                if export_format == "CSV":
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                elif export_format == "JSON":
                    json_str = df.to_json(orient='records', indent=2)
                    st.download_button(
                        label="📥 Download JSON",
                        data=json_str,
                        file_name=f"documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    else:
        st.info("📭 Chưa có tài liệu nào được xử lý. Hãy upload file để bắt đầu!")
        
        # Show example
        with st.expander("💡 Hướng dẫn sử dụng", expanded=True):
            st.markdown("""
            ### 🎯 Cách sử dụng:
            
            1. **Upload file** 📤
               - Click nút "Browse files" ở trên
               - Chọn 1 hoặc nhiều file
               - Mỗi file tối đa 50MB
            
            2. **Cấu hình** ⚙️
               - Điều chỉnh Chunk Size và Overlap
               - Chọn các tùy chọn nâng cao
            
            3. **Xử lý** 🚀
               - Click "Bắt đầu xử lý"
               - Dữ liệu tự động lưu vào MongoDB
            
            4. **Xem kết quả** 📊
               - Vào "Knowledge Graph" để xem graph
               - Dữ liệu lưu trong MongoDB, không tạo file
            
            ### 💡 MongoDB Storage:
            - Tất cả dữ liệu lưu trong database
            - Không tạo JSON files
            - Dễ query và scale
            """)

except Exception as e:
    st.error(f"❌ Lỗi kết nối MongoDB: {e}")
    st.info("💡 Kiểm tra: mongod đang chạy và MONGODB_URI trong .env")

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#6b7280;'>"
    "📤 Upload <strong>MongoDB Version</strong> – Đại học Thủy lợi"
    "</p>",
    unsafe_allow_html=True
)