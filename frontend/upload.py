# pages/upload.py
import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import os

#  FIX: Use absolute path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.core.pipeline import DocumentPipeline, DocChunkConfig
from backend.core.graph_builder import merge_admin_graphs

# ================= AUTH =================
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

# ================= CSS =================
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
    .upload-box { 
        border: 2px dashed #667eea; 
        border-radius: 12px; 
        padding: 2rem; 
        text-align: center; 
        background: #1a1a2e; 
        transition: all 0.3s;
    }
    .upload-box:hover { 
        border-color: #dc2626; 
        background: #1e1e2e;
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
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
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
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown(f"""
<div class="header-container">
    <div class="header-title">📤 Upload Document <span class="admin-badge">ADMIN ONLY</span></div>
</div>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
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

# ================= MAIN CONTENT =================

# Khởi tạo pipeline
pipeline = DocumentPipeline(user_id=user_id, enable_advanced=True)

# Upload section
st.markdown("### 📁 Upload Tài liệu")

# Info box
st.markdown("""
<div class="info-card">
    <strong>📋 Định dạng hỗ trợ:</strong><br>
    • Documents: PDF, DOCX, TXT, MD<br>
    • Data: CSV, XLSX, JSON, XML<br>
    • Code: PY, JS, JAVA, CPP<br>
    • Max size: 50MB per file
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
        file_size = f.size / 1024 / 1024  # MB
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
        help="Kích thước mỗi chunk. Lớn hơn = ít chunk hơn nhưng mỗi chunk dài hơn"
    )

with col2:
    chunk_overlap = st.slider(
        "🔄 Overlap (tokens)",
        min_value=0,
        max_value=200,
        value=50,
        step=10,
        help="Số tokens chồng lấn giữa các chunk liền kề"
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
    
    enable_gleaning = st.checkbox(
        "✨ Bật Gleaning (Tinh chỉnh bằng LLM)",
        value=False,
        help="⚠️ Cải thiện chất lượng extraction nhưng tốn thêm LLM calls"
    )
    
    if enable_gleaning:
        st.markdown("""
        <div class="warning-card">
            <strong>⚠️ Lưu ý:</strong> Gleaning sẽ gọi LLM thêm 2-3 lần để refine entities/relationships. 
            Điều này tăng chi phí API và thời gian xử lý.
        </div>
        """, unsafe_allow_html=True)

# ✅ FIX: Add file validation
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

# Process button
st.markdown("---")
if uploaded_files:
    if st.button("🚀 Bắt đầu xử lý", type="primary", use_container_width=True):
        
        # ✅ FIX: Validate file sizes
        invalid_files = []
        for f in uploaded_files:
            if f.size > MAX_FILE_SIZE:
                invalid_files.append(f"{f.name} ({f.size / 1024 / 1024:.1f}MB > 50MB)")
        
        if invalid_files:
            st.error("❌ **File quá lớn!**")
            for fname in invalid_files:
                st.markdown(f"- {fname}")
            st.stop()
        
        # Processing
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        success_count = 0
        failed_count = 0
        error_messages = []
        
        for i, f in enumerate(uploaded_files):
            status_text.text(f"⏳ Đang xử lý [{i+1}/{len(uploaded_files)}]: {f.name}")
            
            try:
                result = pipeline.process_uploaded_file(
                    uploaded_file=f,
                    chunk_config=DocChunkConfig(
                        max_token_size=chunk_size,
                        overlap_token_size=chunk_overlap
                    ),
                    enable_extraction=enable_extraction,
                    enable_graph=enable_graph,
                    enable_embedding=enable_embedding,
                    enable_gleaning=enable_gleaning
                )
                
                if result.get('success', False):
                    success_count += 1
                    
                    # Show detailed results
                    with st.expander(f"✅ {f.name} - Thành công"):
                        st.json({
                            'Chunks': result.get('chunks_count', 0),
                            'Tokens': result.get('total_tokens', 0),
                            'Entities': result.get('entities_count', 0),
                            'Relationships': result.get('relationships_count', 0),
                            'Graph Nodes': result.get('graph_nodes', 0),
                            'Graph Edges': result.get('graph_edges', 0),
                            'Embeddings': result.get('total_embeddings', 0)
                        })
                else:
                    failed_count += 1
                    error_messages.append(f"❌ {f.name}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                failed_count += 1
                error_messages.append(f"❌ {f.name}: {str(e)}")
                st.error(f"Lỗi xử lý {f.name}: {str(e)}")
            
            # Update progress
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        # Clear status
        progress_bar.empty()
        status_text.empty()
        
        # Show results
        if success_count > 0:
            st.markdown(f"""
            <div class="success-card">
                <strong>🎉 Hoàn thành!</strong><br>
                ✅ Thành công: {success_count} file<br>
                ❌ Thất bại: {failed_count} file
            </div>
            """, unsafe_allow_html=True)
        
        # Show errors
        if error_messages:
            with st.expander("⚠️ Chi tiết lỗi", expanded=False):
                for msg in error_messages:
                    st.markdown(msg)
        
        # Auto merge graphs
        if success_count > 0:
            with st.spinner("🔄 Đang tổng hợp Knowledge Graph..."):
                try:
                    merged = merge_admin_graphs(user_id)
                    if merged:
                        st.success("✅ Graph tổng hợp đã cập nhật!")
                    else:
                        st.warning("⚠️ Không có dữ liệu mới để merge.")
                except Exception as e:
                    st.error(f"❌ Lỗi merge graph: {str(e)}")
            
            # Reload page to show new files
            st.rerun()

else:
    st.info("👆 Vui lòng chọn file để upload")

# Quick actions
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    if st.button("🕸️ Xem Knowledge Graph", type="primary", use_container_width=True):
        st.switch_page("pages/graph.py")

with col2:
    if st.button("📊 Xem Statistics", use_container_width=True):
        docs = pipeline.get_processed_docs()
        if docs:
            total_chunks = sum(d['chunks'] for d in docs)
            total_tokens = sum(d['tokens'] for d in docs)
            
            st.markdown(f"""
            <div class="info-card">
                <strong>📈 Thống kê hệ thống:</strong><br>
                • Documents: {len(docs)}<br>
                • Total Chunks: {total_chunks}<br>
                • Total Tokens: {total_tokens:,}<br>
                • Avg Tokens/Doc: {total_tokens // len(docs) if docs else 0:,}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Chưa có dữ liệu")

# ================= DOCUMENT LIST =================
st.markdown("---")
st.markdown("### 📚 Tài liệu đã xử lý")

docs = pipeline.get_processed_docs()

if docs:
    # Create DataFrame
    df = pd.DataFrame(docs)
    
    # Format columns
    df_display = df.copy()
    df_display['has_graph'] = df_display['has_graph'].apply(lambda x: '✅' if x else '❌')
    df_display['has_embeddings'] = df_display['has_embeddings'].apply(lambda x: '✅' if x else '❌')
    
    # Rename columns
    df_display.columns = ['File', 'Chunks', 'Tokens', 'Thời gian', 'Graph', 'Embeddings']
    
    # Display table
    st.dataframe(
        df_display,
        use_container_width=True,
        height=400
    )
    
    # Delete document section
    with st.expander("🗑️ Xóa tài liệu", expanded=False):
        doc_to_delete = st.selectbox(
            "Chọn tài liệu cần xóa",
            options=[Path(d['file']).stem for d in docs],
            help="⚠️ Hành động này không thể hoàn tác!"
        )
        
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Xóa", type="secondary"):
                if pipeline.delete_document(doc_to_delete):
                    st.success(f"✅ Đã xóa: {doc_to_delete}")
                    st.rerun()
                else:
                    st.error(f"❌ Không thể xóa: {doc_to_delete}")
    
    # Export option
    with st.expander("💾 Export dữ liệu", expanded=False):
        export_format = st.radio(
            "Chọn định dạng export",
            options=["CSV", "JSON", "Excel"],
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
            elif export_format == "Excel":
                # Note: Requires openpyxl
                st.info("📌 Excel export requires openpyxl. Install: pip install openpyxl")
else:
    st.info("📭 Chưa có tài liệu nào được xử lý. Hãy upload file để bắt đầu!")
    
    # Show example
    with st.expander("💡 Hướng dẫn sử dụng", expanded=True):
        st.markdown("""
        ### 🎯 Cách sử dụng:
        
        1. **Upload file** 📤
           - Click nút "Browse files" ở trên
           - Chọn 1 hoặc nhiều file (PDF, DOCX, TXT...)
           - Mỗi file tối đa 50MB
        
        2. **Cấu hình** ⚙️
           - Điều chỉnh Chunk Size và Overlap
           - Chọn các tùy chọn nâng cao nếu cần
        
        3. **Xử lý** 🚀
           - Click "Bắt đầu xử lý"
           - Đợi hệ thống phân tích tài liệu
        
        4. **Xem kết quả** 📊
           - Vào "Knowledge Graph" để xem graph
           - Dùng Chat để hỏi đáp (coming soon)
        
        ### 💡 Tips:
        - File nhỏ (< 5 pages) dùng chunk size 200-300
        - File lớn dùng chunk size 400-600
        - Bật Gleaning chỉ khi cần chất lượng cao
        """)

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#6b7280;'>"
    "📤 Upload <strong>mini-lightrag</strong> – Đại học Thủy lợi"
    "</p>",
    unsafe_allow_html=True
)