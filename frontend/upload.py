# pages/upload.py
import streamlit as st
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backend.core.pipeline import DocumentPipeline, DocChunkConfig
from backend.core.graph_builder import merge_admin_graphs

# ================= AUTH =================
if not st.session_state.get('authenticated', False):
    st.switch_page("login.py")

if st.session_state.get('role') != 'admin':
    st.error("Chỉ **Admin** được phép truy cập.")
    if st.button("Quay lại Login"): st.switch_page("login.py")
    st.stop()

user_id = st.session_state.get('user_id', 'admin_00000000')
username = st.session_state.get('username', 'Admin')
st.set_page_config(page_title="LightRAG | Upload", page_icon="Upload", layout="wide")

# ================= CSS =================
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .header-container { background: linear-gradient(90deg, #1e1e1e 0%, #2d2d2d 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; border-left: 5px solid #dc2626; }
    .header-title { color: #dc2626; font-size: 2rem; font-weight: 700; margin: 0; }
    .admin-badge { background: #dc2626; color: white; padding: 0.3rem 0.8rem; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
    .upload-box { border: 2px dashed #667eea; border-radius: 12px; padding: 2rem; text-align: center; background: #1a1a2e; }
    .upload-box:hover { border-color: #dc2626; }
    .stButton > button { width: 100%; border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ================= UI =================
st.markdown(f"""
<div class="header-container">
    <div class="header-title">Upload Document <span class="admin-badge">ADMIN</span></div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Admin Info")
    st.markdown(f"**User**: {username}<br>**ID**: `{user_id}`")
    st.markdown("---")
    if st.button("Logout", use_container_width=True, type="secondary"):
        for k in ['authenticated', 'user_id', 'username', 'role']:
            st.session_state.pop(k, None)
        st.switch_page("login.py")

# Khởi tạo pipeline
pipeline = DocumentPipeline(user_id=user_id, enable_advanced=True)

st.markdown("### Upload Tài liệu")
uploaded_files = st.file_uploader(
    "Chọn file", type=['pdf','docx','txt','md','csv','xlsx','html'],
    accept_multiple_files=True
)

col1, col2 = st.columns(2)
with col1: chunk_size = st.slider("Chunk Size", 100, 1000, 300)
with col2: chunk_overlap = st.slider("Overlap", 0, 200, 50)

# === THÊM CHECKBOX GLEANING ===
with st.expander("Tùy chọn nâng cao", expanded=False):
    enable_gleaning = st.checkbox("Bật Gleaning (tinh chỉnh bằng LLM)", value=False)

if uploaded_files and st.button("Xử lý", type="primary", use_container_width=True):
    bar = st.progress(0)
    status = st.empty()
    success = failed = 0
    for i, f in enumerate(uploaded_files):
        status.text(f"[{i+1}/{len(uploaded_files)}] {f.name}")
        try:
            res = pipeline.process_uploaded_file(
                uploaded_file=f,
                chunk_config=DocChunkConfig(max_token_size=chunk_size, overlap_token_size=chunk_overlap),
                enable_extraction=True, enable_graph=True, enable_embedding=True,
                enable_gleaning=enable_gleaning  # ← truyền vào
            )
            success += res.get('success', False)
            failed += not res.get('success', False)
        except Exception as e:
            failed += 1
            st.error(f"{f.name}: {e}")
        bar.progress((i+1)/len(uploaded_files))
    bar.empty(); status.empty()
    if success: st.success(f"Xử lý thành công {success} file!")

    # AUTO MERGE
    with st.spinner("Tổng hợp Knowledge Graph..."):
        merged = merge_admin_graphs(user_id)
        st.success("Graph tổng hợp đã cập nhật!") if merged else st.warning("Không có dữ liệu mới.")
    st.rerun()

st.markdown("---")
if st.button("Xem Knowledge Graph", type="primary", use_container_width=True):
    st.switch_page("pages/graph.py")

# ================= DANH SÁCH FILE ĐÃ XỬ LÝ (DÙNG PIPELINE METHOD) =================
st.markdown("### Đã xử lý")

docs = pipeline.get_processed_docs()

if docs:
    df = pd.DataFrame(docs)
    st.dataframe(
        df[['file', 'chunks', 'tokens', 'time', 'has_graph', 'has_embeddings']],
        use_container_width=True
    )
else:
    st.info("Chưa có file.")

st.markdown("---")
st.markdown("<p style='text-align:center; color:#6b7280;'>Upload <strong>mini-lightrag</strong> – Đại học Thủy lợi</p>", unsafe_allow_html=True)