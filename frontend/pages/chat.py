# frontend/pages/chat.py
"""
💬 Chat Interface - RAG-powered Q&A
"""
import streamlit as st
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.db.vector_db import VectorDatabase
from backend.db.mongo_storage import MongoStorage
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.utils.llm_utils import call_llm_async
import asyncio

# ================= Auth Check =================
if not st.session_state.get('authenticated', False):
    st.switch_page("login.py")

user_id = st.session_state.get('user_id', 'admin_00000000')
username = st.session_state.get('username', 'User')
role = st.session_state.get('role', 'user')

# ================= Page Config =================
st.set_page_config(
    page_title="LightRAG | Chat",
    page_icon="💬",
    layout="wide"
)

# ================= CSS =================
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    
    .header-container { 
        background: linear-gradient(90deg, #1e1e1e 0%, #2d2d2d 100%); 
        padding: 1.5rem; 
        border-radius: 10px; 
        margin-bottom: 2rem; 
        border-left: 5px solid #3b82f6; 
    }
    .header-title { 
        color: #3b82f6; 
        font-size: 2rem; 
        font-weight: 700; 
        margin: 0; 
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    .assistant-message {
        background: #1e1e1e;
        color: white;
        margin-right: 20%;
        border-left: 4px solid #3b82f6;
    }
    
    .context-preview {
        background: #1a1a1a;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #10b981;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    
    .source-tag {
        display: inline-block;
        background: #374151;
        color: #9ca3af;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        margin: 0.2rem;
    }
    
    .stat-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .badge-chunks { background: #3b82f6; color: white; }
    .badge-entities { background: #8b5cf6; color: white; }
    .badge-time { background: #10b981; color: white; }
</style>
""", unsafe_allow_html=True)

# ================= Header =================
st.markdown(f"""
<div class="header-container">
    <div class="header-title">💬 RAG Chat Assistant</div>
</div>
""", unsafe_allow_html=True)

# ================= Sidebar =================
with st.sidebar:
    st.markdown(f"## 👤 {username}")
    st.markdown(f"**Role**: {role}<br>**ID**: `{user_id}`", unsafe_allow_html=True)
    st.markdown("---")
    
    # Navigation
    st.markdown("### 🧭 Navigation")
    if role == 'admin':
        if st.button("📤 Upload"):
            st.switch_page("pages/upload.py")
        if st.button("🕸️ Graph"):
            st.switch_page("pages/graph.py")
    
    st.markdown("---")
    
    # Settings
    st.markdown("### ⚙️ Chat Settings")
    
    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        options=['auto', 'vector', 'graph', 'hybrid'],
        help="auto = Let AI decide"
    )
    
    top_k = st.slider(
        "Results per search",
        min_value=3,
        max_value=15,
        value=5,
        step=1
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher = more creative"
    )
    
    show_context = st.checkbox("Show retrieved context", value=False)
    show_metadata = st.checkbox("Show metadata", value=True)
    
    st.markdown("---")
    
    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    
    # Logout
    if st.button("🚪 Logout"):
        for k in ['authenticated', 'user_id', 'username', 'role']:
            st.session_state.pop(k, None)
        st.switch_page("login.py")

# ================= Initialize Components =================
@st.cache_resource
def init_retriever(user_id: str):
    """Initialize retriever (cached)"""
    try:
        vector_db = VectorDatabase(user_id)
        storage = MongoStorage(user_id)
        retriever = HybridRetriever(vector_db, storage)
        
        # Check data availability
        vec_stats = vector_db.get_statistics()
        graph = storage.get_graph()
        
        return retriever, {
            'vectors': vec_stats['active_vectors'],
            'nodes': len(graph.get('nodes', [])),
            'docs': vec_stats['total_documents']
        }
    except Exception as e:
        st.error(f"❌ Failed to initialize retriever: {e}")
        return None, None

retriever, stats = init_retriever(user_id)

# ================= Check Data Availability =================
if not retriever or not stats:
    st.error("❌ Failed to initialize chat system")
    st.stop()

if stats['vectors'] == 0:
    st.warning("⚠️ No documents uploaded yet. Please upload documents first.")
    
    col1, col2 = st.columns(2)
    with col1:
        if role == 'admin':
            if st.button("📤 Go to Upload"):
                st.switch_page("pages/upload.py")
    with col2:
        st.info("💡 Upload documents to enable chat")
    
    st.stop()

# Show stats
st.markdown(f"""
<div class="context-preview">
    <strong>📊 Knowledge Base Status:</strong><br>
    📄 Documents: {stats['docs']} | 
    🧮 Vectors: {stats['vectors']} | 
    🕸️ Graph Nodes: {stats['nodes']}
</div>
""", unsafe_allow_html=True)

# ================= Initialize Chat History =================
if 'messages' not in st.session_state:
    st.session_state.messages = []

# ================= Display Chat History =================
for message in st.session_state.messages:
    role = message['role']
    content = message['content']
    
    if role == 'user':
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 You:</strong><br>{content}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message assistant-message">
            <strong>🤖 Assistant:</strong><br>{content}
        </div>
        """, unsafe_allow_html=True)
        
        # Show metadata if enabled
        if show_metadata and 'metadata' in message:
            meta = message['metadata']
            st.markdown(f"""
            <div style="margin-left: 2rem; margin-top: 0.5rem;">
                <span class="stat-badge badge-chunks">📄 {meta.get('num_chunks', 0)} chunks</span>
                <span class="stat-badge badge-entities">🕸️ {meta.get('num_entities', 0)} entities</span>
                <span class="stat-badge badge-time">⏱️ {meta.get('retrieval_time_ms', 0)}ms</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Show context if enabled
        if show_context and 'context' in message:
            with st.expander("📋 Retrieved Context", expanded=False):
                st.text(message['context'][:1000] + "..." if len(message['context']) > 1000 else message['context'])

# ================= Chat Input =================
st.markdown("---")

# Quick examples
st.markdown("### 💡 Example Questions")
examples = [
    "What is the main topic of the documents?",
    "Summarize the key points",
    "Explain the concept of [topic]",
    "Compare [entity A] and [entity B]"
]

example_cols = st.columns(len(examples))
for col, example in zip(example_cols, examples):
    with col:
        if st.button(example, key=f"example_{example}", use_container_width=True):
            st.session_state.pending_query = example
            st.rerun()

st.markdown("---")


user_query = st.chat_input("Ask me anything about your documents...")


if 'pending_query' in st.session_state:
    user_query = st.session_state.pending_query
    del st.session_state.pending_query

# ================= Process Query =================
if user_query:
    # Add user message
    st.session_state.messages.append({
        'role': 'user',
        'content': user_query
    })
    
    # Show user message immediately
    st.markdown(f"""
    <div class="chat-message user-message">
        <strong>👤 You:</strong><br>{user_query}
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("🤔 Thinking..."):
        try:
            force_mode = None if retrieval_mode == 'auto' else retrieval_mode
            
            context = retriever.retrieve(
                query=user_query,
                force_mode=force_mode,
                top_k=top_k
            )
            
            system_prompt = """You are a helpful AI assistant that answers questions based on provided context.

Instructions:
1. Answer accurately using ONLY the provided context
2. Cite sources using [1], [2], etc. when referencing specific information
3. If the context doesn't contain enough information, acknowledge this
4. Be concise but comprehensive
5. Use clear, professional language"""

            user_prompt = f"""
{context.formatted_text}

Now, please answer this question:
{user_query}
"""
            try:
                # Try async first
                response = asyncio.run(call_llm_async(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=2000
                ))
            except:
                # Fallback to sync
                from backend.utils.llm_utils import call_llm
                response = call_llm(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=2000
                )
            
            # Add assistant message
            assistant_message = {
                'role': 'assistant',
                'content': response,
                'metadata': context.metadata,
                'context': context.formatted_text
            }
            st.session_state.messages.append(assistant_message)
            
            # Rerun to show new message
            st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            
            # Add error message
            st.session_state.messages.append({
                'role': 'assistant',
                'content': f"Sorry, I encountered an error: {str(e)}"
            })

# ================= Footer =================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; font-size: 0.9rem;">
    <p>💬 RAG Chat powered by <strong>mini-lightrag</strong> – Đại học Thủy lợi</p>
    <p style="font-size: 0.8rem;">
        🔍 Hybrid Retrieval | 🧮 Vector Search | 🕸️ Knowledge Graph | 🤖 LLM Generation
    </p>
</div>
""", unsafe_allow_html=True)

# ================= Debug Info (Admin only) =================
if role == 'admin' and st.sidebar.checkbox("🔧 Debug Mode", value=False):
    with st.sidebar:
        st.markdown("### 🐛 Debug Info")
        st.json({
            'user_id': user_id,
            'role': role,
            'messages_count': len(st.session_state.messages),
            'stats': stats,
            'retrieval_mode': retrieval_mode,
            'top_k': top_k,
            'temperature': temperature
        })
