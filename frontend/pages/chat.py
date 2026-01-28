# frontend/pages/chat.py (SHARED DATA VERSION)
"""
💬 Chat Interface - Users share admin's data
"""
import streamlit as st
import sys
import os
import time
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from backend.db.vector_db import VectorDatabase
from backend.db.mongo_storage import MongoStorage
from backend.db.conversation_storage import ConversationStorage
from backend.db.feedback_storage import FeedbackStorage
from backend.retrieval.hybrid_retriever import EnhancedHybridRetriever
from backend.retrieval.conversation_manager import ConversationManager
from backend.evaluation.response_evaluator import ResponseEvaluator
from backend.utils.llm_utils import call_llm_async, call_llm_stream
import asyncio


# ================= Auth Check =================
if not st.session_state.get('authenticated', False):
    st.switch_page("login.py")

user_id = st.session_state.get('user_id', 'admin_00000000')
username = st.session_state.get('username', 'User')
role = st.session_state.get('role', 'user')

#
DATA_USER_ID = 'admin_00000000'  # All users read from admin's data

# ================= Page Config =================
st.set_page_config(
    page_title="LightRAG | Chat",
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
    
    .role-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-left: 1rem;
    }
    .badge-admin { background: #dc2626; color: white; }
    .badge-user { background: #10b981; color: white; }
    
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
    
    .info-box {
        background: #1e3a8a;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #3b82f6;
    }
    
    /* Feedback Styles */
    .feedback-container {
        background: #1a1a1a;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0 1rem 2rem;
        border-left: 3px solid #f59e0b;
    }
    .star-rating {
        font-size: 1.5rem;
        cursor: pointer;
        user-select: none;
    }
    .star-rating span {
        color: #4b5563;
        transition: color 0.2s;
    }
    .star-rating span:hover,
    .star-rating span.selected {
        color: #fbbf24;
    }
    .feedback-submitted {
        background: #065f46;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        margin: 0.5rem 0;
    }
    
    /* Evaluation Styles */
    .evaluation-container {
        background: #1a1a1a;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0 1rem 2rem;
        border-left: 3px solid #8b5cf6;
    }
    .evaluation-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.3rem;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-relevancy {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
    }
    .badge-faithfulness {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
    .badge-response-time {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    .evaluation-score {
        font-size: 1.2rem;
        font-weight: 700;
    }
    .evaluation-reason {
        background: #2d2d2d;
        padding: 0.5rem;
        border-radius: 5px;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        color: #d1d5db;
    }
    
    /* Loading Spinner */
    .spinner {
        border: 3px solid #2d2d2d;
        border-top: 3px solid #3b82f6;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
        display: inline-block;
        margin-right: 0.5rem;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# ================= Initialize Storage =================
@st.cache_resource
def init_storage(data_user_id: str, conv_user_id: str):
    """
    Initialize storage - Users read from admin's data
    
    Args:
        data_user_id: User ID for data (admin_00000000 for all users)
        conv_user_id: User ID for conversation history (unique per user)
    """
    try:
        vector_db = VectorDatabase(data_user_id)
        mongo_storage = MongoStorage(data_user_id)
        
        conv_storage = ConversationStorage(conv_user_id)
        
        feedback_storage = FeedbackStorage(conv_user_id)
        
        retriever = EnhancedHybridRetriever(vector_db, mongo_storage)
        
        vec_stats = vector_db.get_statistics()
        graph = mongo_storage.get_graph()
        
        return {
            'vector_db': vector_db,
            'mongo_storage': mongo_storage,
            'conv_storage': conv_storage,
            'feedback_storage': feedback_storage,
            'retriever': retriever,
            'stats': {
                'vectors': vec_stats['active_vectors'],
                'nodes': len(graph.get('nodes', [])),
                'docs': vec_stats['total_documents']
            }
        }
    except Exception as e:
        st.error(f"❌ Failed to initialize: {e}")
        return None

storage = init_storage(DATA_USER_ID, user_id)

if not storage:
    st.error("❌ Failed to initialize chat system")
    st.stop()

retriever = storage['retriever']
conv_storage = storage['conv_storage']
feedback_storage = storage['feedback_storage']
stats = storage['stats']

# ================= Check Data =================
if stats['vectors'] == 0:
    st.warning("⚠️ No documents available yet.")
    if role == 'admin':
        if st.button("📤 Go to Upload"):
            st.switch_page("pages/upload.py")
    else:
        st.info("💡 Please contact admin to upload documents.")
    st.stop()

# ================= Header =================
role_badge_class = "badge-admin" if role == "admin" else "badge-user"
role_display = "Admin" if role == "admin" else "User"

st.markdown(f"""
<div class="header-container">
    <div class="header-title">
        💬 Multi-Conversation Chat
        <span class="role-badge {role_badge_class}">{role_display.upper()}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Show users info 
if role == 'user':
    st.markdown(f"""
    <div class="info-box">
        <strong>📚 Shared Knowledge Base</strong><br>
        You are chatting with documents uploaded by admin.<br>
        Your conversation history is private and saved separately.
    </div>
    """, unsafe_allow_html=True)

if 'current_conversation_id' not in st.session_state or st.session_state.current_conversation_id is None:
    conversations = conv_storage.list_conversations(limit=1)
    
    if conversations:
        st.session_state.current_conversation_id = conversations[0]['conversation_id']
    else:
        new_conv_id = conv_storage.create_conversation()
        st.session_state.current_conversation_id = new_conv_id

current_conversation_id = st.session_state.current_conversation_id

# ================= Sidebar =================
with st.sidebar:
    st.markdown(f"## 👤 {username}")
    st.markdown(f"**Role**: {role}<br>**ID**: `{user_id}`", unsafe_allow_html=True)
    
    # ✅ Show data source
    if role == 'user':
        st.markdown(f"""
        <div style="background: #1e3a8a; padding: 0.5rem; border-radius: 5px; margin-top: 0.5rem;">
            <small>📚 Data source: Admin</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Conversation List
    st.markdown("### 💬 Your Conversations")
    
    if 'creating_new_conv' not in st.session_state:
        st.session_state.creating_new_conv = False
    
    if st.button("➕ New Chat", use_container_width=True, type="primary", key="new_chat_btn"):
        if not st.session_state.creating_new_conv:
            st.session_state.creating_new_conv = True
            
            try:
                new_conv_id = conv_storage.create_conversation()
                st.session_state.current_conversation_id = new_conv_id
                st.session_state.messages = []
                
                if 'conv_manager' in st.session_state:
                    del st.session_state.conv_manager
                
                st.session_state.creating_new_conv = False
                st.rerun()
            
            except Exception as e:
                st.error(f"❌ Failed to create conversation: {e}")
                st.session_state.creating_new_conv = False
    
    # List conversations
    conversations = conv_storage.list_conversations(limit=20)
    current_conv_id = st.session_state.get('current_conversation_id')
    
    for conv in conversations:
        conv_id = conv['conversation_id']
        title = conv['title']
        msg_count = conv.get('message_count', 0)
        updated = conv['updated_at'].strftime("%m/%d %H:%M")
        
        is_active = (conv_id == current_conv_id)
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            if st.button(
                f"{'🟢' if is_active else '⚪'} {title}",
                key=f"conv_{conv_id}",
                use_container_width=True,
                type="secondary" if not is_active else "primary"
            ):
                st.session_state.current_conversation_id = conv_id
                
                messages = conv_storage.get_messages(conv_id)
                st.session_state.messages = [
                    {
                        'role': m['role'],
                        'content': m['content'],
                        'metadata': m.get('metadata', {})
                    }
                    for m in messages
                ]
                
                if 'conv_manager' not in st.session_state:
                    st.session_state.conv_manager = ConversationManager(
                        max_history=5,
                        conv_storage=conv_storage,
                        conversation_id=conv_id
                    )
                else:
                    st.session_state.conv_manager.set_conversation(conv_id, conv_storage)
                
                st.rerun()
        
        with col2:
            if st.button("🗑️", key=f"del_{conv_id}"):
                conv_storage.delete_conversation(conv_id)
                
                if conv_id == current_conv_id:
                    new_conv_id = conv_storage.create_conversation()
                    st.session_state.current_conversation_id = new_conv_id
                    st.session_state.messages = []
                
                st.rerun()
        
        st.caption(f"{msg_count} msgs · {updated}")
    
    st.markdown("---")
    
    # Navigation
    st.markdown("### 🧭 Navigation")
    if role == 'admin':
        if st.button("📤 Upload"):
            st.switch_page("pages/upload.py")
        if st.button("🕸️ Graph"):
            st.switch_page("pages/graph.py")
        if st.button("📊 Analytics"):  
            st.switch_page("pages/analytics.py")
    else:
        st.info("👁️ View-only access")
    
    st.markdown("---")
    
    # Settings
    st.markdown("### ⚙️ Settings")
    
    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        options=['auto', 'vector', 'graph', 'hybrid']
    )
    
    top_k = st.slider("Results", 3, 15, 10)
    temperature = st.slider("Temperature", 0.5, 1.0, 0.8, 0.1)
    
    st.markdown("---")
    
    use_history = st.checkbox("Enable history", value=True)
    max_history_turns = st.slider("Max turns", 1, 10, 5)
    show_rewrite = st.checkbox("Show rewrite", value=False)
    
    st.markdown("---")
    
    # Refresh data button
    if st.button("🔄 Refresh Data", use_container_width=True, help="Reload documents from database"):
        st.cache_resource.clear()
        st.success("✅ Cache cleared! Reloading...")
        st.rerun()
    
    st.markdown("---")
    
    if st.button("🚪 Logout"):
        for k in ['authenticated', 'user_id', 'username', 'role']:
            st.session_state.pop(k, None)
        st.switch_page("login.py")

# Verify conversation_id
if current_conversation_id is None:
    st.error("❌ Failed to initialize conversation. Please refresh the page.")
    st.stop()

# Initialize conversation manager
if 'conv_manager' not in st.session_state:
    st.session_state.conv_manager = ConversationManager(
        max_history=max_history_turns,
        conv_storage=conv_storage,
        conversation_id=current_conversation_id
    )
else:
    st.session_state.conv_manager.max_history = max_history_turns
    st.session_state.conv_manager.conversation_id = current_conversation_id
    st.session_state.conv_manager.conv_storage = conv_storage

# Load messages
if 'messages' not in st.session_state:
    messages = conv_storage.get_messages(current_conversation_id)
    st.session_state.messages = [
        {
            'role': m['role'],
            'content': m['content'],
            'metadata': m.get('metadata', {})
        }
        for m in messages
    ]

# ================= Show Stats =================
conv_id_display = current_conversation_id[:12] if current_conversation_id else "N/A"

st.markdown(f"""
<div class="context-preview">
    <strong>📊 Knowledge Base:</strong>
    📄 Docs: {stats['docs']} | 
    🧮 Vectors: {stats['vectors']} | 
    🕸️ Nodes: {stats['nodes']} | 
    💬 Conversation: {conv_id_display}...
    {'<br>📚 <strong>Source: Admin data (shared)</strong>' if role == 'user' else ''}
</div>
""", unsafe_allow_html=True)

# ================= Display Messages =================
for message in st.session_state.messages:
    role_msg = message['role']
    content = message['content']
    
    if role_msg == 'user':
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
        
        if 'metadata' in message:
            meta = message['metadata']
            st.markdown(f"""
            <div style="margin-left: 2rem; margin-top: 0.5rem;">
                <span class="stat-badge badge-chunks">📄 {meta.get('num_chunks', 0)}</span>
                <span class="stat-badge badge-entities">🕸️ {meta.get('num_entities', 0)}</span>
                <span class="stat-badge badge-time">⏱️ {meta.get('retrieval_time_ms', 0)}ms</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Show retrieved chunks
            if 'retrieved_chunks' in message and message['retrieved_chunks']:
                with st.expander(f"📄 Retrieved Documents ({len(message['retrieved_chunks'])})", expanded=False):
                    for i, chunk in enumerate(message['retrieved_chunks'], 1):
                        st.markdown(f"**[{i}] {chunk['filename']}** (Score: {chunk['score']:.3f})")
                        st.text(chunk['content'][:300] + "..." if len(chunk['content']) > 300 else chunk['content'])
                        st.markdown("---")
            
            # Show retrieved entities
            if 'retrieved_entities' in message and message['retrieved_entities']:
                with st.expander(f"🕸️ Retrieved Entities ({len(message['retrieved_entities'])})", expanded=False):
                    for i, entity in enumerate(message['retrieved_entities'], 1):
                        st.markdown(f"**[{i}] {entity['name']}** ({entity['type']}) - Score: {entity['score']:.3f}")
                        
                        if entity.get('description'):
                            st.markdown(f"*{entity['description'][:200]}...*" if len(entity['description']) > 200 else f"*{entity['description']}*")
                        
                        if entity.get('relationships'):
                            st.markdown("**🔗 Relationships:**")
                            for rel in entity['relationships']:
                                rel_type = rel.get('relationship_type', 'RELATED_TO')
                                target = rel.get('target', 'Unknown')
                                category = rel.get('category', 'general')
                                strength = rel.get('strength', 0.0)
                                st.markdown(f"- **{rel_type}** → {target} [{category}] (strength: {strength:.2f})")
                        
                        st.markdown("---")
        
        #  FEEDBACK & EVALUATION 
        msg_index = st.session_state.messages.index(message)
        
        # Initialize states
        if 'feedbacks' not in st.session_state:
            st.session_state.feedbacks = {}
        
        if 'evaluating' not in st.session_state:
            st.session_state.evaluating = {}
        
        feedback_key = f"{current_conversation_id}_{msg_index}"
        
        existing_feedback = feedback_storage.get_feedback(current_conversation_id, msg_index)
        
        # Check what type of feedback exists
        has_manual_feedback = existing_feedback and existing_feedback.get('rating') is not None
        has_auto_evaluation = existing_feedback and existing_feedback.get('auto_evaluated')
        
        # Show existing feedbacks
        if has_manual_feedback or has_auto_evaluation:
            if has_manual_feedback:
                st.markdown(f"""
                <div class="feedback-submitted">
                    ✅ Đánh giá thủ công - Rating: {'⭐' * existing_feedback['rating']} ({existing_feedback['rating']}/5)
                </div>
                """, unsafe_allow_html=True)
            
            if has_auto_evaluation:
                rel_score = existing_feedback.get('relevancy_score', 0)
                faith_score = existing_feedback.get('faithfulness_score', 0)
                resp_time = existing_feedback.get('response_time_ms', 0)
                
                st.markdown(f"""
                <div class="evaluation-container">
                    <strong>📊 Đánh giá tự động:</strong><br><br>
                    <div class="evaluation-badge badge-relevancy">
                        🎯 Độ liên quan: <span class="evaluation-score">{rel_score}/5</span>
                    </div>
                    <div class="evaluation-badge badge-faithfulness">
                        ✅ Độ trung thực: <span class="evaluation-score">{faith_score}/5</span>
                    </div>
                    <div class="evaluation-badge badge-response-time">
                        ⏱️ Thời gian: <span class="evaluation-score">{resp_time:.0f}ms</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show reasons in expander
                rel_reason = existing_feedback.get('relevancy_reason', '')
                faith_reason = existing_feedback.get('faithfulness_reason', '')
                
                if rel_reason or faith_reason:
                    with st.expander("📝 Chi tiết đánh giá tự động", expanded=False):
                        if rel_reason:
                            st.markdown(f"**🎯 Lý do đánh giá độ liên quan:**")
                            st.markdown(f'<div class="evaluation-reason">{rel_reason}</div>', unsafe_allow_html=True)
                        if faith_reason:
                            st.markdown(f"**✅ Lý do đánh giá độ trung thực:**")
                            st.markdown(f'<div class="evaluation-reason">{faith_reason}</div>', unsafe_allow_html=True)
        
        # Show feedback options if not both completed
        if not (has_manual_feedback and has_auto_evaluation):
            with st.expander("💬 Đánh giá câu trả lời này", expanded=False):
                # Create tabs for manual and auto evaluation
                tab1, tab2 = st.tabs(["⭐ Đánh giá thủ công", "🤖 Đánh giá tự động"])
                
                # TAB 1: Manual Feedback
                with tab1:
                    if has_manual_feedback:
                        st.info("✅ Bạn đã đánh giá thủ công câu trả lời này rồi!")
                    else:
                        st.markdown("**Mức độ hài lòng:**")
                        
                        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 3])
                        
                        with col1:
                            if st.button("⭐", key=f"star1_{feedback_key}"):
                                st.session_state.feedbacks[feedback_key] = {'rating': 1}
                        with col2:
                            if st.button("⭐⭐", key=f"star2_{feedback_key}"):
                                st.session_state.feedbacks[feedback_key] = {'rating': 2}
                        with col3:
                            if st.button("⭐⭐⭐", key=f"star3_{feedback_key}"):
                                st.session_state.feedbacks[feedback_key] = {'rating': 3}
                        with col4:
                            if st.button("⭐⭐⭐⭐", key=f"star4_{feedback_key}"):
                                st.session_state.feedbacks[feedback_key] = {'rating': 4}
                        with col5:
                            if st.button("⭐⭐⭐⭐⭐", key=f"star5_{feedback_key}"):
                                st.session_state.feedbacks[feedback_key] = {'rating': 5}
                        
                        current_rating = st.session_state.feedbacks.get(feedback_key, {}).get('rating', 0)
                        
                        if current_rating > 0:
                            st.success(f"Đã chọn: {'⭐' * current_rating} ({current_rating}/5)")
                        
                        feedback_text = st.text_area(
                            "Nhận xét (tùy chọn):",
                            key=f"feedback_text_{feedback_key}",
                            placeholder="Chia sẻ ý kiến của bạn về câu trả lời...",
                            height=80
                        )
                        
                        # Submit button
                        if st.button("📤 Gửi đánh giá thủ công", key=f"submit_manual_{feedback_key}", type="primary"):
                            if current_rating > 0:
                                # Save manual feedback
                                success = feedback_storage.save_feedback(
                                    conversation_id=current_conversation_id,
                                    message_index=msg_index,
                                    rating=current_rating,
                                    feedback_text=feedback_text,
                                    # Preserve auto evaluation if exists
                                    relevancy_score=existing_feedback.get('relevancy_score') if existing_feedback else None,
                                    faithfulness_score=existing_feedback.get('faithfulness_score') if existing_feedback else None,
                                    response_time_ms=existing_feedback.get('response_time_ms') if existing_feedback else None,
                                    auto_evaluated=existing_feedback.get('auto_evaluated', False) if existing_feedback else False,
                                    relevancy_reason=existing_feedback.get('relevancy_reason') if existing_feedback else None,
                                    faithfulness_reason=existing_feedback.get('faithfulness_reason') if existing_feedback else None
                                )
                                
                                if success:
                                    st.success("✅ Cảm ơn bạn đã đánh giá!")
                                    if feedback_key in st.session_state.feedbacks:
                                        del st.session_state.feedbacks[feedback_key]
                                    st.rerun()
                                else:
                                    st.error("❌ Không thể lưu feedback. Vui lòng thử lại.")
                            else:
                                st.warning("⚠️ Vui lòng chọn số sao trước khi gửi!")
                
                # TAB 2: Auto Evaluation
                with tab2:
                    if has_auto_evaluation:
                        st.info("✅ Câu trả lời này đã được đánh giá tự động rồi!")
                    else:
                        st.markdown("""
                        **Hệ thống sẽ tự động đánh giá câu trả lời theo 3 tiêu chí:**
                        - 🎯 **Độ liên quan**: Câu trả lời có liên quan đến câu hỏi không
                        - ✅ **Độ trung thực**: Câu trả lời có trung thực với nguồn tài liệu không
                        - ⏱️ **Thời gian phản hồi**: Tốc độ phản hồi của hệ thống
                        """)
                        
                        # Check if currently evaluating
                        is_evaluating = st.session_state.evaluating.get(feedback_key, False)
                        
                        if is_evaluating:
                            st.markdown("""
                            <div style="text-align: center; padding: 1rem;">
                                <div class="spinner"></div>
                                <span>Đang đánh giá tự động...</span>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            if st.button("🚀 Bắt đầu đánh giá tự động", key=f"eval_{feedback_key}", type="primary"):
                                st.session_state.evaluating[feedback_key] = True
                                
                                try:
                                    # Get question and answer
                                    question = ""
                                    answer = message['content']
                                    
                                    # Find corresponding user question
                                    for i in range(msg_index - 1, -1, -1):
                                        if st.session_state.messages[i]['role'] == 'user':
                                            question = st.session_state.messages[i]['content']
                                            break
                                    
                                    # Get context from metadata
                                    context_text = ""
                                    if 'retrieved_chunks' in message:
                                        context_text = "\n\n".join([
                                            f"[{chunk['filename']}]: {chunk['content']}"
                                            for chunk in message['retrieved_chunks'][:3]
                                        ])
                                    
                                    # Get response time from metadata
                                    response_time_ms = message.get('metadata', {}).get('response_time_ms', 0)
                                    
                                    # Create evaluator
                                    evaluator = ResponseEvaluator()
                                    
                                    # Run evaluation
                                    with st.spinner("🔍 Đang đánh giá..."):
                                        eval_result = asyncio.run(evaluator.evaluate_response(
                                            question=question,
                                            answer=answer,
                                            context=context_text,
                                            response_time_ms=response_time_ms,
                                            llm_func=call_llm_async
                                        ))
                                    
                                    # Save to database (preserve manual feedback if exists)
                                    success = feedback_storage.save_feedback(
                                        conversation_id=current_conversation_id,
                                        message_index=msg_index,
                                        rating=existing_feedback.get('rating') if existing_feedback else None,
                                        feedback_text=existing_feedback.get('feedback_text') if existing_feedback else None,
                                        relevancy_score=eval_result['relevancy_score'],
                                        faithfulness_score=eval_result['faithfulness_score'],
                                        response_time_ms=eval_result['response_time_ms'],
                                        auto_evaluated=True,
                                        relevancy_reason=eval_result['relevancy_reason'],
                                        faithfulness_reason=eval_result['faithfulness_reason']
                                    )
                                    
                                    if success:
                                        st.success("✅ Đánh giá tự động hoàn tất!")
                                        st.session_state.evaluating[feedback_key] = False
                                        st.rerun()
                                    else:
                                        st.error("❌ Không thể lưu kết quả đánh giá")
                                        st.session_state.evaluating[feedback_key] = False
                                
                                except Exception as e:
                                    st.error(f"❌ Lỗi đánh giá: {e}")
                                    st.session_state.evaluating[feedback_key] = False


# ================= Chat Input =================
st.markdown("---")

user_query = st.chat_input("Ask me anything...")

if user_query:
    st.session_state.messages.append({
        'role': 'user',
        'content': user_query
    })
    
    st.markdown(f"""
    <div class="chat-message user-message">
        <strong>👤 You:</strong><br>{user_query}
    </div>
    """, unsafe_allow_html=True)
    
    with st.spinner("🤔 Thinking..."):
        try:
            # Start tracking response time
            start_time = time.time()
            
            # Query rewriting
            original_query = user_query
            if use_history:
                user_query = st.session_state.conv_manager.rewrite_query(
                    user_query,
                    llm_func=call_llm_async
                )
                
                if show_rewrite and user_query != original_query:
                    st.info(f"🔄 Rewritten: {user_query}")
            
            # Retrieval (from admin's data)
            force_mode = None if retrieval_mode == 'auto' else retrieval_mode
            context = retriever.retrieve(
                query=user_query,
                force_mode=force_mode,
                top_k=top_k
            )
            
            # Build prompt
            messages_for_llm = []
            
            system_prompt = """Bạn là một trợ lý AI thông minh và thân thiện, chuyên hỗ trợ người dùng tìm kiếm thông tin từ tài liệu và đồ thị tri thức.

🎯 PHONG CÁCH GIAO TIẾP:
- Trả lời tự nhiên, mượt mà như đang trò chuyện
- Sử dụng ngôn ngữ đơn giản, dễ hiểu, tránh thuật ngữ kỹ thuật không cần thiết
- Thể hiện sự nhiệt tình và quan tâm đến câu hỏi của người dùng
- Có thể sử dụng emoji một cách tinh tế để tạo sự gần gũi (không lạm dụng)

📋 NGUYÊN TẮC TRẢ LỜI:
1. **Xác định đúng đối tượng**: Đọc kỹ câu hỏi để xác định chính xác entity/người được hỏi
2. **Tập trung vào entity đó**: Chỉ trả lời về entity được hỏi, không trả lời về các entity khác
3. **Sử dụng context phù hợp**:
   - Câu hỏi về mối quan hệ → Dùng LOCAL CONTEXT (Knowledge Graph)
   - Câu hỏi về thông tin tài liệu → Dùng GLOBAL CONTEXT (Documents) và trích dẫn [1], [2]
4. **Kiểm tra trước khi trả lời**: Đảm bảo câu trả lời đúng về entity được hỏi

💡 CẤU TRÚC CÂU TRẢ LỜI TỰ NHIÊN:
- Bắt đầu bằng câu mở đầu ngắn gọn, thân thiện
- Trình bày thông tin theo luồng logic, dễ theo dõi
- Sử dụng câu chuyển tiếp mượt mà giữa các ý
- Kết thúc bằng tóm tắt hoặc gợi ý nếu phù hợp

❌ TRÁNH:
- Liệt kê thông tin theo dạng bullet points trừ khi thực sự cần thiết
- Sử dụng cấu trúc câu máy móc như "Theo tài liệu...", "Dựa vào context..."
- Lặp lại câu hỏi của người dùng trong câu trả lời
- Trả lời quá dài dòng hoặc quá ngắn gọn

✅ VÍ DỤ:
Câu hỏi: "Vũ Hoàng Hiệp có quan hệ với những ai?"
❌ Sai: "Dựa vào knowledge graph, tôi thấy Nguyễn Văn A có các mối quan hệ sau..."
✅ Đúng: "Vũ Hoàng Hiệp có mối quan hệ với nhiều người. Anh ấy là đồng nghiệp của Nguyễn Văn A, cùng làm việc tại công ty XYZ. Ngoài ra, anh còn có mối quan hệ hợp tác với..."

Hãy nhớ: Trả lời chính xác nhưng vẫn giữ được sự tự nhiên và thân thiện!"""

            if use_history:
                history_context = st.session_state.conv_manager.get_context_for_llm()
                messages_for_llm.extend(history_context)
            
            user_prompt = f"""
{context.formatted_text}

Question: {user_query}
"""
            messages_for_llm.append({"role": "user", "content": user_prompt})
            
            # Prepare prompt for streaming
            if use_history and len(messages_for_llm) > 1:
                full_prompt = "\n\n".join([
                    f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                    for m in messages_for_llm[:-1]
                ]) + f"\n\nUser: {user_prompt}"
            else:
                full_prompt = user_prompt
            
            # Stream LLM response
            response_placeholder = st.empty()
            response_container = [""]  # Use list to avoid nonlocal binding issues
            
            try:
                # Create async generator for streaming
                async def stream_response():
                    async for chunk in call_llm_stream(
                        prompt=full_prompt,
                        system_prompt=system_prompt,
                        temperature=temperature,
                        max_tokens=2000
                    ):
                        yield chunk
                
                # Display streaming response
                async def collect_response():
                    async for chunk in stream_response():
                        response_container[0] += chunk
                        # Update display with markdown formatting
                        response_placeholder.markdown(f"""
                        <div class="chat-message assistant-message">
                            <strong>🤖 Assistant:</strong><br>{response_container[0]}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Run streaming
                asyncio.run(collect_response())
                response = response_container[0]
                
            except Exception as e:
                st.error(f"❌ LLM call failed: {e}")
                raise
            
            # Calculate response time
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            # Save to MongoDB
            if use_history:
                st.session_state.conv_manager.add_message('user', original_query, save_to_db=True)
                st.session_state.conv_manager.add_message('assistant', response, save_to_db=True)
            
            # Add response time to metadata
            metadata_with_time = context.metadata.copy()
            metadata_with_time['response_time_ms'] = response_time_ms
            
            # Save to UI with full context 
            assistant_message = {
                'role': 'assistant',
                'content': response,
                'metadata': metadata_with_time,
                'retrieved_chunks': [
                    {
                        'content': chunk.content,
                        'filename': chunk.filename,
                        'score': chunk.score,
                        'chunk_id': chunk.chunk_id
                    }
                    for chunk in context.global_chunks[:top_k]  # Sử dụng giá trị từ slider
                ],
                'retrieved_entities': [
                    {
                        'name': entity.entity_name,
                        'type': entity.entity_type,
                        'description': entity.description,
                        'relationships': entity.relationships[:3],  
                        'score': entity.score
                    }
                    for entity in context.local_entities[:top_k]  
                ]
            }
            st.session_state.messages.append(assistant_message)
            
            # Auto-generate title
            if len(st.session_state.messages) == 2:
                conv_storage.auto_generate_title(
                    current_conversation_id,
                    llm_func=call_llm_async
                )
            
            st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error: {e}")

# ================= Footer =================
st.markdown("---")
footer_text = "💬 Multi-Conversation Chat – mini-lightrag v2.2"
if role == 'user':
    footer_text += " (Shared Knowledge Base)"

st.markdown(f"""
<div style="text-align: center; color: #6b7280; font-size: 0.9rem;">
    <p>{footer_text}</p>
</div>
""", unsafe_allow_html=True)