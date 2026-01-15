# frontend/login.py 
import streamlit as st
import hashlib
import json
from pathlib import Path
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.utils.file_utils import ensure_dir
from backend.config import get_mongodb  

try:
    from backend.db.user_manager import load_users, save_users
    USE_MONGODB = True
except:
    USE_MONGODB = False
    
    USER_DATA_FILE = Path("backend/data/users.json")
    ensure_dir(USER_DATA_FILE.parent)
    if not USER_DATA_FILE.exists():
        with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump({}, f, indent=2)
    
    def load_users():
        try:
            with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    
    def save_users(users):
        with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(users, f, indent=2, ensure_ascii=False)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def validate_username(username: str) -> bool:
    return username.isalnum() and 3 <= len(username) <= 20

def validate_password(password: str) -> bool:
    return len(password) >= 6

def create_default_admin():
    users = load_users()
    if users is None:
        users = {}
    
    if "admin" not in users:
        users["admin"] = {
            "username": "admin",
            "password": hash_password("admin123"),
            "user_id": "admin_00000000",
            "role": "admin",
            "created_at": datetime.now().isoformat()
        }
        save_users(users)

create_default_admin()

st.set_page_config(
    page_title="LightRAG | Login",
    layout="centered"
)

st.markdown("""
<style>
    [data-testid="stSidebar"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    section.main > div.block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 100% !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        padding: 2rem;
    }
    .login-container {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        max-width: 400px;
        margin: 0 auto;
        border: 1px solid #667eea;
    }
    .login-title {
        text-align: center;
        color: #667eea;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .login-subtitle {
        text-align: center;
        color: #9ca3af;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.6rem;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .error-msg {
        background-color: #fee2e2;
        color: #dc2626;
        padding: 0.75rem;
        border-radius: 8px;
        border-left: 4px solid #dc2626;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    .success-msg {
        background-color: #dcfce7;
        color: #16a34a;
        padding: 0.75rem;
        border-radius: 8px;
        border-left: 4px solid #16a34a;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
    .info-msg {
        background-color: #dbeafe;
        color: #1d4ed8;
        padding: 0.75rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'username' not in st.session_state:
    st.session_state.username = None
if 'role' not in st.session_state:
    st.session_state.role = None
if 'login_mode' not in st.session_state:
    st.session_state.login_mode = "login"

# Redirect based on role after login
if st.session_state.authenticated:
    if st.session_state.role == 'admin':
        st.switch_page("pages/upload.py")
    else:
        st.switch_page("pages/chat.py")

with st.container():
    st.markdown("<div class='login-container'>", unsafe_allow_html=True)

    if st.session_state.login_mode == "login":
        st.markdown("<h1 class='login-title'>🔒 Đăng Nhập</h1>", unsafe_allow_html=True)
        st.markdown("<p class='login-subtitle'>Chào mừng trở lại! Vui lòng nhập thông tin.</p>", unsafe_allow_html=True)

        with st.form("login_form"):
            username = st.text_input("Tên người dùng", placeholder="Nhập username", key="login_username")
            password = st.text_input("Mật khẩu", type="password", placeholder="Nhập mật khẩu", key="login_password")

            col1, col2 = st.columns([1, 1])
            with col1:
                login_btn = st.form_submit_button("Đăng Nhập", type="primary")
            with col2:
                signup_nav = st.form_submit_button("Đăng Ký")

            if signup_nav:
                st.session_state.login_mode = "signup"
                st.rerun()

            if login_btn:
                if not username or not password:
                    st.markdown("<div class='error-msg'>❌ Vui lòng nhập đầy đủ thông tin.</div>", unsafe_allow_html=True)
                else:
                    users = load_users()
                    if users is None:
                        st.markdown("<div class='error-msg'>❌ Lỗi kết nối cơ sở dữ liệu. Vui lòng thử lại sau.</div>", unsafe_allow_html=True)
                    else:
                        user_key = username.lower()
                        if user_key in users and users[user_key]["password"] == hash_password(password):
                            st.session_state.authenticated = True
                            st.session_state.user_id = users[user_key]["user_id"]
                            st.session_state.username = users[user_key]["username"]
                            st.session_state.role = users[user_key]["role"]
                            
                            try:
                                db = get_mongodb()
                                login_logs = db['login_logs']
                                login_logs.insert_one({
                                    'user_id': st.session_state.user_id,
                                    'username': st.session_state.username,
                                    'role': st.session_state.role,
                                    'timestamp': datetime.now(),
                                    'ip_address': 'N/A',
                                    'user_agent': 'Streamlit App'
                                })
                            except Exception as e:
                                pass
                            
                            # Only create dirs for admin
                            if st.session_state.role == 'admin':
                                ensure_dir(Path(f"backend/data/{st.session_state.user_id}/uploads"))
                                ensure_dir(Path(f"backend/data/{st.session_state.user_id}/chunks"))
                                ensure_dir(Path(f"backend/data/{st.session_state.user_id}/graphs"))

                            st.success(f"✅ Đăng nhập thành công! Chào {st.session_state.role.title()}.")
                            st.rerun()
                        else:
                            st.markdown("<div class='error-msg'>❌ Sai tên đăng nhập hoặc mật khẩu!</div>", unsafe_allow_html=True)

    else:  # Signup mode
        st.markdown("<h1 class='login-title'>📝 Đăng Ký</h1>", unsafe_allow_html=True)
        st.markdown("<p class='login-subtitle'>Tạo tài khoản mới để bắt đầu.</p>", unsafe_allow_html=True)

        with st.form("signup_form"):
            new_username = st.text_input("Tên người dùng", placeholder="3-20 ký tự, chỉ chữ/số", key="signup_username")
            new_password = st.text_input("Mật khẩu", type="password", placeholder="Tối thiểu 6 ký tự", key="signup_password")
            confirm_password = st.text_input("Xác nhận mật khẩu", type="password", placeholder="Nhập lại mật khẩu", key="signup_confirm")

            col1, col2 = st.columns([1, 1])
            with col1:
                signup_btn = st.form_submit_button("Tạo Tài Khoản", type="primary")
            with col2:
                back_nav = st.form_submit_button("Quay Lại")

            if back_nav:
                st.session_state.login_mode = "login"
                st.rerun()

            if signup_btn:
                error = None
                if not validate_username(new_username):
                    error = "❌ Tên người dùng phải từ 3-20 ký tự, chỉ chứa chữ cái và số."
                elif not validate_password(new_password):
                    error = "❌ Mật khẩu phải có ít nhất 6 ký tự."
                elif new_password != confirm_password:
                    error = "❌ Mật khẩu xác nhận không khớp."
                else:
                    users = load_users()
                    if users is None:
                        error = "❌ Lỗi kết nối cơ sở dữ liệu. Vui lòng thử lại sau."
                    elif new_username.lower() in users:
                        error = "❌ Tên người dùng đã tồn tại."

                if error:
                    st.markdown(f"<div class='error-msg'>{error}</div>", unsafe_allow_html=True)
                else:
                    import uuid
                    user_id = f"user_{uuid.uuid4().hex[:8]}"
                    users[new_username.lower()] = {
                        "username": new_username,
                        "password": hash_password(new_password),
                        "user_id": user_id,
                        "role": "user",
                        "created_at": datetime.now().isoformat()
                    }
                    save_users(users)

                    st.markdown("<div class='success-msg'>✅ Đăng ký thành công! Vui lòng đăng nhập.</div>", unsafe_allow_html=True)
                    st.session_state.login_mode = "login"
                    st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

with st.expander("ℹ️ Thông tin tài khoản", expanded=False):
    st.markdown("""
    <div class='info-msg'>
        <strong>Admin:</strong> `admin` / `admin123`<br>
        • Can upload documents<br>
        • Can view knowledge graph<br>
        • Can chat with uploaded documents<br>
        • Can view analytics<br>
        <br>
        <strong>User:</strong> Register to create account<br>
        • Can chat with admin's uploaded documents<br>
        • Cannot upload documents<br>
        • Cannot view graph<br>
        • Cannot view analytics<br>
        • Personal conversation history is saved separately
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; margin-top: 3rem; color: #6b7280; font-size: 0.8rem;'>
    <p>mini-lightrag v2.2 – Shared Knowledge Base – Đại học Thủy lợi</p>
</div>
""", unsafe_allow_html=True)