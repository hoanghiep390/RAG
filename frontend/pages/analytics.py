# frontend/pages/analytics.py
""" User Analytics Dashboard - Admin Only"""
import streamlit as st
import sys, os
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from backend.config import get_mongodb
from backend.db.user_manager import load_users

# Auth Check
if not st.session_state.get('authenticated') or st.session_state.get('role') != 'admin':
    st.error("⛔ Admin only")
    if st.button("Back"): st.switch_page("pages/chat.py")
    st.stop()

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide") 

#  CSS
st.markdown("""
<style>
.main{background:#0e1117}
.stat-card{
    background: linear-gradient(135deg, #667eea, #764ba2);
    padding: 1.5rem;
    border-radius: 10px;
    color: #fff;
    text-align: center;
}
.stat-value{
    font-size: 2.5rem;
    font-weight: 700;
}
.stat-label{
    font-size: .9rem;
    opacity: .9;
    margin-top: .5rem;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown(f"## 👤 {st.session_state.username}")
    st.markdown("---")
    if st.button("📤 Upload"): st.switch_page("pages/upload.py")
    if st.button("💬 Chat"): st.switch_page("pages/chat.py")
    if st.button("🕸️ Graph"): st.switch_page("pages/graph.py")
    st.markdown("---")
    
    days = st.selectbox("Period", [7, 30, 90], index=0)
    if st.button("🔄 Refresh"): st.cache_data.clear(); st.rerun()
    st.markdown("---")
    if st.button("🚪 Logout"):
        for k in ['authenticated','user_id','username','role']: st.session_state.pop(k, None)
        st.switch_page("login.py")

# Data Functions
@st.cache_data(ttl=300)
def load_data(days):
    db = get_mongodb()
    end = datetime.now()
    start = end - timedelta(days=days)
    
    # Login logs
    logs = list(db.login_logs.find({'timestamp': {'$gte': start, '$lte': end}}).sort('timestamp', -1))
    
    # User stats
    users = load_users()
    stats = {}
    for un, ud in users.items():
        uid = ud['user_id']
        conv_ids = [c['conversation_id'] for c in db.conversations.find({'user_id': uid}, {'conversation_id': 1})]
        stats[uid] = {
            'username': ud['username'],
            'role': ud['role'],
            'conversations': db.conversations.count_documents({'user_id': uid, 'created_at': {'$gte': start}}),
            'messages': db.messages.count_documents({'conversation_id': {'$in': conv_ids}, 'created_at': {'$gte': start}}),
            'logins': len([l for l in logs if l.get('user_id') == uid]),
            'last_login': max([l['timestamp'] for l in logs if l.get('user_id') == uid], default=None)
        }
    
    # Daily stats
    daily = {}
    for i in range(days+1):
        d = (start.date() + timedelta(days=i)).strftime('%Y-%m-%d')
        daily[d] = 0
    for log in logs:
        d = log['timestamp'].strftime('%Y-%m-%d')
        if d in daily: daily[d] += 1
    
    return logs, stats, daily

# Init collection
try:
    db = get_mongodb()
    db.login_logs.create_index([('user_id', 1), ('timestamp', -1)])
except: pass

# Load data
logs, user_stats, daily_stats = load_data(days)

# Header
st.title("📊 Analytics Dashboard")

# Overview Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"{len(user_stats)}👥 Users", unsafe_allow_html=True)
with col2:
    st.markdown(f"{len(logs)}🔐 Logins", unsafe_allow_html=True)
with col3:
    st.markdown(f"{sum(s['conversations'] for s in user_stats.values())}💬 Chats", unsafe_allow_html=True)
with col4:
    st.markdown(f"{sum(s['messages'] for s in user_stats.values())}✉️ Messages", unsafe_allow_html=True)

st.markdown("---")

# Daily Chart
st.subheader(f"📊 Daily Logins (Last {days} Days)")
dates = sorted(daily_stats.keys())
counts = [daily_stats[d] for d in dates]
fig = go.Figure(go.Bar(x=dates, y=counts, marker=dict(color=counts, colorscale='Blues'), text=counts, textposition='outside'))
fig.update_layout(template='plotly_dark', height=300, showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig, width='stretch')

st.markdown("---")

# User Table
st.subheader("👥 User Activity")
df = pd.DataFrame([{
    'Username': s['username'],
    'Role': s['role'],
    'Logins': s['logins'],
    'Last Login': s['last_login'].strftime('%Y-%m-%d %H:%M') if s['last_login'] else 'Never',
    'Chats': s['conversations'],
    'Messages': s['messages']
} for s in user_stats.values()]).sort_values('Logins', ascending=False)

st.dataframe(df, hide_index=True, height=300)
st.download_button("📥 CSV", df.to_csv(index=False).encode('utf-8'), f"analytics_{datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv")

st.markdown("---")

# Login History
st.subheader("📜 Recent Logins")
filter_user = st.selectbox("Filter", ['All'] + sorted([s['username'] for s in user_stats.values()]))
show_n = st.selectbox("Show", [10, 25, 50], index=0)

filtered = logs if filter_user == 'All' else [l for l in logs if user_stats.get(l.get('user_id'), {}).get('username') == filter_user]

for log in filtered[:show_n]:
    ui = user_stats.get(log.get('user_id'), {})
    st.markdown(f"**🔐 {ui.get('username', 'Unknown')}** ({ui.get('role', 'N/A')}) - {log['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")

st.markdown("---")
st.caption("📊 Analytics v2.2 - Data cached 5min")