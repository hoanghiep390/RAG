# pages/graph.py
import streamlit as st
import networkx as nx
from pathlib import Path
import json
import sys
import os
from datetime import datetime
import streamlit.components.v1 as components
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from backend.core.graph_builder import KnowledgeGraph, merge_admin_graphs

# ================= AUTH =================
if not st.session_state.get('authenticated', False):
    st.switch_page("login.py")
if st.session_state.get('role') != 'admin':
    st.error("Chỉ Admin được xem.")
    if st.button("Upload"): st.switch_page("pages/upload.py")
    st.stop()

user_id = st.session_state.get('user_id', 'admin_00000000')
username = st.session_state.get('username', 'Admin')
st.set_page_config(page_title="LightRAG | Graph", page_icon="Graph", layout="wide")

# ================= CSS =================
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .header-container { background: linear-gradient(90deg, #1e1e1e 0%, #2d2d2d 100%); padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; border-left: 5px solid #dc2626; }
    .header-title { color: #dc2626; font-size: 2rem; font-weight: 700; margin: 0; }
    .admin-badge { background: #dc2626; color: white; padding: 0.3rem 0.8rem; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
    .stat-card { background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center; }
    .stat-value { font-size: 2.5rem; font-weight: 700; margin: 0; }
    .stat-label { font-size: 0.9rem; opacity: 0.9; margin-top: 0.5rem; }
    .entity-card { background: #1e1e1e; padding: 1rem; border-radius: 8px; border-left: 4px solid #dc2626; margin: 0.5rem 0; }
    .entity-type-badge { display: inline-block; padding: 0.3rem 0.8rem; border-radius: 15px; font-size: 0.8rem; font-weight: 600; margin-right: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown(f"""
<div class="header-container">
    <div class="header-title">Combined Knowledge Graph <span class="admin-badge">TỔNG HỢP</span></div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Admin")
    st.markdown(f"**{username}**<br>`{user_id}`")
    st.markdown("---")
    if st.button("Upload"): st.switch_page("pages/upload.py")
    if st.button("Logout", type="secondary"): 
        for k in ['authenticated', 'user_id', 'username', 'role']: st.session_state.pop(k, None)
        st.switch_page("login.py")

# ================= LOAD GRAPH WITH RECOVERY =================
graph_path = Path(f"backend/data/{user_id}/graphs/COMBINED_graph.json")

if not graph_path.exists():
    st.warning("Không tìm thấy graph tổng hợp.")
    with st.spinner("Đang khôi phục..."):
        merged = merge_admin_graphs(user_id)
        if merged and graph_path.exists():
            st.success("Đã khôi phục!")
            st.rerun()
        else:
            st.error("Không thể khôi phục.")
            if st.button("Upload tài liệu"): st.switch_page("pages/upload.py")
            st.stop()

with st.spinner("Đang tải..."):
    try:
        data = json.load(open(graph_path, 'r', encoding='utf-8'))
        kg = KnowledgeGraph()
        gdata = data.get('graph', data)
        for n in gdata.get('nodes', []):
            docs = list(n.get('source_documents', [])) or ['unknown']
            kg.add_entity(n['id'], n.get('type','UNKNOWN'), n.get('description',''), '', docs[0])
        for l in gdata.get('links', []):
            docs = list(l.get('source_documents', [])) or ['unknown']
            kg.add_relationship(l['source'], l['target'], l.get('description',''), l.get('strength',1.0), None, docs[0])
    except Exception as e:
        st.error(f"Lỗi: {e}"); st.stop()

# ================= STATS =================
stats = kg.get_statistics()
col1, col2, col3, col4 = st.columns(4)
with col1: st.markdown(f"<div class='stat-card'><div class='stat-value'>{stats['num_entities']}</div><div class='stat-label'>Entities</div></div>", True)
with col2: st.markdown(f"<div class='stat-card'><div class='stat-value'>{stats['num_relationships']}</div><div class='stat-label'>Relations</div></div>", True)
with col3: st.markdown(f"<div class='stat-card'><div class='stat-value'>{stats['density']:.3f}</div><div class='stat-label'>Density</div></div>", True)
with col4: st.markdown(f"<div class='stat-card'><div class='stat-value'>{stats['avg_degree']:.1f}</div><div class='stat-label'>Avg Degree</div></div>", True)
st.markdown("---")

# ================= TABS =================
tab1, tab2, tab3, tab4 = st.tabs(["Visualization", "Entities", "Relationships", "Documents"])

with tab1:
    if st.button("Tạo Graph", type="primary"):
        with st.spinner("Đang vẽ..."):
            try:
                from pyvis.network import Network
                net = Network(height="700px", bgcolor="#0e1117", font_color="#fff")
                net.set_options("""{ "physics": { "enabled": true, "solver": "forceAtlas2Based" } }""")
                colors = {'PERSON': '#FF6B6B', 'ORGANIZATION': '#4ECDC4', 'LOCATION': '#45B7D1', 'UNKNOWN': '#95A5A6'}
                for n, d in kg.G.nodes(data=True):
                    net.add_node(n, label=n, title=d.get('description','')[:100], color=colors.get(d.get('type'), '#95A5A6'), size=20)
                for s, t, d in kg.G.edges(data=True):
                    net.add_edge(s, t, title=d.get('description',''), value=d.get('strength',1)*2)
                net.save_graph("temp.html")
                components.html(open("temp.html").read(), height=750)
            except Exception as e: st.error(e)

with tab2:
    search = st.text_input("Tìm entity")
    ents = []
    for n, d in kg.G.nodes(data=True):
        if not search or search.lower() in n.lower():
            ents.append({'name': n, 'type': d.get('type','UNKNOWN'), 'desc': d.get('description',''), 'conn': len(list(kg.G.neighbors(n)))})
    ents.sort(key=lambda x: x['conn'], reverse=True)
    for e in ents:
        color = colors.get(e['type'], '#95A5A6')
        with st.expander(f"{e['name']} ({e['conn']} liên kết)"):
            st.markdown(f"<span class='entity-type-badge' style='background:{color}'>{e['type']}</span>", True)
            st.write(e['desc'][:200] or "_Không có_")

with tab3:
    rels = sorted([{'s':s,'t':t,'d':d.get('description',''),'str':d.get('strength',1)} for s,t,d in kg.G.edges(data=True)], key=lambda x: x['str'], reverse=True)
    for r in rels[:50]:
        col = "#dc2626" if r['str'] >= 0.7 else "#f59e0b" if r['str'] >= 0.4 else "#6b7280"
        st.markdown(f"<div class='entity-card'><strong>{r['s']}</strong> → <strong>{r['t']}</strong> <span class='entity-type-badge' style='background:{col}'>S: {r['str']:.2f}</span><br><small>{r['d'] or '_Không có_'}</small></div>", True)

with tab4:
    st.markdown("### Entity theo Tài liệu")
    doc_map = {}
    for n, d in kg.G.nodes(data=True):
        docs = list(d.get('source_documents', ['unknown']))
        for doc in docs:
            clean = doc.split('_graph')[0]
            doc_map.setdefault(clean, []).append({'name': n, 'type': d.get('type','UNKNOWN'), 'desc': d.get('description','')[:100]})
    for doc, ents in doc_map.items():
        with st.expander(f"{doc} ({len(ents)} entities)"):
            for e in ents[:20]:
                c = colors.get(e['type'], '#95A5A6')
                st.markdown(f"<span class='entity-type-badge' style='background:{c}'>{e['type']}</span> **{e['name']}**", True)
                if e['desc']: st.caption(e['desc'])

st.markdown("---")
st.markdown("<p style='text-align:center; color:#6b7280;'>Graph <strong>Combined Knowledge Graph</strong> – Đại học Thủy lợi</p>", unsafe_allow_html=True)