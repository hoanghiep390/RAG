# frontend/pages/graph.py

import streamlit as st
import networkx as nx
from pathlib import Path
import sys
import os
from datetime import datetime
import streamlit.components.v1 as components
import pandas as pd
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.db.mongo_storage import MongoStorage

#  AUTH 
if not st.session_state.get('authenticated', False):
    st.switch_page("login.py")
if st.session_state.get('role') != 'admin':
    st.error("Chỉ Admin được xem.")
    if st.button("Upload"): st.switch_page("pages/upload.py")
    st.stop()

user_id = st.session_state.get('user_id', 'admin_00000000')
username = st.session_state.get('username', 'Admin')
st.set_page_config(page_title="LightRAG | Graph", page_icon="🕸️", layout="wide")

# CSS 
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
    .stat-card { 
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%); 
        padding: 1.5rem; 
        border-radius: 10px; 
        color: white; 
        text-align: center; 
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3);
        transition: transform 0.2s;
    }
    .stat-card:hover { transform: translateY(-2px); }
    .stat-value { font-size: 2.5rem; font-weight: 700; margin: 0; }
    .stat-label { font-size: 0.9rem; opacity: 0.9; margin-top: 0.5rem; }
    .entity-card { 
        background: #1e1e1e; 
        padding: 1rem; 
        border-radius: 8px; 
        border-left: 4px solid #dc2626; 
        margin: 0.5rem 0; 
    }
    .entity-type-badge { 
        display: inline-block; 
        padding: 0.3rem 0.8rem; 
        border-radius: 15px; 
        font-size: 0.8rem; 
        font-weight: 600; 
        margin-right: 0.5rem; 
    }
    .keyword-tag {
        display: inline-block;
        background: #374151;
        color: #9ca3af;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        margin: 0.2rem;
    }
</style>
""", unsafe_allow_html=True)

# HEADER 
st.markdown(f"""
<div class="header-container">
    <div class="header-title">🕸️ Combined Knowledge Graph <span class="admin-badge">MONGODB</span></div>
</div>
""", unsafe_allow_html=True)

# SIDEBAR 
with st.sidebar:
    st.markdown("## 👤 Admin")
    st.markdown(f"**{username}**<br>`{user_id}`", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.button("📤 Upload", width="stretch"): 
        st.switch_page("pages/upload.py")
    
    st.markdown("---")
    
    st.markdown("### ⚡ Quick Actions")
    if st.button("🔄 Refresh Graph", width="stretch"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    
    if st.button("🚪 Logout", width="stretch", type="secondary"):
        for k in ['authenticated', 'user_id', 'username', 'role']: 
            st.session_state.pop(k, None)
        st.switch_page("login.py")

# LOAD GRAPH FROM MONGODB 
@st.cache_data
def load_graph_from_mongodb(user_id: str):
    """✅ NEW: Load graph from MongoDB"""
    try:
        storage = MongoStorage(user_id)
        graph_data = storage.get_graph()
        
        if not graph_data or not graph_data.get('nodes'):
            return None
        
        # Create NetworkX graph
        G = nx.DiGraph()
        
        # Add nodes
        for node in graph_data['nodes']:
            G.add_node(
                node['id'],
                type=node.get('type', 'UNKNOWN'),
                description=node.get('description', ''),
                sources=node.get('sources', []),
                source_documents=node.get('source_documents', [])
            )
        
        # Add edges
        for link in graph_data['links']:
            G.add_edge(
                link['source'],
                link['target'],
                description=link.get('description', ''),
                keywords=link.get('keywords', ''),
                strength=link.get('strength', 1.0),
                chunks=link.get('chunks', []),
                source_documents=link.get('source_documents', [])
            )
        
        return G
    
    except Exception as e:
        st.error(f"❌ Error loading graph from MongoDB: {e}")
        return None

# Load graph
try:
    G = load_graph_from_mongodb(user_id)
except Exception as e:
    st.error(f"❌ MongoDB connection error: {e}")
    st.info("💡 Make sure MongoDB is running and MONGODB_URI is correct in .env")
    st.stop()

if G is None or G.number_of_nodes() == 0:
    st.warning("⚠️ Không tìm thấy knowledge graph.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📤 Upload Tài liệu", width="stretch"):
            st.switch_page("pages/upload.py")
    with col2:
        st.info("💡 Upload tài liệu để tạo knowledge graph")
    st.stop()

#  STATISTICS 
stats = {
    'num_entities': G.number_of_nodes(),
    'num_relationships': G.number_of_edges(),
    'density': nx.density(G),
    'avg_degree': sum(dict(G.degree()).values()) / max(G.number_of_nodes(), 1) if G.number_of_nodes() > 0 else 0,
    'entity_types': {}
}

# Count entity types
for _, data in G.nodes(data=True):
    entity_type = data.get('type', 'UNKNOWN')
    stats['entity_types'][entity_type] = stats['entity_types'].get(entity_type, 0) + 1

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"<div class='stat-card'><div class='stat-value'>{stats['num_entities']}</div>"
        f"<div class='stat-label'>📊 Entities</div></div>",
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"<div class='stat-card'><div class='stat-value'>{stats['num_relationships']}</div>"
        f"<div class='stat-label'>🔗 Relations</div></div>",
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"<div class='stat-card'><div class='stat-value'>{stats['density']:.3f}</div>"
        f"<div class='stat-label'>🎯 Density</div></div>",
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"<div class='stat-card'><div class='stat-value'>{stats['avg_degree']:.1f}</div>"
        f"<div class='stat-label'>⚡ Avg Degree</div></div>",
        unsafe_allow_html=True
    )

st.markdown("---")

#  ENTITY COLORS 
ENTITY_COLORS = {
    'PERSON': '#FF6B6B',
    'ORGANIZATION': '#4ECDC4',
    'LOCATION': '#45B7D1',
    'EVENT': '#F39C12',
    'PRODUCT': '#9B59B6',
    'CONCEPT': '#1ABC9C',
    'TECHNOLOGY': '#3498DB',
    'UNKNOWN': '#95A5A6'
}

# TABS 
tab1, tab2, tab3, tab4 = st.tabs([
    "🌐 Visualization", 
    "🔍 Search", 
    "🏷️ Entities", 
    "🔗 Relationships"
])

# TAB 1: VISUALIZATION 
with tab1:
    st.markdown("### Interactive Graph Visualization")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        node_size = st.slider("Node Size", 10, 50, 20, 5)
    
    with col2:
        edge_width = st.slider("Edge Width", 1, 10, 2, 1)
    
    with col3:
        max_nodes = st.slider("Max Nodes", 50, 500, 200, 50)
    
    if st.button("🎨 Generate Visualization", type="primary", width="stretch"):
        with st.spinner("🎨 Creating interactive graph..."):
            try:
                from pyvis.network import Network
                
                net = Network(
                    height="750px",
                    width="100%",
                    bgcolor="#0e1117",
                    font_color="#ffffff",
                    directed=True
                )
                
                physics_options = """
                {
                    "physics": {
                        "enabled": true,
                        "solver": "forceAtlas2Based",
                        "forceAtlas2Based": {
                            "gravitationalConstant": -50,
                            "centralGravity": 0.01,
                            "springLength": 200,
                            "springConstant": 0.08
                        },
                        "stabilization": {"iterations": 150}
                    }
                }
                """
                net.set_options(physics_options)
                
                # Limit nodes if too many
                nodes_to_show = list(G.nodes())[:max_nodes]
                
                # Add nodes
                for n in nodes_to_show:
                    d = G.nodes[n]
                    entity_type = d.get('type', 'UNKNOWN')
                    color = ENTITY_COLORS.get(entity_type, '#95A5A6')
                    desc = d.get('description', '')[:200]
                    
                    net.add_node(
                        n,
                        label=n,
                        title=f"<b>{n}</b><br>Type: {entity_type}<br>{desc}...",
                        color=color,
                        size=node_size
                    )
                
                # Add edges
                for s, t in G.edges():
                    if s in nodes_to_show and t in nodes_to_show:
                        d = G.edges[s, t]
                        strength = d.get('strength', 1.0)
                        desc = d.get('description', '')
                        
                        net.add_edge(
                            s, t,
                            title=desc,
                            value=strength * edge_width,
                            color={'color': '#848484', 'opacity': 0.8}
                        )
                
                # Save and display
                net.save_graph("temp_graph.html")
                with open("temp_graph.html", 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                components.html(html_content, height=800)
                
                st.success(f"✅ Graph created with {len(nodes_to_show)} nodes!")
                
                # Download button
                st.download_button(
                    label="📥 Download HTML",
                    data=html_content,
                    file_name=f"graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

#  TAB 2: SEARCH 
with tab2:
    st.markdown("### 🔍 Search Entities")
    
    search_query = st.text_input(
        "Search entities or relationships",
        placeholder="e.g., Apple, technology, produces...",
        key="main_search"
    )
    
    if search_query:
        results = []
        query_lower = search_query.lower()
        
        for n, d in G.nodes(data=True):
            score = 0
            entity_type = d.get('type', 'UNKNOWN')
            
            if query_lower in n.lower():
                score += 10
            if query_lower in d.get('description', '').lower():
                score += 5
            if query_lower in entity_type.lower():
                score += 3
            
            if score > 0:
                results.append({
                    'name': n,
                    'type': entity_type,
                    'desc': d.get('description', ''),
                    'conn': G.degree(n),
                    'score': score
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        
        st.markdown(f"**Found {len(results)} results**")
        
        for r in results[:20]:
            color = ENTITY_COLORS.get(r['type'], '#95A5A6')
            
            with st.expander(f"**{r['name']}** ({r['conn']} connections)"):
                st.markdown(
                    f"<span class='entity-type-badge' style='background:{color}'>{r['type']}</span>",
                    unsafe_allow_html=True
                )
                st.markdown(r['desc'][:300])

#  TAB 3: ENTITIES 
with tab3:
    st.markdown("### 🏷️ Entity Browser")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        entity_search = st.text_input("🔍 Search entity", placeholder="Enter entity name...")
    
    with col2:
        selected_type = st.selectbox("Filter by Type", ['ALL'] + sorted(list(stats['entity_types'].keys())))
    
    ents = []
    for n, d in G.nodes(data=True):
        entity_type = d.get('type', 'UNKNOWN')
        
        if entity_search and entity_search.lower() not in n.lower():
            continue
        if selected_type != 'ALL' and entity_type != selected_type:
            continue
        
        ents.append({
            'name': n,
            'type': entity_type,
            'desc': d.get('description', ''),
            'conn': G.degree(n)
        })
    
    ents.sort(key=lambda x: x['conn'], reverse=True)
    
    st.markdown(f"**Showing {len(ents)} entities**")
    
    for e in ents[:50]:
        color = ENTITY_COLORS.get(e['type'], '#95A5A6')
        
        with st.expander(f"**{e['name']}** ({e['conn']} connections)"):
            st.markdown(
                f"<span class='entity-type-badge' style='background:{color}'>{e['type']}</span>",
                unsafe_allow_html=True
            )
            st.write(e['desc'][:300] if e['desc'] else "_No description_")

#  TAB 4: RELATIONSHIPS
with tab4:
    st.markdown("### 🔗 Relationship Browser")
    
    min_strength = st.slider("Minimum Strength", 0.0, 10.0, 0.0, 0.5)
    
    rels = []
    for s, t, d in G.edges(data=True):
        strength = d.get('strength', 1.0)
        
        if strength < min_strength:
            continue
        
        rels.append({
            's': s,
            't': t,
            'd': d.get('description', ''),
            'kw': d.get('keywords', ''),
            'str': strength
        })
    
    rels.sort(key=lambda x: x['str'], reverse=True)
    
    st.markdown(f"**Showing {len(rels)} relationships (≥{min_strength})**")
    
    for r in rels[:50]:
        if r['str'] >= 5:
            col = "#dc2626"
        elif r['str'] >= 2:
            col = "#f59e0b"
        else:
            col = "#6b7280"
        
        st.markdown(
            f"<div class='entity-card'>"
            f"<strong>{r['s']}</strong> → <strong>{r['t']}</strong> "
            f"<span class='entity-type-badge' style='background:{col}'>S: {r['str']:.1f}</span><br>"
            f"<small>{r['d'] if r['d'] else '_No description_'}</small>",
            unsafe_allow_html=True
        )
        if r['kw']:
            for kw in r['kw'].split(',')[:5]:
                st.markdown(
                    f"<span class='keyword-tag'>{kw.strip()}</span>",
                    unsafe_allow_html=True
                )
        st.markdown("</div>", unsafe_allow_html=True)

# FOOTER
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#6b7280;'>"
    "🕸️ Graph <strong>MongoDB Version</strong> – Đại học Thủy lợi"
    "</p>",
    unsafe_allow_html=True
)