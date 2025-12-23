# frontend/pages/graph.py 

import streamlit as st
import networkx as nx
from pathlib import Path
import sys, os
from datetime import datetime
import streamlit.components.v1 as components
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.db.mongo_storage import MongoStorage

# AUTH
if not st.session_state.get('authenticated', False):
    st.switch_page("login.py")
if st.session_state.get('role') != 'admin':
    st.error("⛔ Chỉ Admin được xem.")
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
        padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem; 
        border-left: 5px solid #dc2626; 
    }
    .header-title { color: #dc2626; font-size: 2rem; font-weight: 700; margin: 0; }
    .stat-card { 
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%); 
        padding: 1.5rem; border-radius: 10px; color: white; text-align: center; 
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3); transition: transform 0.2s;
    }
    .stat-card:hover { transform: translateY(-2px); }
    .stat-value { font-size: 2.5rem; font-weight: 700; margin: 0; }
    .stat-label { font-size: 0.9rem; opacity: 0.9; margin-top: 0.5rem; }
    .badge { 
        display: inline-block; padding: 0.3rem 0.8rem; border-radius: 15px; 
        font-size: 0.8rem; font-weight: 600; margin-right: 0.5rem; 
    }
    .entity-card { 
        background: #1e1e1e; padding: 1rem; border-radius: 8px; 
        border-left: 4px solid #dc2626; margin: 0.5rem 0; 
    }
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown(f"""
<div class="header-container">
    <div class="header-title">🕸️ Knowledge Graph <span class="badge" style="background:#dc2626;color:white;">HYBRID</span></div>
</div>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown(f"## 👤 {username}")
    st.markdown(f"`{user_id}`", unsafe_allow_html=True)
    st.markdown("---")
    
    if st.button("📤 Upload"): st.switch_page("pages/upload.py")
    if st.button("💬 Chat"): st.switch_page("pages/chat.py")
    if st.button("📊 Analytics"):st.switch_page("pages/analytics.py")
    st.markdown("---")
    
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    if st.button("🚪 Logout"):
        for k in ['authenticated', 'user_id', 'username', 'role']: 
            st.session_state.pop(k, None)
        st.switch_page("login.py")

# LOAD GRAPH
@st.cache_data
def load_graph(user_id: str):
    try:
        storage = MongoStorage(user_id)
        graph_data = storage.get_graph()
        
        if not graph_data or not graph_data.get('nodes'):
            return None
        
        G = nx.DiGraph()
        
        for node in graph_data['nodes']:
            G.add_node(
                node['id'],
                type=node.get('type', 'UNKNOWN'),
                description=node.get('description', ''),
                sources=node.get('sources', []),
                source_documents=node.get('source_documents', [])
            )
        
        for link in graph_data['links']:
            G.add_edge(
                link['source'], link['target'],
                keywords=link.get('keywords', ''),
                description=link.get('description', ''),
                strength=link.get('strength', 1.0),
                chunks=link.get('chunks', []),
                source_documents=link.get('source_documents', [])
            )
        
        return G
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

G = load_graph(user_id)

if not G or G.number_of_nodes() == 0:
    st.warning("⚠️ No graph data")
    if st.button("📤 Upload"): st.switch_page("pages/upload.py")
    st.stop()

# STATISTICS
entity_types = {}
rel_keywords = {}

for _, d in G.nodes(data=True):
    t = d.get('type', 'UNKNOWN')
    entity_types[t] = entity_types.get(t, 0) + 1

for _, _, d in G.edges(data=True):
    kw = d.get('keywords', '')
    if kw:
        # Count unique keywords
        for k in kw.split(','):
            k = k.strip()
            if k:
                rel_keywords[k] = rel_keywords.get(k, 0) + 1

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"<div class='stat-card'><div class='stat-value'>{G.number_of_nodes()}</div>"
        f"<div class='stat-label'>📊 Entities</div></div>",
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"<div class='stat-card'><div class='stat-value'>{G.number_of_edges()}</div>"
        f"<div class='stat-label'>🔗 Relations</div></div>",
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"<div class='stat-card'><div class='stat-value'>{len(rel_keywords)}</div>"
        f"<div class='stat-label'>🏷️ Keywords</div></div>",
        unsafe_allow_html=True
    )

with col4:
    avg_strength = sum(d.get('strength', 1.0) for _, _, d in G.edges(data=True)) / max(G.number_of_edges(), 1)
    st.markdown(
        f"<div class='stat-card'><div class='stat-value'>{avg_strength:.1f}</div>"
        f"<div class='stat-label'>💪 Avg Strength</div></div>",
        unsafe_allow_html=True
    )

st.markdown("---")

# COLOR LEGEND
st.markdown("### 🎨 Entity Type Colors")
legend_cols = st.columns(6)

entity_type_colors_display = [
    ('person', 'PERSON', '#FF6B6B'),
    ('organization', 'ORGANIZATION', '#4ECDC4'),
    ('location', 'LOCATION', '#45B7D1'),
    ('event', 'EVENT', '#F39C12'),
    ('product', 'PRODUCT', '#9B59B6'),
    ('concept', 'CONCEPT', '#1ABC9C'),
    ('technology', 'TECHNOLOGY', '#3498DB'),
    ('date', 'DATE', '#E74C3C'),
    ('metric', 'METRIC', '#16A085'),
    ('equipment', 'EQUIPMENT', '#D35400'),
    ('category', 'CATEGORY', '#8E44AD'),
    ('other', 'OTHER', '#95A5A6'),
]

for idx, (key, label, color) in enumerate(entity_type_colors_display):
    with legend_cols[idx % 6]:
        st.markdown(
            f"<div style='background:{color}; padding:0.5rem; border-radius:5px; "
            f"text-align:center; color:white; font-size:0.8rem; margin-bottom:0.5rem;'>"
            f"<b>{label}</b></div>",
            unsafe_allow_html=True
        )

# COLORS - Match với 12 entity types trong extraction.py
ENTITY_COLORS = {
    # Core types
    'PERSON': '#FF6B6B',       
    'person': '#FF6B6B',
    
    'ORGANIZATION': '#4ECDC4',     
    'organization': '#4ECDC4',
    
    'LOCATION': '#45B7D1',         
    'location': '#45B7D1',
    
    'EVENT': '#F39C12',           
    'event': '#F39C12',
    
    'PRODUCT': '#9B59B6',          
    'product': '#9B59B6',
    
    'CONCEPT': '#1ABC9C',          
    'concept': '#1ABC9C',
    
    'TECHNOLOGY': '#3498DB',       
    'technology': '#3498DB',
    
    # Extended types
    'DATE': '#E74C3C',             
    'date': '#E74C3C',
    
    'METRIC': '#16A085',          
    'metric': '#16A085',
    
    'EQUIPMENT': '#D35400',        
    'equipment': '#D35400',
    
    'CATEGORY': '#8E44AD',         
    'category': '#8E44AD',
    
    'OTHER': '#95A5A6',            
    'other': '#95A5A6',
    'UNKNOWN': '#95A5A6'
}

# TABS
tab1, tab2, tab3, tab4 = st.tabs(["🌐 Viz", "🔍 Search", "🏷️ Entities", "🔗 Relations"])

# TAB 1: VIZ
with tab1:
    st.markdown("### Interactive Graph")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        node_size = st.slider("Node Size", 10, 50, 20, 5)
    with col2:
        edge_width = st.slider("Edge Width", 1, 10, 2, 1)
    with col3:
        max_nodes = st.slider("Max Nodes", 50, 500, 200, 50)
    
    # 🆕 FILTERS
    st.markdown("#### 🎯 Filters")
    col1, col2 = st.columns(2)
    
    with col1:
        filter_entity_type = st.multiselect(
            "Entity Types",
            options=['ALL'] + sorted(entity_types.keys()),
            default=['ALL']
        )
    
    with col2:
        filter_keywords = st.multiselect(
            "Keywords",
            options=['ALL'] + sorted(rel_keywords.keys())[:50],  # Limit to top 50
            default=['ALL']
        )
    
    min_strength_filter = st.slider("Min Edge Strength", 0.0, 10.0, 0.0, 0.5)
    
    if st.button("🎨 Generate", type="primary"):
        with st.spinner("Creating graph..."):
            try:
                from pyvis.network import Network
                
                net = Network(height="750px", width="100%", bgcolor="#0e1117", 
                             font_color="#ffffff", directed=True)
                
                net.set_options("""{"physics": {"enabled": true, "solver": "forceAtlas2Based",
                    "forceAtlas2Based": {"gravitationalConstant": -50, "centralGravity": 0.01,
                    "springLength": 200, "springConstant": 0.08}, "stabilization": {"iterations": 150}}}""")
                
                # Filter nodes
                nodes_to_show = []
                for n, d in G.nodes(data=True):
                    if 'ALL' not in filter_entity_type and d.get('type') not in filter_entity_type:
                        continue
                    nodes_to_show.append(n)
                    if len(nodes_to_show) >= max_nodes:
                        break
                
                # Add nodes
                for n in nodes_to_show:
                    d = G.nodes[n]
                    color = ENTITY_COLORS.get(d.get('type', 'UNKNOWN'), '#95A5A6')
                    desc = d.get('description', '')[:200]
                    
                    net.add_node(
                        n, label=n,
                        title=f"<b>{n}</b><br>Type: {d.get('type')}<br>{desc}...",
                        color=color, size=node_size
                    )
                
                # Filter & add edges
                for s, t, d in G.edges(data=True):
                    if s not in nodes_to_show or t not in nodes_to_show:
                        continue
                    
                    keywords = d.get('keywords', '')
                    strength = d.get('strength', 1.0)
                    
                    # Filter by strength
                    if strength < min_strength_filter:
                        continue
                    
                    # Filter by keywords
                    if 'ALL' not in filter_keywords:
                        kw_list = [k.strip() for k in keywords.split(',') if k.strip()]
                        if not any(k in filter_keywords for k in kw_list):
                            continue
                    
                    # Edge color based on strength
                    if strength >= 5.0:
                        edge_color = '#ef4444'  # Red - strong
                    elif strength >= 3.0:
                        edge_color = '#f59e0b'  # Orange - medium
                    else:
                        edge_color = '#6b7280'  # Gray - weak
                    
                    net.add_edge(
                        s, t,
                        title=f"<b>Keywords:</b> {keywords}<br><b>Strength:</b> {strength:.1f}<br>{d.get('description', '')}",
                        value=strength * edge_width,
                        color={'color': edge_color, 'opacity': 0.8}
                    )
                
                net.save_graph("temp_graph.html")
                with open("temp_graph.html", 'r', encoding='utf-8') as f:
                    html = f.read()
                
                components.html(html, height=800)
                
                st.success(f"✅ {len(nodes_to_show)} nodes, {net.num_edges()} edges")
                
                st.download_button(
                    "📥 Download",
                    data=html,
                    file_name=f"graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
                
            except Exception as e:
                st.error(f"❌ {e}")

# TAB 2: SEARCH
with tab2:
    st.markdown("### 🔍 Search")
    
    query = st.text_input("Search", placeholder="entity, type, relation...")
    
    if query:
        results = []
        q = query.lower()
        
        for n, d in G.nodes(data=True):
            score = 0
            if q in n.lower(): score += 10
            if q in d.get('description', '').lower(): score += 5
            if q in d.get('type', '').lower(): score += 3
            
            if score > 0:
                results.append({
                    'name': n, 'type': d.get('type'), 'desc': d.get('description', ''),
                    'conn': G.degree(n), 'score': score
                })
        
        results.sort(key=lambda x: x['score'], reverse=True)
        st.markdown(f"**{len(results)} results**")
        
        for r in results[:20]:
            with st.expander(f"**{r['name']}** ({r['conn']} conn)"):
                st.markdown(f"<span class='badge' style='background:{ENTITY_COLORS.get(r['type'], '#95A5A6')}'>{r['type']}</span>", unsafe_allow_html=True)
                st.write(r['desc'][:300])

# TAB 3: ENTITIES
with tab3:
    st.markdown("### 🏷️ Entities")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        entity_search = st.text_input("🔍 Search", placeholder="name...")
    with col2:
        selected_type = st.selectbox("Type", ['ALL'] + sorted(entity_types.keys()))
    
    ents = []
    for n, d in G.nodes(data=True):
        if entity_search and entity_search.lower() not in n.lower():
            continue
        if selected_type != 'ALL' and d.get('type') != selected_type:
            continue
        ents.append({'name': n, 'type': d.get('type'), 'desc': d.get('description', ''), 'conn': G.degree(n)})
    
    ents.sort(key=lambda x: x['conn'], reverse=True)
    st.markdown(f"**{len(ents)} entities**")
    
    for e in ents[:50]:
        with st.expander(f"**{e['name']}** ({e['conn']} conn)"):
            st.markdown(f"<span class='badge' style='background:{ENTITY_COLORS.get(e['type'], '#95A5A6')}'>{e['type']}</span>", unsafe_allow_html=True)
            st.write(e['desc'][:300] if e['desc'] else "_No desc_")

# TAB 4: RELATIONS
with tab4:
    st.markdown("### 🔗 Relationships")
    
    col1, col2 = st.columns(2)
    with col1:
        min_strength = st.slider("Min Strength", 0.0, 10.0, 0.0, 0.5)
    with col2:
        filter_keyword_rel = st.selectbox("Keyword", ['ALL'] + sorted(rel_keywords.keys())[:50])
    
    rels = []
    for s, t, d in G.edges(data=True):
        strength = d.get('strength', 1.0)
        keywords = d.get('keywords', '')
        
        if strength < min_strength:
            continue
        
        # Filter by keyword
        if filter_keyword_rel != 'ALL':
            kw_list = [k.strip() for k in keywords.split(',') if k.strip()]
            if filter_keyword_rel not in kw_list:
                continue
        
        rels.append({
            's': s, 't': t, 'kw': keywords,
            'd': d.get('description', ''), 'str': strength
        })
    
    rels.sort(key=lambda x: x['str'], reverse=True)
    st.markdown(f"**{len(rels)} relationships**")
    
    for r in rels[:50]:
        # Color based on strength
        if r['str'] >= 5.0:
            strength_color = '#ef4444'  # Red
        elif r['str'] >= 3.0:
            strength_color = '#f59e0b'  # Orange
        else:
            strength_color = '#6b7280'  # Gray
        
        st.markdown(f"""
        <div class='entity-card'>
            <strong>{r['s']}</strong> → <strong>{r['t']}</strong><br>
            <span class='badge' style='background:#3b82f6'>{r['kw']}</span>
            <span class='badge' style='background:{strength_color}'>Strength: {r['str']:.1f}</span><br>
            <small>{r['d'] if r['d'] else '_No description_'}</small>
        </div>
        """, unsafe_allow_html=True)

# FOOTER
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#6b7280;'>"
    "🕸️ Graph <strong>HYBRID MODE</strong> – Static + Dynamic Relationships"
    "</p>",
    unsafe_allow_html=True
)