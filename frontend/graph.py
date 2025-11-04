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

# ✅ FIX: Use absolute path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
st.set_page_config(page_title="LightRAG | Graph", page_icon="🕸️", layout="wide")

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
    st.markdown(f"**{username}**<br>`{user_id}`", unsafe_allow_html=True)
    st.markdown("---")
    if st.button("📤 Upload"): st.switch_page("pages/upload.py")
    if st.button("🚪 Logout", type="secondary"): 
        for k in ['authenticated', 'user_id', 'username', 'role']: st.session_state.pop(k, None)
        st.switch_page("login.py")

# ================= LOAD GRAPH WITH RECOVERY =================
graph_path = Path(f"backend/data/{user_id}/graphs/COMBINED_graph.json")

if not graph_path.exists():
    st.warning("⚠️ Không tìm thấy graph tổng hợp.")
    with st.spinner("🔄 Đang khôi phục..."):
        try:
            merged = merge_admin_graphs(user_id)
            if merged and graph_path.exists():
                st.success("✅ Đã khôi phục graph!")
                st.rerun()
            else:
                st.error("❌ Không thể khôi phục. Vui lòng upload tài liệu trước.")
                if st.button("📤 Đi tới Upload"): 
                    st.switch_page("pages/upload.py")
                st.stop()
        except Exception as e:
            st.error(f"❌ Lỗi khôi phục: {str(e)}")
            if st.button("📤 Đi tới Upload"): 
                st.switch_page("pages/upload.py")
            st.stop()

# ✅ FIX: Load graph with error handling
with st.spinner("📊 Đang tải graph..."):
    try:
        data = json.load(open(graph_path, 'r', encoding='utf-8'))
        kg = KnowledgeGraph()
        gdata = data.get('graph', data)
        
        # Load nodes
        for n in gdata.get('nodes', []):
            docs = list(n.get('source_documents', [])) or ['unknown']
            kg.add_entity(
                entity_name=n['id'],
                entity_type=n.get('type', 'UNKNOWN'),
                description=n.get('description', ''),
                source_id='',
                source_document=docs[0]
            )
        
        # Load edges
        for l in gdata.get('links', []):
            docs = list(l.get('source_documents', [])) or ['unknown']
            kg.add_relationship(
                source_entity=l['source'],
                target_entity=l['target'],
                description=l.get('description', ''),
                strength=l.get('strength', 1.0),
                chunk_id=None,
                source_document=docs[0]
            )
    except Exception as e:
        st.error(f"❌ Lỗi tải graph: {str(e)}")
        st.stop()

# ================= STATS =================
stats = kg.get_statistics()
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"<div class='stat-card'><div class='stat-value'>{stats['num_entities']}</div>"
        f"<div class='stat-label'>Entities</div></div>",
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"<div class='stat-card'><div class='stat-value'>{stats['num_relationships']}</div>"
        f"<div class='stat-label'>Relations</div></div>",
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"<div class='stat-card'><div class='stat-value'>{stats['density']:.3f}</div>"
        f"<div class='stat-label'>Density</div></div>",
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"<div class='stat-card'><div class='stat-value'>{stats['avg_degree']:.1f}</div>"
        f"<div class='stat-label'>Avg Degree</div></div>",
        unsafe_allow_html=True
    )

st.markdown("---")

# ✅ FIX: Define entity colors globally (before tabs)
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

# ================= TABS =================
tab1, tab2, tab3, tab4 = st.tabs(["🌐 Visualization", "🏷️ Entities", "🔗 Relationships", "📄 Documents"])

with tab1:
    st.markdown("### Interactive Graph Visualization")
    
    if st.button("🎨 Tạo Graph", type="primary"):
        with st.spinner("🎨 Đang vẽ graph..."):
            try:
                from pyvis.network import Network
                
                # Create network
                net = Network(
                    height="700px",
                    bgcolor="#0e1117",
                    font_color="#ffffff"
                )
                
                # Set physics options
                net.set_options("""
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
                        "stabilization": {
                            "iterations": 100
                        }
                    },
                    "nodes": {
                        "font": {
                            "size": 14,
                            "color": "#ffffff"
                        }
                    },
                    "edges": {
                        "color": {
                            "inherit": false,
                            "color": "#848484"
                        },
                        "smooth": {
                            "type": "continuous"
                        }
                    }
                }
                """)
                
                # Add nodes with colors
                for n, d in kg.G.nodes(data=True):
                    entity_type = d.get('type', 'UNKNOWN')
                    color = ENTITY_COLORS.get(entity_type, '#95A5A6')
                    
                    net.add_node(
                        n,
                        label=n,
                        title=f"<b>{n}</b><br>Type: {entity_type}<br>{d.get('description', '')[:100]}",
                        color=color,
                        size=20
                    )
                
                # Add edges
                for s, t, d in kg.G.edges(data=True):
                    strength = d.get('strength', 1.0)
                    net.add_edge(
                        s,
                        t,
                        title=d.get('description', ''),
                        value=strength * 2,
                        color={'color': '#848484', 'opacity': 0.8}
                    )
                
                # Save and display
                net.save_graph("temp_graph.html")
                with open("temp_graph.html", 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                components.html(html_content, height=750)
                
                st.success("✅ Graph visualization created!")
                
            except Exception as e:
                st.error(f"❌ Lỗi tạo graph: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

with tab2:
    st.markdown("### Entity Browser")
    
    # Search box
    search = st.text_input("🔍 Tìm entity", placeholder="Nhập tên entity...")
    
    # Filter by type
    entity_types = ['ALL'] + sorted(list(set(d.get('type', 'UNKNOWN') for _, d in kg.G.nodes(data=True))))
    selected_type = st.selectbox("Lọc theo loại", entity_types)
    
    # Collect entities
    ents = []
    for n, d in kg.G.nodes(data=True):
        entity_type = d.get('type', 'UNKNOWN')
        
        # Apply filters
        if search and search.lower() not in n.lower():
            continue
        if selected_type != 'ALL' and entity_type != selected_type:
            continue
        
        ents.append({
            'name': n,
            'type': entity_type,
            'desc': d.get('description', ''),
            'conn': len(list(kg.G.neighbors(n)))
        })
    
    # Sort by connections
    ents.sort(key=lambda x: x['conn'], reverse=True)
    
    st.markdown(f"**Tìm thấy {len(ents)} entities**")
    
    # Display entities
    for e in ents:
        color = ENTITY_COLORS.get(e['type'], '#95A5A6')
        
        with st.expander(f"**{e['name']}** ({e['conn']} liên kết)"):
            st.markdown(
                f"<span class='entity-type-badge' style='background:{color}'>{e['type']}</span>",
                unsafe_allow_html=True
            )
            st.write(e['desc'][:300] if e['desc'] else "_Không có mô tả_")
            
            # Show connections
            if e['conn'] > 0:
                st.markdown("**Liên kết với:**")
                neighbors = list(kg.G.neighbors(e['name']))
                for neighbor in neighbors[:10]:  # Show max 10
                    edge_data = kg.G.edges[e['name'], neighbor]
                    st.markdown(f"- **{neighbor}**: {edge_data.get('description', 'N/A')[:100]}")

with tab3:
    st.markdown("### Relationship Browser")
    
    # Collect relationships
    rels = []
    for s, t, d in kg.G.edges(data=True):
        rels.append({
            's': s,
            't': t,
            'd': d.get('description', ''),
            'str': d.get('strength', 1.0)
        })
    
    # Sort by strength
    rels.sort(key=lambda x: x['str'], reverse=True)
    
    st.markdown(f"**Tổng {len(rels)} relationships**")
    
    # Strength filter
    min_strength = st.slider("Lọc theo độ mạnh tối thiểu", 0.0, 1.0, 0.0, 0.1)
    filtered_rels = [r for r in rels if r['str'] >= min_strength]
    
    st.markdown(f"**Hiển thị {len(filtered_rels)} relationships (≥{min_strength})** ")
    
    # Display relationships
    for r in filtered_rels[:100]:  # Show max 100
        # Color based on strength
        if r['str'] >= 0.7:
            col = "#dc2626"  # Red - Strong
        elif r['str'] >= 0.4:
            col = "#f59e0b"  # Orange - Medium
        else:
            col = "#6b7280"  # Gray - Weak
        
        st.markdown(
            f"<div class='entity-card'>"
            f"<strong>{r['s']}</strong> → <strong>{r['t']}</strong> "
            f"<span class='entity-type-badge' style='background:{col}'>S: {r['str']:.2f}</span><br>"
            f"<small>{r['d'] if r['d'] else '_Không có mô tả_'}</small>"
            f"</div>",
            unsafe_allow_html=True
        )

with tab4:
    st.markdown("### Document Analysis")
    
    # Group entities by document
    doc_map = {}
    for n, d in kg.G.nodes(data=True):
        docs = list(d.get('source_documents', ['unknown']))
        for doc in docs:
            # Clean document name
            clean_doc = doc.split('_graph')[0]
            
            if clean_doc not in doc_map:
                doc_map[clean_doc] = []
            
            doc_map[clean_doc].append({
                'name': n,
                'type': d.get('type', 'UNKNOWN'),
                'desc': d.get('description', '')[:100]
            })
    
    st.markdown(f"**Tổng {len(doc_map)} tài liệu**")
    
    # Display by document
    for doc, ents in sorted(doc_map.items(), key=lambda x: len(x[1]), reverse=True):
        with st.expander(f"📄 **{doc}** ({len(ents)} entities)"):
            # Show entity type distribution
            type_counts = {}
            for e in ents:
                type_counts[e['type']] = type_counts.get(e['type'], 0) + 1
            
            st.markdown("**Phân bố entity:**")
            cols = st.columns(len(type_counts))
            for i, (etype, count) in enumerate(type_counts.items()):
                color = ENTITY_COLORS.get(etype, '#95A5A6')
                with cols[i]:
                    st.markdown(
                        f"<div style='text-align:center; padding:0.5rem; background:{color}; "
                        f"border-radius:8px; color:white;'>"
                        f"<b>{etype}</b><br>{count}</div>",
                        unsafe_allow_html=True
                    )
            
            st.markdown("---")
            st.markdown("**Top entities:**")
            
            # Show entities
            for e in ents[:20]:  # Show max 20 per document
                color = ENTITY_COLORS.get(e['type'], '#95A5A6')
                st.markdown(
                    f"<span class='entity-type-badge' style='background:{color}'>{e['type']}</span> "
                    f"**{e['name']}**",
                    unsafe_allow_html=True
                )
                if e['desc']:
                    st.caption(e['desc'])

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#6b7280;'>"
    "🕸️ Graph <strong>Combined Knowledge Graph</strong> – Đại học Thủy lợi"
    "</p>",
    unsafe_allow_html=True
)