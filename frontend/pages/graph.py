# pages/graph_v2.py
"""
✅ IMPROVED: Graph UI with Search, Filter, Export, Statistics Dashboard
Based on LightRAG original + modern UX practices
"""

import streamlit as st
import networkx as nx
from pathlib import Path
import json
import sys
import os
from datetime import datetime
import streamlit.components.v1 as components
import pandas as pd
from collections import Counter

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
    .search-highlight { background-color: #fbbf24; color: #000; padding: 0.1rem 0.3rem; }
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown(f"""
<div class="header-container">
    <div class="header-title">🕸️ Combined Knowledge Graph <span class="admin-badge">ENHANCED</span></div>
</div>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("## 👤 Admin")
    st.markdown(f"**{username}**<br>`{user_id}`", unsafe_allow_html=True)
    st.markdown("---")
    
    # Navigation
    if st.button("📤 Upload", use_container_width=True): 
        st.switch_page("pages/upload.py")
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("### ⚡ Quick Actions")
    if st.button("🔄 Refresh Graph", use_container_width=True):
        st.rerun()
    
    if st.button("🗑️ Clear Cache", use_container_width=True):
        st.cache_data.clear()
        st.success("Cache cleared!")
    
    st.markdown("---")
    
    if st.button("🚪 Logout", use_container_width=True, type="secondary"):
        for k in ['authenticated', 'user_id', 'username', 'role']: 
            st.session_state.pop(k, None)
        st.switch_page("login.py")

# ================= LOAD GRAPH =================
@st.cache_data
def load_graph_data(user_id: str):
    """Cached graph loading"""
    graph_path = Path(f"backend/data/{user_id}/graphs/COMBINED_graph.json")
    
    if not graph_path.exists():
        return None
    
    try:
        data = json.load(open(graph_path, 'r', encoding='utf-8'))
        kg = KnowledgeGraph()
        gdata = data.get('graph', data)
        
        for n in gdata.get('nodes', []):
            docs = list(n.get('source_documents', [])) or ['unknown']
            kg.add_entity(
                entity_name=n['id'],
                entity_type=n.get('type', 'UNKNOWN'),
                description=n.get('description', ''),
                source_id='',
                source_document=docs[0]
            )
        
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
        
        return kg
    except Exception as e:
        st.error(f"❌ Error loading graph: {e}")
        return None

kg = load_graph_data(user_id)

if kg is None:
    st.warning("⚠️ Không tìm thấy graph tổng hợp.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Khôi phục Graph", use_container_width=True):
            with st.spinner("🔄 Đang khôi phục..."):
                try:
                    merged = merge_admin_graphs(user_id)
                    if merged:
                        st.cache_data.clear()
                        st.success("✅ Đã khôi phục!")
                        st.rerun()
                    else:
                        st.error("❌ Không có dữ liệu để merge.")
                except Exception as e:
                    st.error(f"❌ Lỗi: {str(e)}")
    with col2:
        if st.button("📤 Upload Tài liệu", use_container_width=True):
            st.switch_page("pages/upload.py")
    st.stop()

# ================= STATISTICS =================
stats = kg.get_statistics()

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

# ================= ENTITY COLORS =================
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌐 Visualization", 
    "🔍 Search", 
    "🏷️ Entities", 
    "🔗 Relationships", 
    "📊 Analytics"
])

# ================= TAB 1: VISUALIZATION =================
with tab1:
    st.markdown("### Interactive Graph Visualization")
    
    # Visualization controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        layout_algo = st.selectbox(
            "Layout Algorithm",
            ["Force-Directed", "Circular", "Hierarchical", "Spring"],
            index=0
        )
    
    with col2:
        node_size = st.slider("Node Size", 10, 50, 20, 5)
    
    with col3:
        edge_width = st.slider("Edge Width", 1, 10, 2, 1)
    
    # Physics controls
    with st.expander("⚙️ Physics Settings", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            gravity = st.slider("Gravity", -100, 100, -50, 10)
            spring_length = st.slider("Spring Length", 50, 400, 200, 50)
        with col2:
            central_gravity = st.slider("Central Gravity", 0.0, 0.1, 0.01, 0.01)
            spring_constant = st.slider("Spring Constant", 0.0, 0.2, 0.08, 0.01)
    
    if st.button("🎨 Generate Visualization", type="primary", use_container_width=True):
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
                
                # Apply physics settings
                physics_options = f"""
                {{
                    "physics": {{
                        "enabled": true,
                        "solver": "forceAtlas2Based",
                        "forceAtlas2Based": {{
                            "gravitationalConstant": {gravity},
                            "centralGravity": {central_gravity},
                            "springLength": {spring_length},
                            "springConstant": {spring_constant}
                        }},
                        "stabilization": {{"iterations": 150}}
                    }},
                    "nodes": {{"font": {{"size": 14, "color": "#ffffff"}}}},
                    "edges": {{"color": {{"inherit": false, "color": "#848484"}}, "smooth": {{"type": "continuous"}}}}
                }}
                """
                net.set_options(physics_options)
                
                # Add nodes
                for n, d in kg.G.nodes(data=True):
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
                for s, t, d in kg.G.edges(data=True):
                    strength = d.get('strength', 1.0)
                    desc = d.get('description', '')
                    keywords = d.get('keywords', '')
                    
                    tooltip = f"{desc}"
                    if keywords:
                        tooltip += f"<br>Keywords: {keywords}"
                    
                    net.add_edge(
                        s, t,
                        title=tooltip,
                        value=strength * edge_width,
                        color={'color': '#848484', 'opacity': 0.8}
                    )
                
                # Save and display
                net.save_graph("temp_graph.html")
                with open("temp_graph.html", 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                components.html(html_content, height=800)
                
                st.success("✅ Graph created!")
                
                # Download button
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download HTML",
                        data=html_content,
                        file_name=f"graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html"
                    )
                with col2:
                    # Export GraphML
                    graphml_path = f"graph_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.graphml"
                    nx.write_graphml(kg.G, graphml_path)
                    with open(graphml_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download GraphML",
                            data=f.read(),
                            file_name=graphml_path,
                            mime="application/xml"
                        )
                    os.remove(graphml_path)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                with st.expander("🐛 Debug Info"):
                    st.code(traceback.format_exc())

# ================= TAB 2: SEARCH =================
with tab2:
    st.markdown("### 🔍 Advanced Search")
    
    search_query = st.text_input(
        "Search entities, relationships, or descriptions",
        placeholder="e.g., Apple, technology, produces...",
        key="main_search"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_scope = st.multiselect(
            "Search in",
            ["Entity Names", "Descriptions", "Types", "Relationships"],
            default=["Entity Names", "Descriptions"]
        )
    
    with col2:
        filter_types = st.multiselect(
            "Filter by Type",
            list(ENTITY_COLORS.keys()),
            default=[]
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort by",
            ["Relevance", "Connections (High to Low)", "Name (A-Z)"],
            index=0
        )
    
    if search_query:
        results = []
        query_lower = search_query.lower()
        
        for n, d in kg.G.nodes(data=True):
            score = 0
            entity_type = d.get('type', 'UNKNOWN')
            
            # Apply type filter
            if filter_types and entity_type not in filter_types:
                continue
            
            # Search scoring
            if "Entity Names" in search_scope and query_lower in n.lower():
                score += 10
            if "Descriptions" in search_scope and query_lower in d.get('description', '').lower():
                score += 5
            if "Types" in search_scope and query_lower in entity_type.lower():
                score += 3
            
            if score > 0:
                results.append({
                    'name': n,
                    'type': entity_type,
                    'desc': d.get('description', ''),
                    'conn': kg.G.degree(n),
                    'score': score
                })
        
        # Sort results
        if sort_by == "Relevance":
            results.sort(key=lambda x: x['score'], reverse=True)
        elif sort_by == "Connections (High to Low)":
            results.sort(key=lambda x: x['conn'], reverse=True)
        else:
            results.sort(key=lambda x: x['name'])
        
        st.markdown(f"**Found {len(results)} results**")
        
        if results:
            for r in results[:50]:  # Show max 50
                color = ENTITY_COLORS.get(r['type'], '#95A5A6')
                
                with st.expander(f"**{r['name']}** ({r['conn']} connections) - Score: {r['score']}"):
                    st.markdown(
                        f"<span class='entity-type-badge' style='background:{color}'>{r['type']}</span>",
                        unsafe_allow_html=True
                    )
                    
                    # Highlight search query in description
                    desc = r['desc'][:500]
                    if search_query in desc:
                        desc = desc.replace(search_query, f"<span class='search-highlight'>{search_query}</span>")
                    st.markdown(desc, unsafe_allow_html=True)
                    
                    # Show connections
                    neighbors = list(kg.G.neighbors(r['name']))
                    if neighbors:
                        st.markdown("**Connected to:**")
                        for neighbor in neighbors[:5]:
                            edge_data = kg.G.edges[r['name'], neighbor]
                            st.markdown(f"- **{neighbor}**: {edge_data.get('description', 'N/A')[:100]}")
        else:
            st.info("No results found. Try different search terms.")

# ================= TAB 3: ENTITIES =================
with tab3:
    st.markdown("### 🏷️ Entity Browser")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        entity_search = st.text_input("🔍 Search entity", placeholder="Enter entity name...")
    
    with col2:
        selected_type = st.selectbox("Filter by Type", ['ALL'] + sorted(list(stats['entity_types'].keys())))
    
    # Collect entities
    ents = []
    for n, d in kg.G.nodes(data=True):
        entity_type = d.get('type', 'UNKNOWN')
        
        if entity_search and entity_search.lower() not in n.lower():
            continue
        if selected_type != 'ALL' and entity_type != selected_type:
            continue
        
        ents.append({
            'name': n,
            'type': entity_type,
            'desc': d.get('description', ''),
            'conn': kg.G.degree(n),
            'in_degree': kg.G.in_degree(n),
            'out_degree': kg.G.out_degree(n)
        })
    
    ents.sort(key=lambda x: x['conn'], reverse=True)
    
    st.markdown(f"**Showing {len(ents)} entities**")
    
    # Entity cards with pagination
    items_per_page = 20
    total_pages = (len(ents) - 1) // items_per_page + 1 if ents else 1
    
    page = st.selectbox("Page", range(1, total_pages + 1), format_func=lambda x: f"Page {x}/{total_pages}")
    
    start_idx = (page - 1) * items_per_page
    end_idx = start_idx + items_per_page
    
    for e in ents[start_idx:end_idx]:
        color = ENTITY_COLORS.get(e['type'], '#95A5A6')
        
        with st.expander(f"**{e['name']}** ({e['conn']} connections)"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(
                    f"<span class='entity-type-badge' style='background:{color}'>{e['type']}</span>",
                    unsafe_allow_html=True
                )
                st.write(e['desc'][:300] if e['desc'] else "_No description_")
            
            with col2:
                st.metric("In-Degree", e['in_degree'])
                st.metric("Out-Degree", e['out_degree'])
            
            if e['conn'] > 0:
                st.markdown("**Connected Entities:**")
                neighbors = list(kg.G.neighbors(e['name']))
                for neighbor in neighbors[:10]:
                    edge_data = kg.G.edges[e['name'], neighbor]
                    keywords = edge_data.get('keywords', '')
                    st.markdown(
                        f"- **{neighbor}**: {edge_data.get('description', '')[:100]}"
                    )
                    if keywords:
                        for kw in keywords.split(',')[:3]:
                            st.markdown(
                                f"<span class='keyword-tag'>{kw.strip()}</span>",
                                unsafe_allow_html=True
                            )

# ================= TAB 4: RELATIONSHIPS =================
with tab4:
    st.markdown("### 🔗 Relationship Browser")
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_strength = st.slider("Minimum Strength", 0.0, 1.0, 0.0, 0.1)
    
    with col2:
        keyword_filter = st.text_input("Filter by Keyword", placeholder="e.g., produces, owns...")
    
    # Collect relationships
    rels = []
    for s, t, d in kg.G.edges(data=True):
        strength = d.get('strength', 1.0)
        keywords = d.get('keywords', '')
        
        if strength < min_strength:
            continue
        if keyword_filter and keyword_filter.lower() not in keywords.lower():
            continue
        
        rels.append({
            's': s,
            't': t,
            'd': d.get('description', ''),
            'kw': keywords,
            'str': strength
        })
    
    rels.sort(key=lambda x: x['str'], reverse=True)
    
    st.markdown(f"**Showing {len(rels)} relationships (≥{min_strength})**")
    
    # Display relationships
    for r in rels[:100]:
        if r['str'] >= 0.7:
            col = "#dc2626"
        elif r['str'] >= 0.4:
            col = "#f59e0b"
        else:
            col = "#6b7280"
        
        st.markdown(
            f"<div class='entity-card'>"
            f"<strong>{r['s']}</strong> → <strong>{r['t']}</strong> "
            f"<span class='entity-type-badge' style='background:{col}'>S: {r['str']:.2f}</span><br>"
            f"<small>{r['d'] if r['d'] else '_No description_'}</small><br>",
            unsafe_allow_html=True
        )
        if r['kw']:
            for kw in r['kw'].split(',')[:5]:
                st.markdown(
                    f"<span class='keyword-tag'>{kw.strip()}</span>",
                    unsafe_allow_html=True
                )
        st.markdown("</div>", unsafe_allow_html=True)

# ================= TAB 5: ANALYTICS =================
with tab5:
    st.markdown("### 📊 Graph Analytics")
    
    col1, col2 = st.columns(2)
    
    # Entity type distribution
    with col1:
        st.markdown("#### Entity Type Distribution")
        type_data = pd.DataFrame(
            list(stats['entity_types'].items()),
            columns=['Type', 'Count']
        ).sort_values('Count', ascending=False)
        
        st.bar_chart(type_data.set_index('Type'))
    
    # Top connected entities
    with col2:
        st.markdown("#### Top Connected Entities")
        top_entities = sorted(
            [(n, kg.G.degree(n)) for n in kg.G.nodes()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        top_df = pd.DataFrame(top_entities, columns=['Entity', 'Connections'])
        st.bar_chart(top_df.set_index('Entity'))
    
    st.markdown("---")
    
    # Keyword analysis
    st.markdown("#### 🏷️ Most Common Keywords")
    all_keywords = []
    for _, _, d in kg.G.edges(data=True):
        kw = d.get('keywords', '')
        if kw:
            all_keywords.extend([k.strip() for k in kw.split(',')])
    
    if all_keywords:
        keyword_counts = Counter(all_keywords).most_common(20)
        kw_df = pd.DataFrame(keyword_counts, columns=['Keyword', 'Count'])
        st.bar_chart(kw_df.set_index('Keyword'))
    else:
        st.info("No keywords found in relationships")
    
    st.markdown("---")
    
    # Graph metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Graph Diameter", 
                 nx.diameter(kg.G.to_undirected()) if nx.is_connected(kg.G.to_undirected()) else "N/A")
    
    with col2:
        st.metric("Average Clustering", f"{nx.average_clustering(kg.G.to_undirected()):.3f}")
    
    with col3:
        components = nx.number_weakly_connected_components(kg.G)
        st.metric("Connected Components", components)
    
    st.markdown("---")
    
    # Export options
    st.markdown("#### 💾 Export Data")
    
    export_format = st.radio(
        "Select format",
        ["CSV (Nodes & Edges)", "JSON", "GraphML", "Excel"],
        horizontal=True
    )
    
    if st.button("📥 Generate Export", use_container_width=True):
        if export_format == "CSV (Nodes & Edges)":
            # Nodes CSV
            nodes_data = []
            for n, d in kg.G.nodes(data=True):
                nodes_data.append({
                    'id': n,
                    'type': d.get('type'),
                    'description': d.get('description', '')[:200],
                    'degree': kg.G.degree(n)
                })
            nodes_df = pd.DataFrame(nodes_data)
            
            # Edges CSV
            edges_data = []
            for s, t, d in kg.G.edges(data=True):
                edges_data.append({
                    'source': s,
                    'target': t,
                    'description': d.get('description', ''),
                    'keywords': d.get('keywords', ''),
                    'strength': d.get('strength', 1.0)
                })
            edges_df = pd.DataFrame(edges_data)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "📥 Download Nodes CSV",
                    nodes_df.to_csv(index=False),
                    f"nodes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
            with col2:
                st.download_button(
                    "📥 Download Edges CSV",
                    edges_df.to_csv(index=False),
                    f"edges_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
        
        elif export_format == "JSON":
            json_data = json.dumps(kg.to_dict(), indent=2, ensure_ascii=False)
            st.download_button(
                "📥 Download JSON",
                json_data,
                f"graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#6b7280;'>"
    "🕸️ Graph v2.0 <strong>Enhanced Knowledge Graph</strong> – Đại học Thủy lợi"
    "</p>",
    unsafe_allow_html=True
)