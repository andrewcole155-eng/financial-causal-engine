# ==============================================================================
# --- IMPORTS for Streamlit App ---
# ==============================================================================
import json
import logging
import os
import glob  # <-- This will be removed for cloud
import uuid
from typing import Dict, Any, List
import pandas as pd
import networkx as nx
import streamlit as st
from pyvis.network import Network

# --- Local Imports ---
from database_manager import DatabaseManager

# --- Setup structured logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# --- UI HELPER & ANALYSIS FUNCTIONS ---
# ==============================================================================

# ### CLOUD EDIT: REMOVED the load_config() function ###
# We will now get credentials from st.secrets instead of config.json

def format_market_cap(cap: float) -> str:
    """Formats a large number into a readable string with units (B, M, K)."""
    if not isinstance(cap, (int, float)) or cap == 0:
        return "N/A"
    if cap >= 1_000_000_000:
        return f"${cap / 1_000_000_000:.2f}B"
    if cap >= 1_000_000:
        return f"${cap / 1_000_000:.2f}M"
    return f"${cap / 1_000:.2f}K"


# ==============================================================================
# --- NEW/UPDATED FUNCTION 1 of 3: calculate_impact_scores ---
# ==============================================================================
def calculate_impact_scores(graph: nx.DiGraph, start_node: str, event_magnitude: float = 1.0) -> Dict[str, dict]:
    """
    Calculates impact scores by propagating an event through the graph,
    now amplified by the GNN's predicted_risk.
    
    Returns a dictionary where each key is an impacted node and the value is 
    another dict: {'score': float, 'path': list[str]}
    """
    if start_node not in graph:
        return {}

    # Define how much risk amplifies a negative event.
    risk_multipliers = {
        0: 1.0,  # Low Risk
        1: 1.5,  # Medium Risk
        2: 2.0,  # High Risk
    }

    # Store both score and path
    impact_data = {node: {'score': 0.0, 'path': []} for node in graph.nodes}
    impact_data[start_node] = {'score': event_magnitude, 'path': [start_node]}
    
    queue = [start_node]
    visited = {start_node}

    while queue:
        current_node = queue.pop(0)
        
        current_node_data = graph.nodes[current_node]
        current_risk_level = current_node_data.get('predicted_risk') 
        risk_multiplier = risk_multipliers.get(current_risk_level, 1.0)

        for neighbor in graph.neighbors(current_node):
            edge_data = graph.get_edge_data(current_node, neighbor, default={})
            weight = edge_data.get('weight', 0.0)
            relationship_type = edge_data.get('type', 'dependency').lower()

            base_impact = impact_data[current_node]['score'] * weight
            if event_magnitude < 0:
                propagated_impact = base_impact * risk_multiplier
            else:
                propagated_impact = base_impact

            if relationship_type == 'competitor':
                propagated_impact *= -1

            # Check if this new path is stronger than any existing path
            if abs(propagated_impact) > abs(impact_data.get(neighbor, {}).get('score', 0.0)):
                impact_data[neighbor]['score'] = propagated_impact
                # Store the path that led to this impact
                impact_data[neighbor]['path'] = impact_data[current_node]['path'] + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    # Filter for non-zero impacts and format final output
    final_impacts = {
        n: d for n, d in impact_data.items() 
        if d['score'] != 0 and n != start_node
    }
    
    return dict(sorted(final_impacts.items(), key=lambda item: abs(item[1]['score']), reverse=True))


# ==============================================================================
# --- NEW/UPDATED FUNCTION 2 of 3: get_top_ripple_effects ---
# ==============================================================================
@st.cache_data(ttl=600) # Cache this expensive calculation for 10 minutes
def get_top_ripple_effects(_graph: nx.DiGraph, all_events: list, threshold: float) -> pd.DataFrame | None:
    """
    Runs a simulation for ALL recent negative events and finds the
    companies with the worst potential ripple effects.
    We use `_graph` to tell Streamlit not to hash this argument.
    """
    logger.info("Calculating top ripple effects...")
    all_impacts = {} # Stores the *worst* impact found for each company

    # 1. Filter for only significant negative events
    negative_events = [e for e in all_events if e.get('score', 0) < threshold]
    if not negative_events:
        logger.info("No significant negative events to analyze.")
        return None

    # 2. Run simulation for each negative event
    for event in negative_events:
        source_ticker = event['ticker']
        event_score = event['score']
        
        if source_ticker in _graph:
            # Use our new GNN-aware function
            impact_results = calculate_impact_scores(_graph, source_ticker, event_score)
            
            # 3. Store the worst impact for each affected company
            # --- THIS IS THE UPDATED LOGIC ---
            for impacted_company, data in impact_results.items():
                score = data['score']
                path = ' -> '.join(data['path'])

                if impacted_company not in all_impacts or score < all_impacts[impact_company]['Worst Impact Score']:
                    all_impacts[impacted_company] = {
                        'Ticker': impacted_company,
                        'Worst Impact Score': score,
                        'Causal Path': path,  # <-- Store the path
                        'Source Event Ticker': source_ticker,
                        'Source Event Headline': event['headline']
                    }
            # --- END UPDATED LOGIC ---
            
    if not all_impacts:
        logger.info("No ripple effects found from recent events.")
        return None

    # 4. Convert to a DataFrame and sort
    df = pd.DataFrame(all_impacts.values())
    df = df.sort_values(by='Worst Impact Score', ascending=True) # ascending=True shows most negative first
    return df


# ==============================================================================
# --- MAIN STREAMLIT APPLICATION ---
# ==============================================================================

def main():
    """Renders the Streamlit User Interface."""
    st.set_page_config(layout="wide", page_title="Financial Causal Inference Engine")
    st.title("🧠 Financial Causal Inference Engine")

    # ### CLOUD EDIT: We no longer load a config file ###
    # config = load_config() 

    @st.cache_resource
    def get_db_manager():
        """Cached function to initialize the database manager once."""
        
        # ### CLOUD EDIT: Build config from st.secrets instead of config.json ###
        # This checks if secrets are loaded. If not (e.g., local testing), 
        # it gracefully fails.
        if "neo4j" not in st.secrets:
            st.error("Neo4j credentials not found in Streamlit Secrets.")
            st.info("Please add NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD to your secrets.")
            st.stop()
            
        cloud_config = {
            "neo4j": {
                "uri": st.secrets["neo4j"]["uri"],
                "user": st.secrets["neo4j"]["user"],
                "password": st.secrets["neo4j"]["password"]
            }
        }
        # ### END CLOUD EDIT ###
        
        return DatabaseManager(cloud_config) # Pass the cloud config

    db_manager = get_db_manager()
    if not db_manager.is_connected():
        st.error("Fatal: Could not connect to Neo4j. Please ensure the database is running and credentials are correct.")
        st.stop()

    @st.cache_data(ttl=300) # Cache graph data for 5 minutes
    def load_full_graph():
        logger.info("Loading full graph from Neo4j...")
        # Load graph with a default threshold (e.g., 0.1)
        # Adjust this value if 0.1 is too high/low
        return db_manager.get_graph_from_db(weight_threshold=0.1)

    with st.spinner("Loading knowledge graph from database..."):
        financial_graph = load_full_graph()

    if financial_graph.number_of_nodes() == 0:
        st.warning("Knowledge graph is empty. Please run `worker.py` (locally) to populate the *cloud* database.")
        st.stop()

    st.sidebar.success(f"Graph loaded with {financial_graph.number_of_nodes()} nodes and {financial_graph.number_of_edges()} relationships.")
    st.sidebar.info(f"The main graph is pre-filtered to {financial_graph.number_of_edges()} strong relationships. The 'Explore Graph' tab can query the full database.")

    # ### CLOUD EDIT: Removed 'tab_reports' ###
    # The 'reports/' folder does not exist in the Streamlit Cloud
    # ephemeral filesystem. This feature must be re-implemented
    # (e.g., by writing reports to the database itself).
    tab_events, tab_explore, tab_simulate, tab_path = st.tabs([
        "🔔 Recent Events", "🗺️ Explore Graph", "🔬 Simulate Scenarios", "↔️ Causal Pathfinding"
    ])
    # ### END CLOUD EDIT ###

    # --- TAB 1: RECENT EVENTS ---
    with tab_events:
        st.header("🔔 Recently Detected Significant Events")
        st.write("These events are automatically detected and saved by the background worker.")
        
        if st.button("🔄 Refresh Events"):
            st.cache_data.clear() 
            st.cache_resource.clear()
            st.rerun()

        recent_events = db_manager.get_recent_events()
        if not recent_events:
            st.info("No significant events have been detected by the worker yet. Check the worker's logs.")
        else:
            for event in recent_events:
                event_type = "Positive📈" if event['score'] > 0 else "Negative📉"
                with st.expander(f"**{event['timestamp']} - {event_type} for {event['ticker']}**: {event['headline']}"):
                    st.markdown(f"**Sentiment Score:** `{event['score']:.2f}`")
                    st.markdown(f"[Read Full Article]({event['link']})", unsafe_allow_html=True)

                    # --- THIS IS THE UPDATED LOGIC ---
                    impact_results = calculate_impact_scores(financial_graph, event['ticker'], event['score'])
                    if impact_results:
                        st.write("**Potential GNN-Amplified Impacts:**")
                        impact_str = " | ".join([
                            f"**{co}**: {data['score']:.2f}" 
                            for co, data in list(impact_results.items())[:3]
                        ])
                        st.write(impact_str)
                    # --- END UPDATED LOGIC ---

    # --- TAB 2: EXPLORE GRAPH (On-Demand Version) ---
    with tab_explore:
        st.header("🗺️ Interactive Knowledge Graph Explorer")
        st.write("Select a company to load its 1st-degree neighborhood from the graph.")

        all_nodes = sorted(list(financial_graph.nodes()))
        
        selected_company = st.selectbox(
            "Select a company to explore:", 
            all_nodes, 
            key="explore_select"
        )
        
        show_risk = st.toggle("Show GNN Risk Coloring", value=True)

        if st.button("🗺️ Explore Neighborhood"):
            if selected_company:
                with st.spinner(f"Loading neighborhood for {selected_company}..."):
                    # 1. Call our new function
                    neighborhood_graph = db_manager.get_neighborhood_graph(selected_company)

                    if neighborhood_graph.number_of_nodes() > 0:
                        st.info(f"Displaying {neighborhood_graph.number_of_nodes()} companies and {neighborhood_graph.number_of_edges()} relationships.")
                        
                        # 2. Use your existing Pyvis code to render this *small* graph
                        net = Network(height="750px", width="100%", notebook=True, cdn_resources='in_line', directed=True, bgcolor="#222222", font_color="white")
                        net.from_nx(neighborhood_graph)
                        
                        risk_map = {
                            0: {"label": "Low", "color": "#66bb6a"},  # Green
                            1: {"label": "Medium", "color": "#ffa726"}, # Orange
                            2: {"label": "High", "color": "#ef5350"}   # Red
                        }

                        for node in net.nodes:
                            node_data = neighborhood_graph.nodes[node["id"]]
                            node['label'] = node["id"]
                            
                            predicted_risk = node_data.get('predicted_risk') 
                            risk_info = risk_map.get(predicted_risk)
                            
                            title_prefix = ""
                            if show_risk and risk_info:
                                node['color'] = risk_info['color']
                                title_prefix = f"GNN PREDICTED RISK: {risk_info['label'].upper()}\n" \
                                               "----------------------------------\n"
                            
                            if node["id"] == selected_company:
                                node['size'] = 30
                                node['borderWidth'] = 3
                                node['color'] = "#ffffff" # Highlight center node
                                
                            market_cap_str = format_market_cap(node_data.get('market_cap', 0))
                            node["title"] = (
                                f"{title_prefix}"
                                f"Name: {node_data.get('name', 'N/A')}\n"
                                f"Sector: {node_data.get('sector', 'N/A')}\n"
                                f"Market Cap: {market_cap_str}"
                            )

                            if node_data.get('sector') and node_data.get('sector') != 'Discovered':
                                node['group'] = node_data.get('sector')

                        options_str = '{"nodes": {"font": {"size": 18}}, "edges": {"smooth": {"type": "dynamic"}}, "physics": {"barnesHut": {"gravitationalConstant": -20000, "springLength": 350}, "stabilization": {"iterations": 1000}}}'
                        net.set_options(options_str)
                        
                        # ==========================================================
                        # --- ROBUST PYVIS RENDERING (FIX 1 of 2) ---
                        # ==========================================================
                        html_file = f"temp_graph_{uuid.uuid4().hex}.html"
                        try:
                            net.save_graph(html_file)
                            with open(html_file, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                            st.components.v1.html(html_content, height=800, scrolling=True)
                        except Exception as e:
                            st.error(f"Failed to render graph: {e}")
                        finally:
                            if os.path.exists(html_file):
                                os.remove(html_file)
                        # ==========================================================
                        # --- END FIX ---
                        # ==========================================================
                    
                    else:
                        st.warning(f"No relationships found for {selected_company}.")
            else:
                st.warning("Please select a company.")

    # ==============================================================================
    # --- NEW/UPDATED SECTION 3 of 3: tab_simulate ---
    # ==============================================================================
    with tab_simulate:
        st.header("🔬 Impact & Contagion Analysis")

        # --- Automated Analysis ---
        st.subheader("Top Potential Ripple Effects")
        st.write("This automatically runs our GNN-aware simulation on all recent negative events to find the most 'at-risk' companies in the network.")
        
        sensitivity_threshold = st.slider(
            "Event Score Sensitivity (find events with a score *below* this value)",
            min_value=-1.0, max_value=0.0, value=-0.5, step=0.05
        )
        
        all_recent_events = db_manager.get_recent_events()
        
        top_impacts_df = get_top_ripple_effects(financial_graph, all_recent_events, sensitivity_threshold)
        
        if top_impacts_df is None:
            st.info(f"No significant negative events (score < {sensitivity_threshold:.2f}) found to analyze.")
        else:
            st.warning("Found the following potential ripple effects. These companies may be 'at-risk' from contagion.")
            top_impacts_df['Worst Impact Score'] = top_impacts_df['Worst Impact Score'].map('{:,.2f}'.format)
            
            # --- THIS IS THE UPDATED DATAFRAME ---
            st.dataframe(top_impacts_df, use_container_width=True, 
                         column_config={
                             "Source Event Headline": st.column_config.TextColumn("Source Event Headline", max_chars=100),
                             "Causal Path": st.column_config.TextColumn("Causal Path", max_chars=100)
                         })
            # --- END UPDATED DATAFRAME ---

        st.divider() 

        # --- Manual Simulation ---
        st.subheader("Manual 'What-If' Simulation")
        all_nodes = sorted(list(financial_graph.nodes()))
        col1, col2 = st.columns(2)
        selected_company = col1.selectbox("Select a company to trigger an event:", all_nodes)
        
        # --- THIS IS THE UPDATED SLIDER ---
        hypothetical_score = col2.slider(
            "Select a hypothetical event score:",  # <-- 1. Renamed label
            min_value=-1.0, 
            max_value=1.0, 
            value=-0.5,  # <-- 2. Changed default to negative
            step=0.05
        )
        # --- END UPDATED SLIDER ---

        if st.button("💥 Simulate Event"):
            st.subheader(f"Simulating a {'Positive' if hypothetical_score > 0 else 'Negative'} event for {selected_company}")
            
            # --- THIS IS THE UPDATED BUTTON LOGIC ---
            impact_results = calculate_impact_scores(financial_graph, selected_company, hypothetical_score)
            
            sim_col1, sim_col2 = st.columns([1, 2])
            with sim_col1:
                st.subheader("Calculated Impacts")
                if impact_results:
                    # Process the new dictionary structure
                    df_data = []
                    for ticker, data in impact_results.items():
                        df_data.append({
                            'Ticker': ticker,
                            'Impact Score': data['score'],
                            'Causal Path': ' -> '.join(data['path'])
                        })
                    
                    df = pd.DataFrame(df_data)
                    # Display the new DataFrame with Causal Path
                    st.dataframe(df, column_config={
                        "Impact Score": st.column_config.NumberColumn(format="%.2f"),
                        "Causal Path": st.column_config.TextColumn("Causal Path", max_chars=100)
                    })
                    
                    # Store keys for the graph using session state
                    if 'sim_impact_nodes' not in st.session_state:
                        st.session_state['sim_impact_nodes'] = []
                    st.session_state['sim_impact_nodes'] = impact_results.keys()
                
                else:
                    st.info("No downstream impacts found.")
                    st.session_state['sim_impact_nodes'] = []
            
            with sim_col2:
                st.subheader("Focused Impact Graph")
                # Use the session state keys to build the graph
                nodes_to_include = {selected_company, *st.session_state.get('sim_impact_nodes', [])}
                
                subgraph = financial_graph.subgraph(nodes_to_include)
                net = Network(height="500px", width="100%", notebook=True, directed=True, bgcolor="#222222", font_color="white")
                net.from_nx(subgraph)
                for node in net.nodes:
                    node['label'] = node["id"]
                    node["size"] = 20
                    if node["id"] == selected_company:
                        node["color"], node["size"] = "tomato", 30
                net.set_options('{"physics": {"enabled": false}}')
                
                # ==========================================================
                # --- ROBUST PYVIS RENDERING (FIX 2 of 2) ---
                # ==========================================================
                html_file = f"temp_graph_{uuid.uuid4().hex}.html"
                try:
                    net.save_graph(html_file)
                    with open(html_file, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=520, scrolling=True)
                except Exception as e:
                    st.error(f"Failed to render graph: {e}")
                finally:
                    if os.path.exists(html_file):
                        os.remove(html_file)
                # ==========================================================
                # --- END FIX ---
                # ==========================================================

            # --- END UPDATED BUTTON LOGIC ---

    # --- TAB 4: CAUSAL PATHFINDING ---
    with tab_path:
        st.header("↔️ Causal Pathfinding")
        st.write(f"Finds the shortest path between two nodes in the pre-filtered graph (the {financial_graph.number_of_edges()} strongest relationships).")
        all_nodes = sorted(list(financial_graph.nodes()))
        path_col1, path_col2 = st.columns(2)
        start_node = path_col1.selectbox("Start Company:", all_nodes, key="path_start")
        end_node = path_col2.selectbox("End Company:", all_nodes, key="path_end", index=min(1, len(all_nodes)-1))

        if st.button("🗺️ Find Riskiest Path"):
            if start_node != end_node:
                try:
                    all_paths = list(nx.all_simple_paths(financial_graph, source=start_node, target=end_node, cutoff=5))
                    
                    if not all_paths:
                        st.error(f"No path found between {start_node} and {end_node} (within 5 steps).")
                    else:
                        best_path = None
                        max_risk_score = -1

                        for path in all_paths:
                            current_risk_score = 0
                            for node_id in path:
                                node_risk = financial_graph.nodes[node_id].get('predicted_risk', 0)
                                current_risk_score += node_risk
                            
                            if current_risk_score > max_risk_score:
                                max_risk_score = current_risk_score
                                best_path = path
                        
                        st.success(f"Highest-Risk path found: `{' -> '.join(best_path)}`")
                        st.info(f"Total Path Risk Score: **{max_risk_score}**")

                        path_graph = financial_graph.subgraph(best_path)
                        net = Network(height="400px", width="100%", notebook=True, directed=True, bgcolor="#222222", font_color="white")
                        net.from_nx(path_graph)
                        
                        risk_map = {
                            0: {"label": "Low", "color": "#66bb6a"},  # Green
                            1: {"label": "Medium", "color": "#ffa726"}, # Orange
                            2: {"label": "High", "color": "#ef5350"}   # Red
                        }

                        for node_id in best_path:
                            node = net.get_node(node_id)
                            node_data = financial_graph.nodes[node_id]
                            node_risk = node_data.get('predicted_risk')
                            risk_info = risk_map.get(node_risk)
                            
                            node['label'], node['size'] = node_id, 25
                            
                            if risk_info:
                                node['color'] = risk_info['color']
                                node['title'] = f"GNN RISK: {risk_info['label'].upper()}"
                            
                            if node_id == start_node: 
                                node['color'], node['size'] = '#55a630', 30 
                            elif node_id == end_node: 
                                node['color'], node['size'] = '#e63946', 30
                        
                        net.set_options('{"physics": {"enabled": false}}')
                        
                        # ==========================================================
                        # --- ROBUST PYVIS RENDERING (FIX 3 of 3 - Optional but good) ---
                        # ==========================================================
                        html_file = f"temp_graph_{uuid.uuid4().hex}.html"
                        try:
                            net.save_graph(html_file)
                            with open(html_file, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                            st.components.v1.html(html_content, height=420, scrolling=True)
                        except Exception as e:
                            st.error(f"Failed to render graph: {e}")
                        finally:
                            if os.path.exists(html_file):
                                os.remove(html_file)
                        # ==========================================================
                        # --- END FIX ---
                        # ==========================================================

                except nx.NodeNotFound:
                    st.error(f"One of the nodes ({start_node} or {end_node}) was not found in the graph.")
            else:
                st.warning("Please select two different companies.")

    # ### CLOUD EDIT: REMOVED the 'tab_reports' logic ###
    # This code relies on reading from a local 'reports/' folder,
    # which will not exist on Streamlit Cloud.
    
    # --- TAB 5: VIEW REPORTS ---
    # with tab_reports:
    #     st.header("📂 Generated Event Reports")
    #     st.write("Reports generated by the background worker when significant events are detected.")
    #     report_files = sorted(glob.glob("reports/*.txt"), reverse=True)
    #     if not report_files:
    * st.info("No reports have been generated yet.")
    #     else:
    #         selected_report = st.selectbox("Select a report to view:", report_files)
    #         if selected_report:
    #             with open(selected_report, 'r', encoding='utf-8') as f:
    #                 st.code(f.read(), language='text')

if __name__ == "__main__":
    main()