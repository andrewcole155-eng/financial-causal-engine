# ==============================================================================
# --- IMPORTS for Streamlit App ---
# ==============================================================================
import json
import logging
import os
import glob
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

def load_config(config_file: str = "config.json") -> Dict[str, Any]:
    """Loads all configurations from a JSON file."""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Fatal: Error loading configuration file '{config_file}': {e}")
        st.stop()
        return {}


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
    """
    if start_node not in graph:
        return {}

    risk_multipliers = {
        0: 1.0,  # Low Risk
        1: 1.5,  # Medium Risk
        2: 2.0,  # High Risk
    }

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

            if abs(propagated_impact) > abs(impact_data.get(neighbor, {}).get('score', 0.0)):
                impact_data[neighbor]['score'] = propagated_impact
                impact_data[neighbor]['path'] = impact_data[current_node]['path'] + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    final_impacts = {
        n: d for n, d in impact_data.items() 
        if d['score'] != 0 and n != start_node
    }
    
    return dict(sorted(final_impacts.items(), key=lambda item: abs(item[1]['score']), reverse=True))


# ==============================================================================
# --- CACHED FUNCTIONS ---
# ==============================================================================

# ### CACHE FIX: get_db_manager is now defined globally, outside main()
@st.cache_resource
def get_db_manager():
    """Cached function to initialize the database manager once."""
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
    
    return DatabaseManager(cloud_config)

# ### CACHE FIX: This function now has NO arguments.
@st.cache_data(ttl=86400) # Keep the cache in case the file is large
def get_full_graph():
    """Cached function to load the pre-computed graph from a file."""
    
    # --- START FIX: Use an absolute path ---
    try:
        # Get the absolute path of the directory this script is in
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        # Join it with the filename to get the full, robust path
        GRAPH_FILE_PATH = os.path.join(SCRIPT_DIR, "financial_graph.gml")
    except Exception as e:
        logger.warning(f"Could not determine script path, falling back to relative path. Error: {e}")
        # Fallback just in case __file__ is not available
        GRAPH_FILE_PATH = "financial_graph.gml" 
    # --- END FIX ---
    
    logger.info(f"Attempting to load pre-computed graph from {GRAPH_FILE_PATH}...")
    try:
        # Load using the new, full path
        graph = nx.read_gml(GRAPH_FILE_PATH)
        
    except FileNotFoundError:
        logger.error(f"FATAL: Graph file not found at {GRAPH_FILE_PATH}.")
        st.error(f"Graph file not found. Looked for: {GRAPH_FILE_PATH}")
        
        # Add debugging to show what's in the directory
        try:
            st.warning(f"Files found in the script's directory ({SCRIPT_DIR}):")
            st.code(os.listdir(SCRIPT_DIR))
        except Exception as e:
            st.error(f"Could not list directory contents: {e}")
            
        st.stop()
        return None
        
    except Exception as e:
        logger.error(f"Error loading graph file: {e}")
        st.error(f"Error loading graph file: {e}")
        st.stop()
        return None
        
    if graph.number_of_nodes() == 0:
        logger.warning("Loaded graph is empty.")
        return None
        
    logger.info(f"Graph loaded with {graph.number_of_nodes()} nodes.")
    return graph

# ### CACHE FIX: This function no longer takes db_manager as an argument.
@st.cache_data(ttl=600) 
def get_top_ripple_effects(all_events: list, threshold: float) -> pd.DataFrame | None:
    """
    Runs a simulation for ALL recent negative events and finds the
    companies with the worst potential ripple effects.
    """
    
    # ### CACHE FIX: Call the cached data function *inside*
    financial_graph = get_full_graph()
    if financial_graph is None:
        st.warning("Knowledge graph is empty. Please run `worker.py` (locally) to populate the *cloud* database.")
        return None

    logger.info("Calculating top ripple effects...")
    all_impacts = {} 

    negative_events = [e for e in all_events if e.get('score', 0) < threshold]
    if not negative_events:
        logger.info("No significant negative events to analyze.")
        return None

    for event in negative_events:
        source_ticker = event['ticker']
        event_score = event['score']
        
        if source_ticker in financial_graph:
            impact_results = calculate_impact_scores(financial_graph, source_ticker, event_score)
            
            for impacted_company, data in impact_results.items():
                score = data['score']
                path = ' -> '.join(data['path'])

                if impacted_company not in all_impacts or score < all_impacts[impacted_company]['Worst Impact Score']:
                    all_impacts[impacted_company] = {
                        'Ticker': impacted_company,
                        'Worst Impact Score': score,
                        'Causal Path': path,
                        'Source Event Ticker': source_ticker,
                        'Source Event Headline': event['headline']
                    }
            
    if not all_impacts:
        logger.info("No ripple effects found from recent events.")
        return None

    df = pd.DataFrame(all_impacts.values())
    df = df.sort_values(by='Worst Impact Score', ascending=True)
    return df


# ==============================================================================
# --- MAIN STREAMLIT APPLICATION ---
# ==============================================================================

def main():
    """Renders the Streamlit User Interface."""

    st.set_page_config(layout="wide", page_title="Financial Causal Inference Engine")
    st.title("🧠 Financial Causal Inference Engine")

    # Call the cached function to get the db connection
    db_manager = get_db_manager()
    
    if not db_manager.is_connected():
        st.error("Fatal: Could not connect to Neo4j. Please ensure the database is running and credentials are correct.")
        st.info("This can also be caused by a 'Paused' Free Tier database on Neo4j Aura. Please check your Aura dashboard.")
        st.stop()

    st.sidebar.success(f"Connected to Neo4j.")
    st.sidebar.info(f"The knowledge graph will be loaded on-demand when you open a tab.")

    tab_events, tab_explore, tab_simulate, tab_path, tab_reports = st.tabs([
        "🔔 Recent Events", "🗺️ Explore Graph", "🔬 Simulate Scenarios", "↔️ Causal Pathfinding", "📂 View Reports"
    ])

    # --- TAB 1: RECENT EVENTS ---
    with tab_events:
        st.header("🔔 Recently Detected Significant Events")
        st.write("These events are automatically detected and saved by the background worker.")
        
        if st.button("🔄 Refresh Events"):
            st.cache_data.clear() 
            st.cache_resource.clear()
            st.rerun()

        # This function is NOT cached, it runs live and should be fast.
        recent_events = db_manager.get_recent_events()
        
        if not recent_events:
            st.info("No significant events have been detected by the worker yet. Check the worker's logs.")
        else:
            # --- START FIX ---
            # We no longer load the full graph or calculate impacts here.
            # This loop will now be fast and display all events.
            
            st.info(f"Displaying the {len(recent_events)} most recent events.")

            for event in recent_events:
                event_type = "Positive📈" if event['score'] > 0 else "Negative📉"
                with st.expander(f"**{event['timestamp']} - {event_type} for {event['ticker']}**: {event['headline']}"):
                    st.markdown(f"**Sentiment Score:** `{event['score']:.2f}`")
                    st.markdown(f"[Read Full Article]({event['link']})", unsafe_allow_html=True)
                    st.info(f"To see the potential impact of this event, go to the 'Simulate Scenarios' tab and run a simulation for {event['ticker']}.")
            
            # --- END FIX ---

    # --- TAB 2: EXPLORE GRAPH (On-Demand Version) ---
    with tab_explore:
        st.header("🗺️ Interactive Knowledge Graph Explorer")
        st.write("Select a company to load its **Top 25** strongest relationships.") # Updated text

        # This tab will still trigger the slow graph load, but only when a user
        # clicks on it, which is acceptable.
        with st.spinner("Loading full graph for explorer..."):
            financial_graph = get_full_graph()
        
        if financial_graph is None:
             st.warning("Knowledge graph is empty. Please run `worker.py` (locally) to populate the *cloud* database.")
        else:
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
                        # This function is NOT cached, it runs live
                        # This query is now optimized to LIMIT 25 in database_manager.py
                        neighborhood_graph = db_manager.get_neighborhood_graph(selected_company)

                        if neighborhood_graph.number_of_nodes() > 0:
                            
                            net = Network(height="750px", width="100%", notebook=True, cdn_resources='in_line', directed=True, bgcolor="#222222", font_color="white")
                            net.from_nx(neighborhood_graph)
                            
                            risk_map = {
                                0: {"label": "Low", "color": "#66bb6a"},  # Green
                                1: {"label": "Medium", "color": "#ffa726"}, # Orange
                                2: {"label": "High", "color": "#ef5350"}  # Red
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
                                    node['color'] = "#ffffff" 
                                    
                                market_cap_str = format_market_cap(node_data.get('market_cap', 0))
                                node["title"] = (
                                    f"{title_prefix}"
                                    f"Name: {node_data.get('name', 'N/A')}\n"
                                    f"Sector: {node_data.get('sector', 'N/A')}\n"
                                    f"Market Cap: {market_cap_str}"
                                )

                                if node_data.get('sector') and node_data.get('sector') != 'Discovered':
                                    node['group'] = node_data.get('sector')
                            
                            # --- START FIX (from before): DYNAMICALLY SET PHYSICS ---
                            
                            # This threshold is now fine because we only query 25 edges
                            RELATIONSHIP_THRESHOLD = 75 
                            
                            if neighborhood_graph.number_of_edges() > RELATIONSHIP_THRESHOLD:
                                st.warning(f"Graph is large ({neighborhood_graph.number_of_edges()} relationships). Displaying with a simplified, static layout to prevent crashing.")
                                options_str = '{"nodes": {"font": {"size": 18}}, "edges": {"smooth": {"type": "dynamic"}}, "physics": {"enabled": false}}'
                            
                            else:
                                st.info(f"Displaying {neighborhood_graph.number_of_nodes()} companies and {neighborhood_graph.number_of_edges()} relationships.")
                                options_str = '{"nodes": {"font": {"size": 18}}, "edges": {"smooth": {"type": "dynamic"}}, "physics": {"barnesHut": {"gravitationalConstant": -20000, "springLength": 350}, "stabilization": {"iterations": 1000}}}'
                            
                            net.set_options(options_str)
                            
                            # --- END FIX ---
                            
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
        
        # This function is NOT cached, it runs live
        all_recent_events = db_manager.get_recent_events()
        
        # This will still be slow, but it's acceptable on a secondary tab.
        with st.spinner("Calculating top ripple effects (this may take a while)..."):
            top_impacts_df = get_top_ripple_effects(all_recent_events, sensitivity_threshold)
        
        if top_impacts_df is None:
            st.info(f"No significant negative events (score < {sensitivity_threshold:.2f}) or empty graph found.")
        else:
            st.warning("Found the following potential ripple effects. These companies may be 'at-risk' from contagion.")
            top_impacts_df['Worst Impact Score'] = top_impacts_df['Worst Impact Score'].map('{:,.2f}'.format)
            
            st.dataframe(top_impacts_df, use_container_width=True, 
                         column_config={
                             "Source Event Headline": st.column_config.TextColumn("Source Event Headline", max_chars=100),
                             "Causal Path": st.column_config.TextColumn("Causal Path", max_chars=100)
                         })

        st.divider() 

        # --- Manual Simulation ---
        st.subheader("Manual 'What-If' Simulation")

        with st.spinner("Loading full graph for simulation..."):
            financial_graph = get_full_graph()

        if financial_graph is None:
             st.warning("Knowledge graph is empty. Please run `worker.py` (locally) to populate the *cloud* database.")
        else:
            all_nodes = sorted(list(financial_graph.nodes()))
            col1, col2 = st.columns(2)
            selected_company = col1.selectbox("Select a company to trigger an event:", all_nodes)
            
            hypothetical_score = col2.slider(
                "Select a hypothetical event score:",
                min_value=-1.0, 
                max_value=1.0, 
                value=-0.5,
                step=0.05
            )

            if st.button("💥 Simulate Event"):
                st.subheader(f"Simulating a {'Positive' if hypothetical_score > 0 else 'Negative'} event for {selected_company}")
                
                impact_results = calculate_impact_scores(financial_graph, selected_company, hypothetical_score)
                
                sim_col1, sim_col2 = st.columns([1, 2])
                with sim_col1:
                    st.subheader("Calculated Impacts")
                    if impact_results:
                        df_data = []
                        for ticker, data in impact_results.items():
                            df_data.append({
                                'Ticker': ticker,
                                'Impact Score': data['score'],
                                'Causal Path': ' -> '.join(data['path'])
                            })
                        
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, column_config={
                            "Impact Score": st.column_config.NumberColumn(format="%.2f"),
                            "Causal Path": st.column_config.TextColumn("Causal Path", max_chars=100)
                        })
                        
                        if 'sim_impact_nodes' not in st.session_state:
                            st.session_state['sim_impact_nodes'] = []
                        st.session_state['sim_impact_nodes'] = impact_results.keys()
                    
                    else:
                        st.info("No downstream impacts found.")
                        st.session_state['sim_impact_nodes'] = []
                
                with sim_col2:
                    st.subheader("Focused Impact Graph")
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

    # --- TAB 4: CAUSAL PATHFINDING ---
    with tab_path:
        st.header("↔️ Causal Pathfinding")
        
        with st.spinner("Loading full graph for pathfinding..."):
            financial_graph = get_full_graph()
        
        if financial_graph is None:
             st.warning("Knowledge graph is empty. Please run `worker.py` (locally) to populate the *cloud* database.")
        else:
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
                                2: {"label": "High", "color": "#ef5350"}  # Red
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

                    except nx.NodeNotFound:
                        st.error(f"One of the nodes ({start_node} or {end_node}) was not found in the graph.")
                else:
                    st.warning("Please select two different companies.")

    # --- TAB 5: VIEW REPORTS ---
    with tab_reports:
        st.header("📂 Generated Event Reports")
        st.write("Reports generated by the background worker when significant events are detected.")
        st.write("This feature is currently disabled on Streamlit Cloud, as it cannot access the local 'reports/' folder.")
        
        st.info("No reports have been generated yet.")
        
        # ### BUG FIX: The rest of this is commented out
        # ### because glob.glob will not work on Streamlit Cloud.
        # report_files = sorted(glob.glob("reports/*.txt"), reverse=True)
        # if not report_files:
        #    st.info("No reports have been generated yet.")
        # else:
        #     selected_report = st.selectbox("Select a report to view:", report_files)
        #     if selected_report:
        #         with open(selected_report, 'r', encoding='utf-8') as f:
        #             st.code(f.read(), language='text')

if __name__ == "__main__":
    main()