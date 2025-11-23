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
import google.generativeai as genai  # <--- NEW IMPORT FOR AI

# --- Local Imports ---
from database_manager import DatabaseManager

# --- Setup structured logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ==============================================================================
# --- GEMINI AI SETUP & HELPER FUNCTIONS ---
# ==============================================================================
# Check if the key exists in secrets and set it as an environment variable
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
elif "GOOGLE_API_KEY" in st.secrets:
    # Fallback in case you named it GOOGLE_API_KEY in the cloud dashboard
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.warning("⚠️ API Key not found in Streamlit Cloud Secrets.")

def setup_genai():
    """Configures Google Gemini API from Streamlit Secrets."""
    try:
        # Check if key exists in secrets
        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
        elif "general" in st.secrets and "GEMINI_API_KEY" in st.secrets["general"]:
             api_key = st.secrets["general"]["GEMINI_API_KEY"]
        else:
            return False

        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        logger.error(f"Failed to configure Gemini AI: {e}")
        return False

def generate_ai_analysis(prompt: str) -> str:
    """Sends a prompt to Google Gemini and returns the response."""
    try:
        # UPDATED: Use gemini-1.5-flash for best stability and speed
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating AI explanation: {str(e)}"


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
# --- FUNCTION: Inject Live Risk Scores (FIX: FORCE VISUAL SPREAD) ---
# ==============================================================================
def inject_live_risk_data(graph: nx.DiGraph, csv_path: str = "live_risk_scores.csv") -> nx.DiGraph:
    """
    Reads the live risk CSV and updates nodes.
    
    CRITICAL FIX: Uses method='first' for ranking.
    This prevents 'clumping' where everyone is Green or Red.
    It forces a visual distribution even if scores are very close.
    """
    if not os.path.exists(csv_path):
        logger.warning(f"Live risk file {csv_path} not found. Using static graph data.")
        return graph

    try:
        df = pd.read_csv(csv_path)
        
        if df.empty:
            return graph

        # --- DEBUG: Show distribution stats in sidebar ---
        # This will verify if your data is actually loading
        with st.sidebar.expander("📊 Risk Distribution Debug"):
            st.write(df['Risk_Score'].describe())

        # --- LOGIC CHANGE: method='first' ---
        # This breaks ties by forcing a unique rank for every row.
        # It guarantees we fill the 0-100 percentile range smoothly.
        df['Risk_Rank'] = df['Risk_Score'].rank(method='first', pct=True)

        # Create dictionaries for lookup
        score_map = pd.Series(df.Risk_Score.values, index=df.Ticker).to_dict()
        rank_map = pd.Series(df.Risk_Rank.values, index=df.Ticker).to_dict()
        
        updated_count = 0
        
        for node in graph.nodes():
            if node in score_map:
                raw_score = float(score_map[node])
                percentile = float(rank_map[node])
                
                graph.nodes[node]['raw_risk_score'] = raw_score
                
                # Assign Color based on Forced Percentile
                # Top 5% -> RED
                if percentile >= 0.95:
                    risk_level = 2 
                # Next 25% (70th to 95th) -> ORANGE
                elif percentile >= 0.70:
                    risk_level = 1 
                # Bottom 70% -> GREEN
                else:
                    risk_level = 0 
                    
                graph.nodes[node]['predicted_risk'] = risk_level
                updated_count += 1
                
        logger.info(f"Updated {updated_count} nodes using Forced Ranking.")
        
    except Exception as e:
        logger.error(f"Error injecting live risk scores: {e}")
        
    return graph

# ==============================================================================
# --- NEW FUNCTION: Systemic Vulnerability Analysis (The "Butterfly Effect") ---
# ==============================================================================
def analyze_systemic_vulnerability(graph: nx.DiGraph) -> pd.DataFrame:
    """
    Iterates through EVERY node in the graph, simulates a -1.0 (catastrophic) event,
    and calculates the total summed impact on the rest of the network.
    Returns a DataFrame ranked by 'Systemic Damage'.
    """
    vulnerability_scores = []
    all_nodes = list(graph.nodes())
    total_nodes = len(all_nodes)
    
    # Create a progress bar in the UI since this might take a few seconds
    progress_bar = st.progress(0, text="Simulating system-wide shocks...")

    for i, node in enumerate(all_nodes):
        # 1. Run a simulation for this specific node failing
        # We use -1.0 to represent a "Total Collapse" of this node
        impacts = calculate_impact_scores(graph, node, -1.0)
        
        # 2. Sum the total damage to the ecosystem
        # (Sum of all impact scores generated by this event)
        total_system_loss = sum([data['score'] for data in impacts.values()])
        
        vulnerability_scores.append({
            "Ticker": node, 
            "Systemic_Damage": total_system_loss,
            " impacted_count": len(impacts)
        })
        
        # Update progress
        progress_bar.progress((i + 1) / total_nodes)

    progress_bar.empty() # Clear bar when done
    
    df = pd.DataFrame(vulnerability_scores)
    # Sort by Damage (ascending, because damage is a negative number)
    # The most negative number = The most damage
    return df.sort_values("Systemic_Damage", ascending=True)

# ==============================================================================
# --- FUNCTION: calculate_impact_scores (FIX: BIDIRECTIONAL PROPAGATION) ---
# ==============================================================================
def calculate_impact_scores(graph: nx.DiGraph, start_node: str, event_magnitude: float = 1.0) -> Dict[str, dict]:
    """
    Calculates impact scores by propagating an event through the graph.
    
    UPDATED: Now propagates BIDIRECTIONALLY (Upstream & Downstream).
    - If a Supplier fails, the Customer hurts (Forward).
    - If a Customer fails, the Supplier hurts (Backward).
    """
    if start_node not in graph:
        return {}

    risk_multipliers = {
        0: 1.0,  # Low Risk
        1: 1.2,  # Medium Risk
        2: 1.5,  # High Risk
    }

    impact_data = {node: {'score': 0.0, 'path': []} for node in graph.nodes}
    impact_data[start_node] = {'score': event_magnitude, 'path': [start_node]}
    
    queue = [start_node]
    visited = set([start_node]) 

    while queue:
        current_node = queue.pop(0)
        
        # Stop if signal is too weak
        if abs(impact_data[current_node]['score']) < 0.01:
            continue

        # --- CRITICAL FIX: Get BOTH Incoming and Outgoing neighbors ---
        # We convert to list() to avoid runtime errors if graph changes, though it shouldn't here.
        # 1. Outgoing (Successors)
        successors = list(graph.successors(current_node))
        # 2. Incoming (Predecessors)
        predecessors = list(graph.predecessors(current_node))
        
        # Combine them to propagate shock in all directions
        all_neighbors = set(successors + predecessors)

        for neighbor in all_neighbors:
            
            # Determine edge direction for weight lookup
            # (We need to know which way the arrow points to get the edge data)
            if neighbor in successors:
                edge_data = graph.get_edge_data(current_node, neighbor, default={})
                direction = "forward"
            else:
                edge_data = graph.get_edge_data(neighbor, current_node, default={})
                direction = "backward" # Shock traveling up-stream
            
            # Default weight logic (Fixed from previous step)
            weight = edge_data.get('weight', 0.5)
            if weight == 0: weight = 0.5
            
            # Slightly dampen backward shocks (Revenue loss is usually less fatal than Supply cut)
            if direction == "backward":
                weight *= 0.8 
                
            relationship_type = edge_data.get('type', 'dependency').lower()

            # Get neighbor risk multiplier
            neighbor_data = graph.nodes[neighbor]
            neighbor_risk_level = neighbor_data.get('predicted_risk', 0)
            risk_multiplier = risk_multipliers.get(neighbor_risk_level, 1.0)

            base_impact = impact_data[current_node]['score'] * weight
            
            if event_magnitude < 0:
                propagated_impact = base_impact * risk_multiplier
            else:
                propagated_impact = base_impact

            if relationship_type == 'competitor':
                propagated_impact *= -1

            # Update if this impact is stronger than any previous one
            if abs(propagated_impact) > abs(impact_data.get(neighbor, {}).get('score', 0.0)):
                impact_data[neighbor]['score'] = propagated_impact
                impact_data[neighbor]['path'] = impact_data[current_node]['path'] + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    final_impacts = {
        n: d for n, d in impact_data.items() 
        if abs(d['score']) > 0.01 and n != start_node
    }
    
    return dict(sorted(final_impacts.items(), key=lambda item: abs(item[1]['score']), reverse=True))


# ==============================================================================
# --- CACHED FUNCTIONS ---
# ==============================================================================

# --- NEW: Load Company Names for Dropdown ---
@st.cache_data
def load_company_names():
    """Loads the dictionary mapping Tickers -> Company Names."""
    try:
        # Use absolute path to ensure Streamlit Cloud finds it
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        FILE_PATH = os.path.join(SCRIPT_DIR, 'sp500_companies.json')
        
        with open(FILE_PATH, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback if file isn't pushed to git yet
        return {} 

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

@st.cache_data(ttl=600) # Refresh every 10 mins to pick up new CSV data
def get_full_graph():
    """
    Cached function to load the pre-computed graph from a file 
    AND inject the latest live risk scores.
    """
    
    # --- FIX: Use an absolute path ---
    try:
        # Get the absolute path of the directory this script is in
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        # Join it with the filenames to get full, robust paths
        GRAPH_FILE_PATH = os.path.join(SCRIPT_DIR, "financial_graph.gml")
        CSV_FILE_PATH = os.path.join(SCRIPT_DIR, "live_risk_scores.csv")
        
    except Exception as e:
        logger.warning(f"Could not determine script path, falling back to relative path. Error: {e}")
        # Fallback just in case __file__ is not available
        SCRIPT_DIR = "."
        GRAPH_FILE_PATH = "financial_graph.gml" 
        CSV_FILE_PATH = "live_risk_scores.csv"
    # --- END FIX ---
    
    logger.info(f"Attempting to load pre-computed graph from {GRAPH_FILE_PATH}...")
    
    try:
        # 1. Load the Graph Structure (GML)
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
        
    logger.info(f"Graph structure loaded with {graph.number_of_nodes()} nodes.")

    # 2. Inject the Live Risk Data (CSV)
    # This updates the graph object with the latest numbers before returning it
    graph = inject_live_risk_data(graph, CSV_FILE_PATH)

    return graph

@st.cache_data(ttl=600) 
def get_top_ripple_effects(all_events: list, threshold: float) -> pd.DataFrame | None:
    """
    Runs a simulation for ALL recent negative events and finds the
    companies with the worst potential ripple effects.
    """
    
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

    # --- AI SETUP ---
    gemini_active = setup_genai()
    if gemini_active:
        st.sidebar.success("🤖 Gemini AI Active")
    else:
        st.sidebar.warning("⚠️ Google Gemini API Key not found. AI Explanations disabled.")

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
            st.info(f"Displaying the {len(recent_events)} most recent events.")

            for event in recent_events:
                event_type = "Positive📈" if event['score'] > 0 else "Negative📉"
                with st.expander(f"**{event.get('timestamp', 'No Date')} - {event_type} for {event['ticker']}**: {event['headline']}"):
                    st.markdown(f"**Sentiment Score:** `{event.get('score', 0.0):.2f}`")
                    st.markdown(f"[Read Full Article]({event['link']})", unsafe_allow_html=True)
                    st.info(f"To see the potential impact of this event, go to the 'Simulate Scenarios' tab and run a simulation for {event['ticker']}.")
            
    # --- TAB 2: EXPLORE GRAPH (On-Demand Version) ---
    with tab_explore:
        st.header("🗺️ Interactive Knowledge Graph Explorer")
        st.write("Select a company to load its **Top 25** strongest relationships.") 

        with st.spinner("Loading full graph for explorer..."):
            # We load the full graph here primarily to get the LIVE RISK SCORES from the CSV
            financial_graph = get_full_graph()
        
        if financial_graph is None:
                st.warning("Knowledge graph is empty. Please run `worker.py` (locally) to populate the *cloud* database.")
        else:
            all_nodes = sorted(list(financial_graph.nodes()))
            
            # --- UPDATED DROPDOWN WITH COMPANY NAMES ---
            company_map = load_company_names()
            
            # Helper to format the display (e.g. "Apple Inc. (AAPL)")
            def format_ticker(ticker):
                name = company_map.get(ticker)
                if name:
                    return f"{name} ({ticker})"
                return ticker

            selected_company = st.selectbox(
                "Select a company to explore:", 
                all_nodes, 
                index=all_nodes.index("AAPL") if "AAPL" in all_nodes else 0,
                format_func=format_ticker, # <--- Displays Readable Names
                key="explore_select"
            )
            # -------------------------------------------
            
            show_risk = st.toggle("Show GNN Risk Coloring", value=True)

            # Layout for Graph + AI
            col_graph, col_ai = st.columns([3, 1])

            with col_graph:
                if st.button("🗺️ Explore Neighborhood"):
                    if selected_company:
                        with st.spinner(f"Loading neighborhood for {selected_company}..."):
                            
                            # Get structure from Neo4j
                            neighborhood_graph = db_manager.get_neighborhood_graph(selected_company)

                            if neighborhood_graph.number_of_nodes() > 0:
                                
                                # --- FIX: DATA SYNCHRONIZATION ---
                                for node in neighborhood_graph.nodes():
                                    if financial_graph.has_node(node):
                                        # Copy risk attributes from the global graph to the local view
                                        neighborhood_graph.nodes[node]['predicted_risk'] = financial_graph.nodes[node].get('predicted_risk', 0)
                                        neighborhood_graph.nodes[node]['raw_risk_score'] = financial_graph.nodes[node].get('raw_risk_score', 0.0)
                                # ---------------------------------

                                # Manually clean the graph edges of reserved keywords
                                graph_for_pyvis = neighborhood_graph.copy() 
                                for u, v, data in graph_for_pyvis.edges(data=True):
                                    data.pop('source', None)
                                    data.pop('target', None)

                                net = Network(height="750px", width="100%", notebook=True, cdn_resources='in_line', directed=True, bgcolor="#222222", font_color="white")
                                net.from_nx(graph_for_pyvis)
                                
                                risk_map = {
                                    0: {"label": "Low", "color": "#66bb6a"},  # Green
                                    1: {"label": "Medium", "color": "#ffa726"}, # Orange
                                    2: {"label": "High", "color": "#ef5350"}  # Red
                                }

                                for node in net.nodes:
                                    node_id = node["id"]
                                    node_data = neighborhood_graph.nodes[node_id]
                                    node['label'] = node_id
                                    
                                    # Retrieve risk data (now synced)
                                    predicted_risk = node_data.get('predicted_risk', 0)
                                    raw_score = node_data.get('raw_risk_score', 0.0)
                                    
                                    risk_info = risk_map.get(predicted_risk, risk_map[0])
                                    
                                    title_prefix = ""
                                    if show_risk and risk_info:
                                        node['color'] = risk_info['color']
                                        # UPDATED TOOLTIP: Shows raw score
                                        title_prefix = f"⚠️ RISK SCORE: {raw_score:.4f} ({risk_info['label'].upper()})\n" \
                                                       f"----------------------------------\n"
                                    
                                    if node_id == selected_company:
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
                                
                                RELATIONSHIP_THRESHOLD = 75 
                                
                                if neighborhood_graph.number_of_edges() > RELATIONSHIP_THRESHOLD:
                                    st.warning(f"Graph is large ({neighborhood_graph.number_of_edges()} relationships). Displaying with a simplified, static layout to prevent crashing.")
                                    options_str = '{"nodes": {"font": {"size": 18}}, "edges": {"smooth": {"type": "dynamic"}}, "physics": {"enabled": false}}'
                                
                                else:
                                    st.info(f"Displaying {neighborhood_graph.number_of_nodes()} companies and {neighborhood_graph.number_of_edges()} relationships.")
                                    options_str = '{"nodes": {"font": {"size": 18}}, "edges": {"smooth": {"type": "dynamic"}}, "physics": {"barnesHut": {"gravitationalConstant": -20000, "springLength": 350}, "stabilization": {"iterations": 1000}}}'
                                
                                net.set_options(options_str)
                                
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
            
            # --- AI ANALYSIS COLUMN ---
            with col_ai:
                st.subheader("🤖 AI Insight")
                if not gemini_active:
                    st.write("Gemini API not active.")
                elif selected_company:
                    st.write(f"Analyze risks for **{selected_company}**.")
                    prompt_type = st.selectbox("Query Type:", ["Economic Logic", "Contagion Risk"], key="ai_select")
                    
                    if st.button("🧠 Generate"):
                        with st.spinner("Consulting Gemini..."):
                            # Construct context
                            my_risk = financial_graph.nodes[selected_company].get('raw_risk_score', 0.0)
                            neighbors = list(financial_graph.neighbors(selected_company))[:5]
                            neighbor_str = ", ".join(neighbors)
                            
                            prompt = f"Financial Analysis for {selected_company}. Risk Score: {my_risk}. Upstream Entities: {neighbor_str}. Explain the {prompt_type} of this graph structure."
                            
                            explanation = generate_ai_analysis(prompt)
                            st.success("Generated Insight:")
                            st.write(explanation)

    # ==============================================================================
    # --- TAB 3: SIMULATE SCENARIOS (UPDATED) ---
    # ==============================================================================
    with tab_simulate:
        st.header("🔬 Impact & Contagion Analysis")

        # --- Load Graph (Needed for all features here) ---
        financial_graph = get_full_graph()
        if financial_graph is None:
            st.warning("Knowledge graph is empty. Please ensure data is loaded.")
            st.stop()

        # --- Load Company Names for Dropdowns ---
        company_map = load_company_names()
        def format_ticker(ticker):
            name = company_map.get(ticker)
            if name:
                return f"{name} ({ticker})"
            return ticker

        # --- SECTION 1: SYSTEMIC VULNERABILITY (NEW!) ---
        st.subheader("🌪️ Systemic Vulnerability (The 'Butterfly Effect')")
        st.write("Identify 'Super Spreader' nodes. This simulation crashes every single company one by one to see which failure causes the most damage to the entire network.")
        
        if st.button("🚀 Analyze Systemic Vulnerability"):
            with st.spinner("Running stress tests on all nodes..."):
                vuln_df = analyze_systemic_vulnerability(financial_graph)
                
                # Merge with company names for better display
                vuln_df['Company Name'] = vuln_df['Ticker'].map(company_map).fillna(vuln_df['Ticker'])
                
                # Format for display
                st.success("Analysis Complete. These companies pose the highest systemic risk.")
                
                # Top 10 Chart
                top_10 = vuln_df.head(10).copy()
                # Make damage positive for easier bar chart reading (Magnitude of Damage)
                top_10['Damage Magnitude'] = top_10['Systemic_Damage'].abs()
                
                st.bar_chart(top_10, x="Ticker", y="Damage Magnitude", color="#FF4B4B")
                
                # Full Data Table
                st.dataframe(
                    vuln_df[['Ticker', 'Company Name', 'Systemic_Damage', ' impacted_count']], 
                    use_container_width=True,
                    column_config={
                        "Systemic_Damage": st.column_config.NumberColumn("Total Network Loss", format="%.2f"),
                        " impacted_count": st.column_config.NumberColumn("Companies Affected")
                    }
                )

        st.divider()

        # --- SECTION 2: AUTOMATED EVENT ANALYSIS (FIXED) ---
        st.subheader("🔔 Real-World Event Ripple Effects")
        st.write("Analyze recent negative news events to find contagion risks.")
        
        sensitivity_threshold = st.slider(
            "Event Score Sensitivity (find events with a score *below* this value)",
            min_value=-1.0, max_value=0.0, value=-0.2, step=0.05
        )
        
        all_recent_events = db_manager.get_recent_events()
        
        if not all_recent_events:
            st.info("No recent events found in the database.")
        else:
            with st.spinner("Calculating ripple effects..."):
                # Pass the graph explicitly to avoid reloading
                top_impacts_df = get_top_ripple_effects(all_recent_events, sensitivity_threshold)
            
            if top_impacts_df is None or top_impacts_df.empty:
                st.info(f"No ripple effects found. (No events scored below {sensitivity_threshold:.2f} or no connections found).")
            else:
                st.warning("Found the following potential contagion risks:")
                # Format floats for display
                top_impacts_df['Worst Impact Score'] = top_impacts_df['Worst Impact Score'].apply(lambda x: f"{x:.4f}")
                
                st.dataframe(
                    top_impacts_df, 
                    use_container_width=True, 
                    column_config={
                        "Source Event Headline": st.column_config.TextColumn("Source Event Headline", max_chars=100),
                        "Causal Path": st.column_config.TextColumn("Causal Path", max_chars=100)
                    }
                )

        st.divider() 

        # --- SECTION 3: MANUAL SIMULATION (UPDATED DROPDOWN) ---
        st.subheader("🧪 Manual 'What-If' Simulation")
        
        all_nodes = sorted(list(financial_graph.nodes()))
        col1, col2 = st.columns(2)
        
        # UPDATED: Uses format_ticker to show names
        selected_company_sim = col1.selectbox(
            "Select a company to trigger an event:", 
            all_nodes, 
            format_func=format_ticker,
            key="sim_select_company"
        )
        
        hypothetical_score = col2.slider(
            "Select a hypothetical event score:",
            min_value=-1.0, max_value=1.0, value=-0.5, step=0.05,
            help="Negative values = Bad News, Positive values = Good News"
        )

        if st.button("💥 Simulate Event", key="btn_manual_sim"):
            st.subheader(f"Simulating impact of {hypothetical_score} event on {format_ticker(selected_company_sim)}")
            
            impact_results = calculate_impact_scores(financial_graph, selected_company_sim, hypothetical_score)
            
            sim_col1, sim_col2 = st.columns([1, 2])
            with sim_col1:
                st.subheader("Calculated Impacts")
                if impact_results:
                    df_data = []
                    for ticker, data in impact_results.items():
                        df_data.append({
                            'Ticker': ticker,
                            'Name': company_map.get(ticker, ticker), # Lookup name
                            'Impact Score': data['score'],
                            'Causal Path': ' -> '.join(data['path'])
                        })
                    
                    df = pd.DataFrame(df_data)
                    st.dataframe(
                        df, 
                        column_config={
                            "Impact Score": st.column_config.NumberColumn(format="%.4f"),
                            "Causal Path": st.column_config.TextColumn("Causal Path", max_chars=100)
                        },
                        hide_index=True
                    )
                    
                    # Store for graph viz
                    st.session_state['sim_impact_nodes'] = list(impact_results.keys())
                
                else:
                    st.info("No downstream impacts found (Node might be isolated or impact < threshold).")
                    st.session_state['sim_impact_nodes'] = []

            with sim_col2:
                st.subheader("Visual Impact Graph")
                
                # --- FIX: LIMIT VISUALIZATION TO TOP 25 NODES ---
                # The simulation might hit 500 nodes, creating a 'hairball'.
                # We only want to visualize the Source + Top 25 Victims.
                
                if st.session_state.get('sim_impact_nodes'):
                    # 1. Sort impacts by magnitude
                    sorted_nodes = sorted(
                        impact_results.items(), 
                        key=lambda item: abs(item[1]['score']), 
                        reverse=True
                    )
                    
                    # 2. Slice the top 25
                    top_victims = [n for n, data in sorted_nodes[:25]]
                    
                    # 3. Define the subgraph (Source + Top Victims)
                    nodes_to_include = {selected_company_sim, *top_victims}
                    subgraph = financial_graph.subgraph(nodes_to_include)
                    
                    # Create a clean copy for pyvis
                    graph_for_pyvis = subgraph.copy()
                    for u, v, data in graph_for_pyvis.edges(data=True):
                        data.pop('source', None)
                        data.pop('target', None)

                    # --- SETUP PYVIS ---
                    net = Network(
                        height="500px", 
                        width="100%", 
                        notebook=True, 
                        directed=True, 
                        bgcolor="#222222", 
                        font_color="white"
                    )
                    net.from_nx(graph_for_pyvis) 
                    
                    # --- COLORING & PHYSICS ---
                    for node in net.nodes:
                        nid = node['id']
                        node['label'] = nid
                        
                        # Source Node = RED and BIG
                        if nid == selected_company_sim:
                            node['color'] = '#FF4B4B' 
                            node['size'] = 40
                            node['shape'] = 'diamond'
                        
                        # Impacted Nodes = ORANGE
                        elif nid in st.session_state['sim_impact_nodes']:
                            node['color'] = '#FFA726' 
                            node['size'] = 20
                            
                            # Add tooltip with exact impact score
                            score = impact_results.get(nid, {}).get('score', 0)
                            node['title'] = f"Impact: {score:.4f}"

                    # Use BarnesHut physics to push nodes apart so they don't blob
                    net.set_options("""
                    {
                      "physics": {
                        "barnesHut": {
                          "gravitationalConstant": -30000,
                          "centralGravity": 0.3,
                          "springLength": 100,
                          "springConstant": 0.05,
                          "damping": 0.09,
                          "avoidOverlap": 1
                        },
                        "minVelocity": 0.75
                      }
                    }
                    """)
                    
                    html_file = f"temp_graph_{uuid.uuid4().hex}.html"
                    try:
                        net.save_graph(html_file)
                        with open(html_file, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        st.components.v1.html(html_content, height=520, scrolling=True)
                    except Exception as e:
                        st.error(f"Graph Error: {e}")
                    finally:
                        if os.path.exists(html_file):
                            os.remove(html_file)
                else:
                     st.write("Run a simulation to see the graph.")

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
                            st.info(f"Total Path Risk Level: **{max_risk_score}** (Sum of risk levels)")

                            # --- AI INTEGRATION FOR PATHS ---
                            if gemini_active:
                                if st.button("🧠 Explain this Path Logic"):
                                    with st.spinner("Analyzing path logic..."):
                                        path_str = ' -> '.join(best_path)
                                        prompt = f"Explain the economic logic behind this contagion path: {path_str}."
                                        st.write(generate_ai_analysis(prompt))
                            # --------------------------------

                            path_graph = financial_graph.subgraph(best_path)

                            # Create a clean copy for pyvis
                            graph_for_pyvis = path_graph.copy()
                            for u, v, data in graph_for_pyvis.edges(data=True):
                                data.pop('source', None)
                                data.pop('target', None)
                            
                            net = Network(height="400px", width="100%", notebook=True, directed=True, bgcolor="#222222", font_color="white")
                            net.from_nx(graph_for_pyvis) 
                            
                            risk_map = {
                                0: {"label": "Low", "color": "#66bb6a"}, 
                                1: {"label": "Medium", "color": "#ffa726"},
                                2: {"label": "High", "color": "#ef5350"} 
                            }

                            for node_id in best_path:
                                node = net.get_node(node_id)
                                node_data = financial_graph.nodes[node_id]
                                
                                node_risk = node_data.get('predicted_risk', 0)
                                raw_score = node_data.get('raw_risk_score', 0.0)
                                
                                risk_info = risk_map.get(node_risk, risk_map[0])
                                
                                node['label'], node['size'] = node_id, 25
                                
                                if risk_info:
                                    node['color'] = risk_info['color']
                                    # UPDATED TOOLTIP: Shows raw score
                                    node['title'] = f"⚠️ RISK: {raw_score:.4f} ({risk_info['label']})"
                                
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
        
if __name__ == "__main__":
    main()