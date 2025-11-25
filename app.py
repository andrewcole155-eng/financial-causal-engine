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
import google.generativeai as genai 

# --- Local Imports ---
from database_manager import DatabaseManager

if "graph_html" not in st.session_state:
    st.session_state.graph_html = None
if "selected_ticker_for_graph" not in st.session_state:
    st.session_state.selected_ticker_for_graph = None

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
    """
    Sends a prompt to Google Gemini. 
    INCLUDES FALLBACK LOGIC: If the graph data is sparse/empty, 
    it instructs the AI to use its internal knowledge base to fill the gaps.
    """
    try:
        # Check if the prompt indicates missing data (heuristic check)
        # Your graph usually sends "Upstream Entities: "
        is_sparse_data = "Upstream Entities: ," in prompt or "Upstream Entities: ." in prompt or "Upstream Entities:  " in prompt
        
        final_prompt = prompt
        
        if is_sparse_data:
            # Inject "Knowledge Retrieval" instruction
            final_prompt += """
            
            CRITICAL INSTRUCTION: 
            The provided graph data appears incomplete or empty for this specific entity. 
            IGNORE the empty upstream list. 
            INSTEAD, use your internal training knowledge to identify the top 3-5 REAL-WORLD upstream dependencies 
            (e.g., Strategic Partners, Suppliers, Joint Ventures, or Major Institutional Investors) for this company.
            
            Analyze the risk based on these KNOWN real-world relationships, not the empty graph.
            Explicitly state that you are filling in missing graph data with external knowledge.
            """

        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(final_prompt)
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
# --- FUNCTION: Inject Live Risk Scores ---
# ==============================================================================
def inject_live_risk_data(graph: nx.DiGraph, csv_path: str = "live_risk_scores.csv") -> nx.DiGraph:
    if not os.path.exists(csv_path):
        logger.warning(f"Live risk file {csv_path} not found. Using static graph data.")
        return graph

    try:
        df = pd.read_csv(csv_path)
        
        if df.empty:
            return graph

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
# --- FUNCTION: Systemic Vulnerability Analysis (The "Butterfly Effect") ---
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

    progress_bar.empty() 
    
    df = pd.DataFrame(vulnerability_scores)
    # Sort by Damage (ascending, because damage is a negative number)
    # The most negative number = The most damage
    return df.sort_values("Systemic_Damage", ascending=True)

# ==============================================================================
# --- FUNCTION: calculate_impact_scores  ---
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

        # We convert to list() to avoid runtime errors if graph changes, though it shouldn't here.
        # 1. Outgoing (Successors)
        successors = list(graph.successors(current_node))
        # 2. Incoming (Predecessors)
        predecessors = list(graph.predecessors(current_node))
        
        # Combine them to propagate shock in all directions
        all_neighbors = set(successors + predecessors)

        for neighbor in all_neighbors:
            
            # Determine edge direction for weight lookup
            if neighbor in successors:
                edge_data = graph.get_edge_data(current_node, neighbor, default={})
                direction = "forward"
            else:
                edge_data = graph.get_edge_data(neighbor, current_node, default={})
                direction = "backward" 
            
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
# --- Load Company Names for Dropdown ---
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

    st.sidebar.divider() # Optional: Adds a visual line separator
    if st.sidebar.button("🔥 Nuke Cache & Reload Data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    # --- UPDATED TABS DEFINITION ---
    tab_events, tab_explore, tab_simulate, tab_path, tab_help = st.tabs([
        "🔔 Recent Events", "🗺️ Explore Graph", "🔬 Simulate Scenarios", "↔️ Causal Pathfinding", "📘 Help & Guide"
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

    # ==============================================================================
    # --- TAB 2: EXPLORE GRAPH ---
    # ==============================================================================

    with tab_explore:
        st.header("🗺️ Interactive Knowledge Graph Explorer")
        st.write("Select a company to load its strongest relationships.") 

        with st.spinner("Loading full graph for explorer..."):
            # We load the full graph here primarily to get the LIVE RISK SCORES from the CSV
            financial_graph = get_full_graph()
        
        if financial_graph is None:
                st.warning("Knowledge graph is empty. Please run `worker.py` (locally) to populate the *cloud* database.")
        else:
            all_nodes = sorted(list(financial_graph.nodes()))
            company_map = load_company_names()
            
            # Helper to format the display
            def format_ticker(ticker):
                name = company_map.get(ticker)
                if name:
                    return f"{name} ({ticker})"
                return ticker

            selected_company = st.selectbox(
                "Select a company to explore:", 
                all_nodes, 
                index=all_nodes.index("AAPL") if "AAPL" in all_nodes else 0,
                format_func=format_ticker,
                key="explore_select"
            )
            
            show_risk = st.toggle("Show GNN Risk Coloring", value=True)

            # Layout for Graph + AI
            col_graph, col_ai = st.columns([3, 1])

            # --- COLUMN 1: THE GRAPH ---
            with col_graph:
                # --- BUTTON CLICK HANDLER (Generates & Saves) ---
                if st.button("🗺️ Explore Neighborhood"):
                    if selected_company:
                        with st.spinner(f"Loading neighborhood for {selected_company}..."):
                            
                            # Get structure from Neo4j
                            neighborhood_graph = db_manager.get_neighborhood_graph(selected_company)

                            if neighborhood_graph.number_of_nodes() > 0:
                                
                                # Transfer risk scores from Global Graph to Local Neighborhood Graph
                                for node in neighborhood_graph.nodes():
                                    if financial_graph.has_node(node):
                                        neighborhood_graph.nodes[node]['predicted_risk'] = financial_graph.nodes[node].get('predicted_risk', 0)
                                        neighborhood_graph.nodes[node]['raw_risk_score'] = financial_graph.nodes[node].get('raw_risk_score', 0.0)

                                # Manually clean the graph edges of reserved keywords for Pyvis
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
                                    
                                    # Retrieve risk data
                                    predicted_risk = node_data.get('predicted_risk', 0)
                                    raw_score = node_data.get('raw_risk_score', 0.0)
                                    
                                    risk_info = risk_map.get(predicted_risk, risk_map[0])
                                    
                                    title_prefix = ""
                                    if show_risk and risk_info:
                                        node['color'] = risk_info['color']
                                        # TOOLTIP: Shows raw score
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
                                    st.warning(f"Graph is large ({neighborhood_graph.number_of_edges()} relationships). Displaying with a simplified, static layout.")
                                    options_str = '{"nodes": {"font": {"size": 18}}, "edges": {"smooth": {"type": "dynamic"}}, "physics": {"enabled": false}}'
                                
                                else:
                                    st.info(f"Displaying {neighborhood_graph.number_of_nodes()} companies and {neighborhood_graph.number_of_edges()} relationships.")
                                    options_str = '{"nodes": {"font": {"size": 18}}, "edges": {"smooth": {"type": "dynamic"}}, "physics": {"barnesHut": {"gravitationalConstant": -20000, "springLength": 350}, "stabilization": {"iterations": 1000}}}'
                                
                                net.set_options(options_str)
                                
                                # --- SAVE TO SESSION STATE ---
                                html_file = f"temp_graph_{uuid.uuid4().hex}.html"
                                try:
                                    net.save_graph(html_file)
                                    with open(html_file, 'r', encoding='utf-8') as f:
                                        st.session_state.graph_html = f.read()
                                        st.session_state.selected_ticker_for_graph = selected_company
                                except Exception as e:
                                    st.error(f"Failed to render graph: {e}")
                                finally:
                                    if os.path.exists(html_file):
                                        os.remove(html_file)
                                    
                                # === CRITICAL FIX (Re-applied) ===
                                st.rerun()

                            else:
                                st.warning(f"No relationships found for {selected_company}.")
                    else:
                        st.warning("Please select a company.")

                # --- PERSISTENT DISPLAY ---
                if st.session_state.graph_html is not None:
                    if st.session_state.selected_ticker_for_graph != selected_company:
                        st.caption(f"⚠️ Currently displaying graph for: {st.session_state.selected_ticker_for_graph}. Click 'Explore Neighborhood' to update.")
                    
                    st.components.v1.html(st.session_state.graph_html, height=800, scrolling=True)
                else:
                    st.info("Click 'Explore Neighborhood' to generate the graph.")

            # --- COLUMN 2: AI ANALYSIS ---
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
    # --- TAB 3: SIMULATE SCENARIOS ---
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

        # --- SECTION 1: SYSTEMIC VULNERABILITY ---
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

        # --- SECTION 2: AUTOMATED EVENT ANALYSIS ---
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

        # --- SECTION 3: MANUAL SIMULATION ---
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
                            'Name': company_map.get(ticker, ticker), 
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
                
                # --- LIMIT VISUALIZATION TO TOP 25 NODES ---
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

    # --- TAB 4: CAUSAL PATHFINDING (With Dynamic Filtering & Smart Diagnostics) ---
    with tab_path:
        st.header("↔️ Causal Pathfinding")
        
        # 1. LOAD THE GRAPH FIRST
        with st.spinner("Loading full graph for pathfinding..."):
            financial_graph = get_full_graph()
        
        if financial_graph is None:
                st.warning("Knowledge graph is empty. Please run `worker.py` (locally) to populate the *cloud* database.")
        else:
            st.write(f"Finds the shortest path between two nodes in the pre-filtered graph (the {financial_graph.number_of_edges()} strongest relationships).")
            
            all_nodes = sorted(list(financial_graph.nodes()))
            company_map = load_company_names()
            
            # Enhanced Formatter
            def format_ticker(ticker):
                name = company_map.get(ticker)
                if name:
                    return f"{name} ({ticker})"
                return ticker

            path_col1, path_col2 = st.columns(2)
            
            # --- COLUMN 1: START NODE & DIAGNOSTICS ---
            with path_col1:
                start_node = st.selectbox(
                    "Start Company (The Trigger):", 
                    all_nodes, 
                    format_func=format_ticker, 
                    key="path_start",
                    index=all_nodes.index("AAPL") if "AAPL" in all_nodes else 0
                )

                # =========================================================
                # 📍 DIAGNOSTICS & SMART SWAP LOGIC
                # =========================================================
                is_sink_node = False # Flag to control flow

                if start_node and start_node in financial_graph:
                    out_degree = financial_graph.out_degree(start_node)
                    in_degree = financial_graph.in_degree(start_node)
                    
                    # Visual Stats
                    st.caption(f"📊 **Node Stats:** `{out_degree} Outgoing` (Causes) | `{in_degree} Incoming` (Effects)")
                    
                    # CASE 1: The Dead End (Sink Node)
                    if out_degree == 0:
                        is_sink_node = True
                        st.error(f"🚫 **Sink Node Detected:** {start_node} absorbs impacts but doesn't cause downstream ripple effects.")
                        
                        # If it has parents, offer to Find Causes instead
                        if in_degree > 0:
                            st.info(f"💡 However, {start_node} IS affected by {in_degree} other factors.")
                            
                            # --- THE MAGIC BUTTON ---
                            if st.button("🔄 Analyze Incoming Factors (Reverse Look)"):
                                st.divider()
                                st.subheader(f"🕵️‍♀️ What influences {start_node}?")
                                
                                # Find all predecessors (Parents)
                                parents = list(financial_graph.predecessors(start_node))
                                
                                # Sort parents by risk score
                                parent_risks = []
                                for p in parents:
                                    r_score = financial_graph.nodes[p].get('raw_risk_score', 0)
                                    p_name = company_map.get(p, p)
                                    parent_risks.append({'Ticker': p, 'Name': p_name, 'Risk Score': r_score})
                                
                                df_parents = pd.DataFrame(parent_risks).sort_values("Risk Score", ascending=False)
                                
                                st.dataframe(
                                    df_parents, 
                                    hide_index=True,
                                    use_container_width=True,
                                    column_config={
                                        "Risk Score": st.column_config.ProgressColumn(
                                            "Risk Impact", 
                                            format="%.4f",
                                            min_value=0, 
                                            max_value=max(df_parents['Risk Score'].max(), 1.0)
                                        )
                                    }
                                )
                                st.stop() # Stop execution here so we don't show the empty 'End Company' error below
                    
                    # CASE 2: The Isolated Island
                    elif out_degree == 0 and in_degree == 0:
                         is_sink_node = True
                         st.warning(f"⚠️ **Ghost Node:** {start_node} is completely disconnected. Check your data ingestion.")
                # =========================================================

            # --- DYNAMIC FILTERING LOGIC ---
            # We only calculate reachable nodes if it's NOT a sink node
            reachable_nodes = {}
            if start_node and not is_sink_node:
                try:
                    # distinct checking for reachability (fast BFS)
                    reachable_nodes = nx.single_source_shortest_path_length(financial_graph, start_node, cutoff=5)
                except Exception:
                    reachable_nodes = {}
            
            # Create a list of valid targets (exclude the start node itself)
            valid_targets = sorted([n for n in reachable_nodes.keys() if n != start_node])

            # --- COLUMN 2: END NODE ---
            with path_col2:
                if is_sink_node:
                    st.info("🚫 Cannot select a target because the Start Company has no outgoing connections.")
                    end_node = None
                
                elif not valid_targets:
                    st.warning("⚠️ No companies are reachable from this start node within 5 steps.")
                    end_node = None
                else:
                    st.caption(f"✅ Filtered to {len(valid_targets)} companies reachable within 5 steps.")
                    
                    end_node = st.selectbox(
                        "End Company (Filtered by Reachability):", 
                        valid_targets, 
                        format_func=format_ticker, 
                        key="path_end"
                    )

            # --- EXECUTE PATHFINDING ---
            if not is_sink_node and st.button("🗺️ Find Riskiest Path"):
                if start_node and end_node:
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

                            # AI Explanation
                            if gemini_active:
                                if st.button("🧠 Explain this Path Logic"):
                                    with st.spinner("Analyzing path logic..."):
                                        path_str = ' -> '.join(best_path)
                                        prompt = f"Explain the economic logic behind this contagion path: {path_str}."
                                        st.write(generate_ai_analysis(prompt))

                            # Subgraph Visualization
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
                    if not is_sink_node:
                         st.warning("Please select two different companies.")

    # ==============================================================================
    # --- TAB 5: INSTRUCTIONS & GUIDE ---
    # ==============================================================================
    with tab_help:
        st.header("📘 User Guide & Documentation")
        
        # Sidebar for quick navigation within the Help tab
        help_choice = st.radio(
            "Navigate Guide:",
            ["Introduction", "Recent Events", "Explore Graph", "Simulate Scenarios", "Causal Pathfinding"],
            horizontal=True
        )
        
        st.divider()

        if help_choice == "Introduction":
            st.markdown("""
            ### Welcome to the Causal Inference Engine
            This application allows you to move beyond simple correlations and understand the **causal structure** of the financial markets.
            
            **Key Concepts:**
            * **Nodes:** Companies, Indices, or Economic Indicators.
            * **Edges:** The causal direction (e.g., Oil Price → Airline Stocks).
            * **Sentiment:** Real-time news scoring used to shock the system.
            """)
        
        elif help_choice == "Recent Events":
            st.markdown("""
            ### 🔔 Recent Events Tab
            **Purpose:** A real-time feed of market "shocks" and news signals that act as inputs for the causal model.

            **How to Use:**
            * **Refresh Events:** Click the button to reload the latest news from the database.
            * **Inspect Source:** Click the link icon to read the original news source.

            **Interpreting the Data:**
            * **Sentiment Score (-1 to +1):**
                * 🔴 **-1.0 (Bearish):** Examples: Regulatory bans, Lawsuits, Missed Earnings.
                * 🟢 **+1.0 (Bullish):** Examples: Mergers, Record Profits, Rate Cuts.
                * ⚪ **0.0 (Neutral):** Informational news with no immediate price impact.
            """)
        
        elif help_choice == "Explore Graph":
            st.markdown("""
            ### 🗺️ Explore Graph Tab
            **Purpose:** Visual map of your financial universe, showing how assets, sectors, and economic factors are interconnected.

            **How to Use:**
            * **Select Company:** Choose an asset from the dropdown to load its "Neighborhood" (Direct parents and children).
            * **Navigation:** Scroll to zoom, click-and-drag to pan the canvas.
            * **AI Insight:** Use the sidebar to ask Gemini to explain *why* two nodes are connected.

            **Interpreting the Data:**
            * **Nodes (The Dots):**
                * **Size:** Represents **Centrality**. Bigger nodes are "super-spreaders" of volatility.
                * **Color:** 🟢 Green (Low Risk), 🟠 Orange (Medium Risk), 🔴 Red (High Risk).
            * **Edges (The Arrows):**
                * **Direction (A → B):** "A causes B." If the arrow points from *Inflation* → *Retail Stocks*, inflation is the driver.
            """)
        
        elif help_choice == "Simulate Scenarios":
            st.markdown("""
            ### 🔬 Simulate Scenarios Tab
            **Purpose:** A "Stress Test" lab where you can ask "What If?" questions to forecast future outcomes.

            **How to Use:**
            * **Systemic Vulnerability:** Click "Analyze Systemic Vulnerability" to find which single company failure would cause the most total damage to the market.
            * **Manual Simulation:** Select a specific company and a hypothetical shock score (e.g., -0.5 for bad news).
            
            **Interpreting the Data:**
            * **Impact Score:** How strongly a downstream asset reacts to the shock.
            * **Divergence:** High impact scores indicate a non-obvious trading opportunity (Alpha).
            """)

        elif help_choice == "Causal Pathfinding":
            st.markdown("""
            ### ↔️ Causal Pathfinding Tab
            **Purpose:** To find the hidden "domino effect" between two seemingly unrelated assets.
            
            

            **How to Use:**
            * **Select Source:** The trigger event (e.g., `10Y Treasury Yield`).
            * **Select Target:** The asset you trade (e.g., `Bitcoin`).
            * *Note:* The Target dropdown automatically filters to only show assets that are actually reachable.

            **Interpreting the Data:**
            * **Mediators:** The nodes *between* your source and target.
                * *Example:* Bond Yields → Tech Sector → Crypto.
                * *Insight:* If the Tech Sector is resilient, the Bond Yield shock might never reach Crypto.
            """)

if __name__ == "__main__":
    main()