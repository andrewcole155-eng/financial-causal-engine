# ==============================================================================
# --- IMPORTS for Streamlit App ---
# ==============================================================================
import json
import logging
import os
import time
import glob
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List
import pandas as pd
import networkx as nx
import streamlit as st
from pyvis.network import Network
import google.generativeai as genai 
import plotly.express as px
import plotly.graph_objects as go 
from polygon import RESTClient
import torch
from gnn_explainer_logic import setup_explainer, explain_prediction
from predict import get_inference_resources

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
    # Local fallback checking
    if "GOOGLE_API_KEY" not in os.environ:
        st.warning("⚠️ API Key not found in Streamlit Cloud Secrets.")

def setup_genai():
    """Configures Google Gemini API from Streamlit Secrets."""
    try:
        # Check if key exists in secrets or env
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            if "GEMINI_API_KEY" in st.secrets:
                api_key = st.secrets["GEMINI_API_KEY"]
            elif "general" in st.secrets and "GEMINI_API_KEY" in st.secrets["general"]:
                 api_key = st.secrets["general"]["GEMINI_API_KEY"]
        
        if not api_key:
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
# --- CRITICAL: INJECT SECRETS FOR PREDICT.PY ---
# ==============================================================================
# predict.py looks for env vars or config.json. On the cloud, config.json is missing.
# We must manually inject the secrets into os.environ so the pipeline can connect.
if "neo4j" in st.secrets:
    os.environ["NEO4J_URI"] = st.secrets["neo4j"]["uri"]
    os.environ["NEO4J_USER"] = st.secrets["neo4j"]["user"]
    os.environ["NEO4J_PASSWORD"] = st.secrets["neo4j"]["password"]

# ==============================================================================
# --- NEW: VISUALIZATION HELPER (AI EDGE STYLING) ---
# ==============================================================================
def apply_ai_visual_styles(net: Network, nx_graph: nx.Graph):
    """
    Iterates through the PyVis network and updates edges based on their source.
    - AI Edges = Dashed Grey Lines.
    - Supply Chain (10-K) = Thick Blue Lines.
    - Standard Verified = Solid Lines.
    """
    ai_edge_count = 0
    
    for edge in net.edges:
        source = edge['from']
        target = edge['to']
        
        # Get data from original graph (handle directionality safety)
        nx_data = nx_graph.get_edge_data(source, target)
        if not nx_data:
             nx_data = nx_graph.get_edge_data(target, source)

        if nx_data:
            status = nx_data.get('verification_status', 'VERIFIED')
            mechanism = nx_data.get('mechanism', 'Unknown mechanism')

            # Sanitize text
            if mechanism:
                mechanism = mechanism.replace("'", "").replace('"', "").replace("\n", " ")

            # --- STYLE 1: AI INFERENCE (Dashed Grey) ---
            if status == "AI_PROPOSED":
                ai_edge_count += 1
                edge['dashes'] = True 
                edge['color'] = {'color': '#808080', 'highlight': '#a0a0a0', 'hover': '#ffffff'}
                edge['width'] = 3
                edge['title'] = f"🤖 AI INFERRED: {mechanism}"

            # --- STYLE 2: SUPPLY CHAIN / 10-K (Solid Blue) ---
            elif status == "VERIFIED_FILING" or mechanism == "10-K Disclosure":
                edge['dashes'] = False
                # Bright Blue to stand out against the dark background
                edge['color'] = {'color': '#29b6f6', 'highlight': '#4fc3f7', 'hover': '#ffffff'}
                edge['width'] = 5 # Thicker than normal lines
                edge['title'] = f"📄 SEC 10-K FILING: {mechanism}"

            # --- STYLE 3: STANDARD VERIFIED (Solid Default) ---
            else:
                edge['dashes'] = False
                if mechanism:
                     edge['title'] = f"✅ VERIFIED: {mechanism}"

    if ai_edge_count > 0:
        st.toast(f"🤖 Visualizing {ai_edge_count} AI-Inferred Relationships", icon="ℹ️")

    return net

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

def get_cloud_config_dict():
    """Returns a config dictionary from secrets (for use inside cached functions)."""
    if "neo4j" in st.secrets:
        return {
            "neo4j": {
                "uri": st.secrets["neo4j"]["uri"],
                "user": st.secrets["neo4j"]["user"],
                "password": st.secrets["neo4j"]["password"]
            }
        }
    return load_config()

# ==============================================================================
# --- NEW: TRUTH LAYER VISUALIZATION ---
# ==============================================================================
@st.cache_data(ttl=3600)
def fetch_polygon_price_data(ticker, api_key):
    """
    Separate function to fetch data so Streamlit can cache the result.
    Includes RETRY LOGIC for Rate Limits (Fixes the INTC error).
    """
    client = RESTClient(api_key)
    end_date = datetime.now()
    # Ensure this matches your backfill window (90 Days)
    start_date = end_date - timedelta(days=90) 
    
    # --- RETRY LOGIC for Free Tier Limits (5 req/min) ---
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Fetch 30-minute aggregates
            # Note: We iterate immediately to trigger the API call inside the try block
            aggs = []
            for a in client.list_aggs(
                ticker=ticker,
                multiplier=30,
                timespan="minute",
                from_=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d"),
                limit=50000
            ):
                aggs.append(a)
            
            # Convert to DataFrame
            data = [
                {"timestamp": datetime.fromtimestamp(a.timestamp / 1000), "close": a.close} 
                for a in aggs
            ]
            df = pd.DataFrame(data)

            if not df.empty:
                # 1. Force to UTC datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                # 2. Remove timezone info (make it naive) so it compares easily
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)
            
            return df

        except Exception as e:
            # Check if it's a Rate Limit error (usually contains "429" or "Too Many Requests")
            error_msg = str(e).lower()
            if "429" in error_msg or "too many requests" in error_msg:
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ Rate limit hit for {ticker}. Retrying in 15s... (Attempt {attempt+1}/{max_retries})")
                    # Wait 15s to clear the 5/min bucket (60s / 4 = 15s buffer)
                    time.sleep(15) 
                    continue
            
            # If it's a different error, log and return empty
            logger.error(f"Polygon API Error for {ticker}: {e}")
            return pd.DataFrame()
    
    return pd.DataFrame()

def render_truth_layer(ticker: str, db_manager: DatabaseManager):
    """
    Visualizes 7-day Price vs. Sentiment.
    Includes logic to fetch specific ticker history and handle timezone mismatches.
    """
    # 1. Get API Key
    api_key = st.secrets.get("POLYGON_API_KEY") or os.environ.get("POLYGON_API_KEY")
    if not api_key:
        st.warning("⚠️ POLYGON_API_KEY not found. Cannot render Price Chart.")
        return

    st.subheader(f"👁️ The Truth Layer: {ticker} Price vs. Sentiment")

    # 2. Fetch Historical Price Data
    with st.spinner(f"Fetching market data for {ticker}..."):
        price_df = fetch_polygon_price_data(ticker, api_key)
        
        if price_df.empty:
            st.warning(f"Could not fetch price data for {ticker}. (API limit or invalid ticker)")
            return

    # 3. Fetch System's Detected Events (Targeted Query with Fallback)
    try:
        # Try to use the specific method if it exists (Fixes the "Limit 100" issue)
        if hasattr(db_manager, 'get_events_by_ticker'):
            ticker_events = db_manager.get_events_by_ticker(ticker, days=90)
        else:
            # Fallback: Get global recent events and filter manually
            # We bump limit to 200 to have a better chance of finding relevant news
            all_events = db_manager.get_recent_events(limit=200)
            ticker_events = [e for e in all_events if e['ticker'] == ticker]
            
        events_df = pd.DataFrame(ticker_events)
        
    except Exception as e:
        st.error(f"Error fetching events: {e}")
        events_df = pd.DataFrame()

    # 4. Create the Combined Chart
    fig = go.Figure()

    # Layer A: The Price (Truth)
    fig.add_trace(go.Scatter(
        x=price_df['timestamp'],
        y=price_df['close'],
        mode='lines',
        name='Market Price',
        line=dict(color='#636EFA', width=2)
    ))

    # Layer B: The Sentiment (Hypothesis)
    if not events_df.empty:
        try:
            # --- FIX: STANDARDIZE TIMEZONES FOR EVENTS ---
            # 1. Force to UTC datetime
            # The 'mixed' format tells Pandas to handle different ISO formats (like 'Z' suffixes) gracefully
            events_df['timestamp'] = pd.to_datetime(events_df['timestamp'], format='mixed', utc=True)
            # 2. Remove timezone info (make it naive) to match the Price DataFrame
            events_df['timestamp'] = events_df['timestamp'].dt.tz_localize(None)

            # Color logic
            colors = ['#00CC96' if x > 0 else '#EF553B' for x in events_df['score']]
            
            # Align dots to price line
            y_values = []
            for event_ts in events_df['timestamp']:
                # Find nearest price timestamp
                # Now both are "naive" datetimes, so this subtraction works!
                nearest_idx = (price_df['timestamp'] - event_ts).abs().idxmin()
                y_values.append(price_df.loc[nearest_idx, 'close'])

            fig.add_trace(go.Scatter(
                x=events_df['timestamp'],
                y=y_values,
                mode='markers',
                name='News Event',
                marker=dict(size=14, color=colors, line=dict(width=2, color='White')),
                text=events_df['headline'],
                hovertemplate="<b>%{text}</b><br>Time: %{x}<br>Price: $%{y:.2f}<extra></extra>"
            ))
        except Exception as e:
            st.warning(f"Could not render event dots due to timestamp error: {e}")
    else:
        st.caption("No significant news events detected for this ticker in the database (Last 7 Days).")

    fig.update_layout(
        template="plotly_dark",
        height=450,
        hovermode="x unified",
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    st.plotly_chart(fig)

# ==============================================================================
# --- FUNCTION: Inject Live Risk Scores ---
# ==============================================================================
def inject_live_risk_data(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Connects to Neo4j to fetch the latest 'raw_risk_score' for every node.
    """
    updated_count = 0
    try:
        # Create a temporary DB connection to fetch fresh scores
        # We use a temp connection because we can't easily pass the main db_manager 
        # into this @st.cache_data function without hashing issues.
        config = get_cloud_config_dict()
        temp_db = DatabaseManager(config)
        
        query = """
        MATCH (n:Company) 
        WHERE n.raw_risk_score IS NOT NULL 
        RETURN n.ticker as ticker, n.raw_risk_score as score
        """
        results = temp_db.execute_read(query)
        temp_db.close()
        
        if not results:
            logger.warning("⚠️ No risk scores found in Neo4j. Nodes will be Green.")
            return graph
            
        # Convert to Dictionary
        score_map = {row['ticker']: float(row['score']) for row in results}
        
        # Calculate Rank Percentiles for Colors
        df = pd.DataFrame(list(score_map.items()), columns=['Ticker', 'Score'])
        df['Rank'] = df['Score'].rank(method='first', pct=True)
        rank_map = pd.Series(df.Rank.values, index=df.Ticker).to_dict()

        # Apply to Graph
        for node in graph.nodes():
            if node in score_map:
                graph.nodes[node]['raw_risk_score'] = score_map[node]
                
                percentile = rank_map.get(node, 0.0)
                if percentile >= 0.95:   risk_level = 2 # Red
                elif percentile >= 0.70: risk_level = 1 # Orange
                else:                    risk_level = 0 # Green
                    
                graph.nodes[node]['predicted_risk'] = risk_level
                updated_count += 1
                
        logger.info(f"🎨 Successfully painted {updated_count} nodes with fresh Neo4j data.")
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch live risk from DB: {e}")
        
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
    """
    Loads the dictionary mapping Tickers -> Company Names.
    Also injects the Macro-Economic names so they appear correctly.
    """
    mapping = {}
    
    # 1. Load Standard Companies from JSON
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        FILE_PATH = os.path.join(SCRIPT_DIR, 'sp500_companies.json')
        
        with open(FILE_PATH, 'r') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            mapping = {item['ticker']: item['name'] for item in data if 'ticker' in item and 'name' in item}
        elif isinstance(data, dict):
            mapping = data

    except FileNotFoundError:
        pass

    # 2. Inject Macro Asset Names (Hardcoded override)
    # These match the tickers used in macro_ingest.py
    MACRO_NAMES = {
        "^TNX":  "10-Year Treasury Yield",
        "CL=F":  "Crude Oil",
        "GC=F":  "Gold",
        "DX-Y.NYB": "US Dollar Index",
        "^VIX":  "Volatility Index (Fear Gauge)"
    }
    
    mapping.update(MACRO_NAMES)
    
    return mapping

@st.cache_resource
def get_db_manager():
    """Cached function to initialize the database manager once."""
    if "neo4j" not in st.secrets:
        st.error("Neo4j credentials not found in Streamlit Secrets.")
        st.stop()
        
    cloud_config = {
        "neo4j": {
            "uri": st.secrets["neo4j"]["uri"],
            "user": st.secrets["neo4j"]["user"],
            "password": st.secrets["neo4j"]["password"]
        }
    }
    return DatabaseManager(cloud_config)

@st.cache_data(ttl=600) # Refresh every 10 mins
def get_full_graph():
    """
    Fetches the graph structure DIRECTLY from Neo4j (Live Mode).
    Ensures Macro nodes are preserved and sectors are cleaned.
    """
    logger.info("Fetching full graph from Neo4j...")
    
    try:
        config = get_cloud_config_dict()
        temp_db = DatabaseManager(config)
        
        # 1. Fetch Nodes
        nodes_query = """
        MATCH (n:Company) 
        RETURN n.ticker as id, n.name as name, n.sector as sector, 
               n.market_cap as market_cap, n.is_macro as is_macro
        """
        nodes_data = temp_db.execute_read(nodes_query)
        
        # 2. Fetch Edges
        edges_query = """
        MATCH (n:Company)-[r]->(m:Company)
        WHERE r.weight > 0.1
        RETURN n.ticker as source, m.ticker as target, 
               r.weight as weight, r.mechanism as mechanism, 
               r.verification_status as verification_status
        """
        edges_data = temp_db.execute_read(edges_query)
        temp_db.close()
        
        if not nodes_data: return None

        G = nx.DiGraph()
        
        # 3. Add Nodes & CLEAN SECTORS
        for n in nodes_data:
            raw_sector = n.get('sector', 'Unknown')
            if raw_sector is None: raw_sector = "Unknown"
            
            # Basic cleaning
            clean_sector = raw_sector.title() 
            if "Pharmaceautical" in clean_sector: clean_sector = "Healthcare"
            if "Services Computer" in clean_sector: clean_sector = "Technology"
            if "Discovered" in clean_sector: clean_sector = "Watchlist"
            
            # Force "Macro" sector for macro nodes
            if n.get('is_macro'): clean_sector = "Macro"
            
            G.add_node(
                n['id'], 
                name=n.get('name', n['id']),
                sector=clean_sector,
                market_cap=n.get('market_cap', 0),
                is_macro=n.get('is_macro', False)
            )
            
        # 4. Add Edges
        for e in edges_data:
            G.add_edge(
                e['source'], e['target'], 
                weight=e['weight'],
                mechanism=e.get('mechanism'),
                verification_status=e.get('verification_status')
            )
            
        # ==============================================================================
        # 🧹 NOISE REDUCTION (UPDATED TO PROTECT MACRO NODES)
        # ==============================================================================
        suspicious_parents = ['SNAP', 'V', 'META', 'GOOGL', 'GOOG', 'AMZN', 'ADBE', 'CRM', 'MA'] 
        edges_to_remove = []
        
        for u, v, data in G.edges(data=True):
            # SKIP check if either node is a Macro node (we want to keep those!)
            if G.nodes[u].get('is_macro') or G.nodes[v].get('is_macro'):
                continue

            sector_u = G.nodes[u].get('sector', 'Unknown')
            sector_v = G.nodes[v].get('sector', 'Unknown')
            weight = data.get('weight', 0.5)

            if u in suspicious_parents and weight < 0.90:
                edges_to_remove.append((u, v))
                continue
                
            # Cross-Sector Penalty
            if sector_u != 'Unknown' and sector_v != 'Unknown':
                if sector_u != sector_v and weight < 0.75:
                    edges_to_remove.append((u, v))

        if edges_to_remove:
            G.remove_edges_from(edges_to_remove)
            logger.info(f"🧹 Pruned {len(edges_to_remove)} noisy edges (Macro nodes protected).")
        # ==============================================================================

        logger.info(f"✅ Loaded {G.number_of_nodes()} nodes from Neo4j.")
        
        return inject_live_risk_data(G)

    except Exception as e:
        logger.error(f"Error loading graph from DB: {e}")
        return None

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
# --- GNN EXPLAINER HELPERS ---
# ==============================================================================
@st.cache_data
def get_sorted_tickers():
    """
    CRITICAL: Fetches all tickers sorted alphabetically.
    This enables us to map 'AAPL' -> Index 4 for the GNN.
    MUST match the sorting logic used in train.py/gnn_pipeline.py.
    """
    config = get_cloud_config_dict()
    db = DatabaseManager(config)
    # Ensure we sort ASCending to match standard LabelEncoder behavior
    query = "MATCH (c:Company) RETURN c.ticker as ticker ORDER BY c.ticker ASC"
    res = db.execute_read(query)
    db.close()
    return [r['ticker'] for r in res]

def visualize_explanation(explainer_graph: nx.DiGraph, node_map_list: list):
    """
    Renders the subgraph returned by GNNExplainer.
    Maps 'Company_123' back to 'AAPL' using the node_map_list.
    """
    # 1. Remap Labels (Index -> Ticker)
    mapping = {}
    for node in explainer_graph.nodes():
        if "Company_" in node:
            try:
                # Extract index "Company_42" -> 42
                idx = int(node.split("_")[1])
                if 0 <= idx < len(node_map_list):
                    mapping[node] = node_map_list[idx]
                else:
                    mapping[node] = f"Unknown_{idx}"
            except:
                mapping[node] = node
        else:
            mapping[node] = node # Event nodes or other types
            
    # Apply mapping
    G_remapped = nx.relabel_nodes(explainer_graph, mapping)
    
    # 2. PyVis Setup
    net = Network(height="600px", width="100%", notebook=True, directed=True, bgcolor="#111111", font_color="white")
    net.from_nx(G_remapped)
    
    # 3. Style based on Importance
    for edge in net.edges:
        # The explainer logic puts 'weight' as the importance score (0.0 - 1.0)
        # We retrieve it (PyVis sometimes moves attributes around)
        try:
            importance = float(edge.get('weight', 0.5)) 
        except:
            importance = 0.5

        edge['value'] = importance # Thickness based on importance
        edge['title'] = f"Importance: {importance:.2%}"
        
        # Color gradient: Low importance = Grey, High = Red/Gold
        if importance > 0.8:
            edge['color'] = "#FFD700" # Gold
        elif importance > 0.6:
            edge['color'] = "#FF4500" # RedOrange
        else:
            edge['color'] = "#777777" # Grey
            
    # 4. Style Nodes
    for node in net.nodes:
        # If it's a Ticker (exists in our map), color it Blue
        if node['id'] in mapping.values():
            node['color'] = "#00BFFF" # Deep Sky Blue for Companies
            node['size'] = 25
            node['shape'] = 'dot'
            
    net.set_options('{"physics": {"forceAtlas2Based": {"gravitationalConstant": -50, "springLength": 100}}}')
    
    # Save
    html_file = f"explainer_{uuid.uuid4().hex}.html"
    net.save_graph(html_file)
    return html_file

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
    tab_events, tab_explore, tab_simulate, tab_path, tab_xray, tab_help = st.tabs([
        "🔔 Recent Events", "🗺️ Explore Graph", "🔬 Simulate Scenarios", "↔️ Causal Pathfinding", "🔮 GNN X-Ray", "📘 Help & Guide"
    ])


    # --- TAB 1: RECENT EVENTS (UPDATED WITH FILTERS) ---
    with tab_events:
        st.header("🔔 Recently Detected Significant Events")
        
        # --- 1. Load Data for Filters ---
        # We need the graph to know which companies belong to which sectors
        financial_graph = get_full_graph()
        
        # Prepare lists for dropdowns
        if financial_graph is None:
            all_sectors = []
            all_nodes = []
        else:
            all_sectors = sorted(list(set(
                financial_graph.nodes[n].get('sector', 'Unknown') 
                for n in financial_graph.nodes()
                if financial_graph.nodes[n].get('sector')
            )))
            all_nodes = sorted(list(financial_graph.nodes()))

        # Helper to format names (Reused from other tabs)
        company_map = load_company_names()
        def format_ticker_events(ticker):
            if ticker == "All Companies": return "All Companies"
            name = company_map.get(ticker)
            if not name and financial_graph and financial_graph.has_node(ticker):
                name = financial_graph.nodes[ticker].get('name')
            return f"{name} ({ticker})" if name else ticker

        # --- 2. Filter Layout ---
        col_ev_filter, col_ev_select, col_ev_refresh = st.columns([1, 2, 1])

        with col_ev_filter:
            selected_sector_ev = st.selectbox(
                "Filter by Sector:", 
                ["All Sectors"] + all_sectors,
                key="event_sector_filter"
            )

        with col_ev_select:
            # Filter the company list based on the sector selected above
            filtered_nodes = all_nodes
            if selected_sector_ev != "All Sectors" and financial_graph:
                filtered_nodes = [
                    n for n in all_nodes 
                    if financial_graph.nodes[n].get('sector') == selected_sector_ev
                ]
            
            selected_ticker_ev = st.selectbox(
                "Filter by Company:", 
                ["All Companies"] + filtered_nodes,
                format_func=format_ticker_events,
                key="event_ticker_select"
            )

        with col_ev_refresh:
            st.write("") # Spacer to align button
            st.write("") 
            # OLD: if st.button("🔄 Refresh Events", use_container_width=True):
            
            # NEW: (Buttons still typically use use_container_width in many versions, 
            # but if it errors, try removing the argument or using help='stretch')
            if st.button("🔄 Refresh Events"): # Safest to remove arg if causing crash
                st.cache_data.clear() 
                st.cache_resource.clear() 
                st.rerun()

        st.divider()

        # --- 3. Fetch Data based on Selection ---
        events_to_display = []
        
        if selected_ticker_ev != "All Companies":
            # CASE A: Specific Ticker Selected -> Fetch specific events
            try:
                # Try to use the optimized method if it exists
                if hasattr(db_manager, 'get_events_by_ticker'):
                    events_to_display = db_manager.get_events_by_ticker(selected_ticker_ev, days=90)
                else:
                    # Fallback: Fetch all and filter (Slower, but safe)
                    all_events = db_manager.get_recent_events(limit=500)
                    events_to_display = [e for e in all_events if e['ticker'] == selected_ticker_ev]
                    
                st.info(f"Showing events for: **{format_ticker_events(selected_ticker_ev)}**")
            except Exception as e:
                st.error(f"Error fetching specific events: {e}")
        else:
            # CASE B: No Filter -> Fetch global recent events
            events_to_display = db_manager.get_recent_events(limit=50)
            st.write("Displaying the 50 most recent events across the market.")

        # --- 4. Render List ---
        if not events_to_display:
            if selected_ticker_ev != "All Companies":
                st.warning(f"No significant events found for {selected_ticker_ev} in the database.")
            else:
                st.info("No events detected yet.")
        else:
            for event in events_to_display:
                score = event.get('score', 0.0)
                # Determine emoji based on score
                if score > 0:
                    event_type = "Positive📈"
                    color_border = "green"
                elif score < 0:
                    event_type = "Negative📉"
                    color_border = "red"
                else:
                    event_type = "Neutral😐"
                    color_border = "grey"
                
                # Check for timestamp presence
                ts = event.get('timestamp', 'No Date')
                
                with st.expander(f"**{ts} - {event_type} for {event['ticker']}**: {event['headline']}"):
                    st.markdown(f"**Sentiment Score:** `{score:.2f}`")
                    st.markdown(f"[Read Full Article]({event['link']})", unsafe_allow_html=True)
                    
                    # Simulation Button
                    if st.button(f"🔬 Simulate Impact for {event['ticker']}", key=f"btn_{event.get('id', uuid.uuid4())}"):
                        st.info("Go to the 'Simulate Scenarios' tab to run this simulation manually.")

    # ==============================================================================
    # --- TAB 2: EXPLORE GRAPH ---
    # ==============================================================================
    with tab_explore:
        st.header("🗺️ Interactive Knowledge Graph Explorer")
        
    # --- NEW: MARKET WEATHER (SECTOR HEATMAP) - Z-SCORE EDITION ---
        with st.expander("🌤️ Market Weather (Sector Heatmap)", expanded=True):
            with st.spinner("Analyzing market sectors..."):
                sector_data = db_manager.get_sector_risk_data()
                
            if not sector_data:
                st.info("No sector data available yet. Run the worker to populate company nodes.")
            else:
                df_sectors = pd.DataFrame(sector_data)
                
                # --- CALCULATION: Z-SCORE (Relative Risk) ---
                # 1. Calculate Market Mean and Standard Deviation
                mu = df_sectors['AvgRisk'].mean()
                sigma = df_sectors['AvgRisk'].std()

                # 2. Compute Z-Score: (Value - Mean) / StdDev
                # This tells us how many "deviations" a sector is away from the average.
                if sigma == 0:
                    df_sectors['Z_Score'] = 0 # Avoid division by zero if all sectors are identical
                else:
                    df_sectors['Z_Score'] = (df_sectors['AvgRisk'] - mu) / sigma

                # 3. Create Treemap
                fig = px.treemap(
                    df_sectors, 
                    path=[px.Constant("Market"), 'Sector'], 
                    values='CompanyCount',
                    color='Z_Score', 
                    # RdYlGn_r = Red-Yellow-Green (Reversed). 
                    # High Positive Z (Riskier than avg) -> Red
                    # 0 (Average) -> Yellow
                    # Low Negative Z (Safer than avg) -> Green
                    color_continuous_scale='RdYlGn_r', 
                    color_continuous_midpoint=0, # Force the scale to center on the Market Average
                    custom_data=['AvgRisk'], # Pass the RAW risk score to the tooltip
                    title="Relative Sector Risk (Z-Score)"
                )
                
                # 4. Update Tooltip so you see the REAL risk score, not just the Z-score
                fig.update_traces(
                    hovertemplate='<b>%{label}</b><br>Companies: %{value}<br>Relative Risk (Z): %{color:.2f} σ<br><b>Actual Risk: %{customdata[0]:.4f}</b>'
                )

                fig.update_layout(margin=dict(t=30, l=10, r=10, b=10))
                st.plotly_chart(fig, use_container_width=True)

        st.write("Select a company to load its strongest relationships.") 

        with st.spinner("Loading full graph for explorer..."):
            # We load the full graph here primarily to get the LIVE RISK SCORES
            financial_graph = get_full_graph()
        
        if financial_graph is None:
                st.warning("Knowledge graph is empty. Please run `worker.py` (locally) to populate the *cloud* database.")
        else:
            # --- STEP 1: PREPARE FILTERS ---
            # Get all unique sectors from the graph data
            all_sectors = sorted(list(set(
                financial_graph.nodes[n].get('sector', 'Unknown') 
                for n in financial_graph.nodes()
                if financial_graph.nodes[n].get('sector') # filter out empty/None
            )))

            # Layout: Filter on left, Ticker select on right
            col_filter, col_select = st.columns([1, 2])

            with col_filter:
                selected_sector = st.selectbox(
                    "Filter by Sector:",
                    ["All Sectors"] + all_sectors,
                    index=0
                )

            # --- STEP 2: FILTER NODES ---
            all_nodes = sorted(list(financial_graph.nodes()))
            
            if selected_sector != "All Sectors":
                # Only keep nodes that match the selected sector
                all_nodes = [
                    n for n in all_nodes 
                    if financial_graph.nodes[n].get('sector') == selected_sector
                ]

            company_map = load_company_names()
            
            # --- UPDATED FORMATTER ---
            # This fixes the issue where Macro nodes appeared as tickers (e.g. "CL=F") 
            # instead of names (e.g. "Crude Oil")
            def format_ticker(ticker):
                # 1. Try JSON map first (Fastest)
                name = company_map.get(ticker)
                
                # 2. If not in JSON, check the Graph Node attributes (Fallback for Macro/New nodes)
                if not name and financial_graph.has_node(ticker):
                    name = financial_graph.nodes[ticker].get('name')
                
                if name:
                    return f"{name} ({ticker})"
                return ticker

            with col_select:
                selected_company = st.selectbox(
                    "Select a company/asset:", 
                    all_nodes, 
                    index=0 if all_nodes else None,
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
                                
                                # Transfer risk scores & clean sectors from Global Graph to Local Neighborhood Graph
                                for node in neighborhood_graph.nodes():
                                    if financial_graph.has_node(node):
                                        # Sync Risk
                                        neighborhood_graph.nodes[node]['predicted_risk'] = financial_graph.nodes[node].get('predicted_risk', 0)
                                        neighborhood_graph.nodes[node]['raw_risk_score'] = financial_graph.nodes[node].get('raw_risk_score', 0.0)
                                        # Sync Sector (Cleaned)
                                        neighborhood_graph.nodes[node]['sector'] = financial_graph.nodes[node].get('sector', 'Unknown')
                                        # Sync Name (for Tooltip)
                                        if 'name' not in neighborhood_graph.nodes[node]:
                                            neighborhood_graph.nodes[node]['name'] = financial_graph.nodes[node].get('name', node)

                                # Manually clean the graph edges of reserved keywords for Pyvis
                                graph_for_pyvis = neighborhood_graph.copy() 
                                for u, v, data in graph_for_pyvis.edges(data=True):
                                    data.pop('source', None)
                                    data.pop('target', None)

                                net = Network(height="750px", width="100%", notebook=True, cdn_resources='in_line', directed=True, bgcolor="#222222", font_color="white")
                                net.from_nx(graph_for_pyvis)
                                
                                # === NEW: APPLY AI VISUAL STYLES ===
                                # This turns "AI_PROPOSED" edges into dashed grey lines
                                net = apply_ai_visual_styles(net, graph_for_pyvis)
                                
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
                                    
                                # === REFRESH TO SHOW NEW HTML ===
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
                    st.caption("ℹ️ **Legend:** Solid Lines = Verified Data. Dashed Grey Lines = **AI Inferred** (Unverified).")
                else:
                    st.info("Click 'Explore Neighborhood' to generate the graph.")

            # --- COLUMN 2: AI ANALYSIS ---
            with col_ai:
                st.subheader("🤖 AI Insight")
                if not gemini_active:
                    st.write("Gemini API not active.")
                elif selected_company:
                    st.write(f"Analyze risks for **{selected_company}**.")
                    prompt_type = st.selectbox("Query Type:", ["Economic Logic", "Contagion Risk", "Structural Analysis"], key="ai_select")
                    
                    if st.button("🧠 Generate"):
                        with st.spinner("Consulting Gemini..."):
                            # Construct context
                            my_risk = financial_graph.nodes[selected_company].get('raw_risk_score', 0.0)
                            neighbors = list(financial_graph.neighbors(selected_company))[:5]
                            neighbor_str = ", ".join(neighbors)
                            
                            if prompt_type == "Structural Analysis":
                                prompt = f"""
                                **Role:** You are a graph theory expert analyzing {selected_company}.
                                **Data:** It connects to {neighbor_str}.
                                **Task:** Identify if {selected_company} is a 'load-bearing' node or a dependency. 
                                Who relies on it? Who does it rely on?
                                """
                            else:
                                prompt = f"Financial Analysis for {selected_company}. Risk Score: {my_risk}. Upstream Entities: {neighbor_str}. Explain the {prompt_type} of this graph structure."
                            
                            explanation = generate_ai_analysis(prompt)
                            st.success("Generated Insight:")
                            st.write(explanation)

    # ==============================================================================
    # --- TAB 3: SIMULATE SCENARIOS (UPDATED WITH TRUTH LAYER) ---
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
            if not name and financial_graph.has_node(ticker):
                name = financial_graph.nodes[ticker].get('name')
            if name:
                return f"{name} ({ticker})"
            return ticker

        # --- SECTION 1: SYSTEMIC VULNERABILITY ---
        with st.expander("🌪️ Systemic Vulnerability (The 'Butterfly Effect')", expanded=False):
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
                    top_10['Damage Magnitude'] = top_10['Systemic_Damage'].abs()
                    
                    st.bar_chart(top_10, x="Ticker", y="Damage Magnitude", color="#FF4B4B")
                    
                    # Full Data Table
                    # OLD: st.dataframe(..., use_container_width=True)
                    
                    # NEW:
                    st.dataframe(
                        vuln_df[['Ticker', 'Company Name', 'Systemic_Damage', ' impacted_count']], 
                        width=None, # Or removed entirely to let Streamlit decide
                        column_config={
                            "Systemic_Damage": st.column_config.NumberColumn("Total Network Loss", format="%.2f"),
                            " impacted_count": st.column_config.NumberColumn("Companies Affected")
                        }
                    )

        st.divider()

        # --- SECTION 2: MANUAL SIMULATION & TRUTH LAYER ---
        st.subheader("🧪 Manual Simulation & Reality Check")
        
        # --- 1. Get Sectors for Filter ---
        all_sectors = sorted(list(set(
            financial_graph.nodes[n].get('sector', 'Unknown') 
            for n in financial_graph.nodes()
        )))

        # --- 2. Layout: Filter | Company Select | Shock Slider ---
        sim_col_filter, sim_col_asset, sim_col_shock = st.columns([1, 2, 1])
        
        with sim_col_filter:
            sim_sector = st.selectbox("Filter Sector:", ["All Sectors"] + all_sectors, key="sim_sector_filter")

        # --- 3. Filter Nodes ---
        all_nodes = sorted(list(financial_graph.nodes()))
        if sim_sector != "All Sectors":
            all_nodes = [n for n in all_nodes if financial_graph.nodes[n].get('sector') == sim_sector]

        with sim_col_asset:
            selected_company_sim = st.selectbox(
                "Target Asset:", 
                all_nodes, 
                format_func=format_ticker,
                key="sim_select_company"
            )
        
        with sim_col_shock:
            hypothetical_score = st.slider(
                "Hypothetical Shock:",
                min_value=-1.0, max_value=1.0, value=-0.5, step=0.05,
                help="-1.0 = Total Collapse, +1.0 = Huge Breakout"
            )

        # --- THE TRUTH LAYER (CHART) ---
        if selected_company_sim:
            render_truth_layer(selected_company_sim, db_manager)

        st.divider()

        if st.button("💥 Simulate Event", key="btn_manual_sim", type="primary"):
            if not selected_company_sim:
                st.warning("Please select an asset.")
            else:
                st.subheader(f"Simulating impact of {hypothetical_score} event on {format_ticker(selected_company_sim)}")
                
                impact_results = calculate_impact_scores(financial_graph, selected_company_sim, hypothetical_score)
                
                res_col1, res_col2 = st.columns([1, 2])
                with res_col1:
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
                        # Sort by absolute impact
                        df = df.sort_values(by="Impact Score", key=abs, ascending=False)
                        
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

                        # === GENERATE CRISIS REPORT ===
                        if gemini_active:
                            st.divider()
                            if st.button("📝 Generate Crisis Report"):
                                with st.spinner("Writing AI Crisis Report..."):
                                    top_victims = df.head(5)['Ticker'].tolist()
                                    prompt = f"""
                                    **Context:** A simulated shock of {hypothetical_score} hit {selected_company_sim}.
                                    **Data:** The algorithmic impact analysis shows these casualties: {top_victims}.
                                    **Task:** Write a short 'Breaking News' style financial alert explaining the contagion mechanism. Use Wyckoff logic if applicable.
                                    """
                                    report = generate_ai_analysis(prompt)
                                    st.info("🚨 **Breaking Crisis Report**")
                                    st.write(report)
                    
                    else:
                        st.info("No downstream impacts found (Node might be isolated or impact < threshold).")
                        st.session_state['sim_impact_nodes'] = []

                with res_col2:
                    st.subheader("Visual Impact Graph")
                    
                    if st.session_state.get('sim_impact_nodes'):
                        # Sort impacts
                        sorted_nodes = sorted(impact_results.items(), key=lambda item: abs(item[1]['score']), reverse=True)
                        # Slice top 25
                        top_victims = [n for n, data in sorted_nodes[:25]]
                        # Define subgraph
                        nodes_to_include = {selected_company_sim, *top_victims}
                        subgraph = financial_graph.subgraph(nodes_to_include)
                        
                        # Setup Pyvis
                        graph_for_pyvis = subgraph.copy()
                        for u, v, data in graph_for_pyvis.edges(data=True):
                            data.pop('source', None); data.pop('target', None)

                        net = Network(height="500px", width="100%", notebook=True, directed=True, bgcolor="#222222", font_color="white")
                        net.from_nx(graph_for_pyvis) 
                        net = apply_ai_visual_styles(net, graph_for_pyvis)
                        
                        # Physics to prevent clustering
                        net.set_options('{"physics": {"barnesHut": {"gravitationalConstant": -10000, "springLength": 150}}}')

                        # Coloring
                        for node in net.nodes:
                            nid = node['id']
                            node['label'] = nid
                            if nid == selected_company_sim:
                                node['color'], node['size'], node['shape'] = '#FF4B4B', 40, 'diamond'
                            elif nid in st.session_state['sim_impact_nodes']:
                                node['color'], node['size'] = '#FFA726', 20
                                score = impact_results.get(nid, {}).get('score', 0)
                                node['title'] = f"Impact: {score:.4f}"

                        # Render
                        html_file = f"temp_graph_{uuid.uuid4().hex}.html"
                        try:
                            net.save_graph(html_file)
                            with open(html_file, 'r', encoding='utf-8') as f:
                                st.components.v1.html(f.read(), height=520, scrolling=True)
                        except Exception as e:
                            st.error(f"Graph Error: {e}")
                        finally:
                            if os.path.exists(html_file): os.remove(html_file)
                    else:
                        st.write("Run a simulation to see the graph.")

    # ==============================================================================
    # --- TAB 4: CAUSAL PATHFINDING (UPDATED WITH SECTOR FILTER) ---
    # ==============================================================================
    with tab_path:
        st.header("↔️ Causal Pathfinding")
        
        # 1. LOAD THE GRAPH FIRST
        with st.spinner("Loading full graph for pathfinding..."):
            financial_graph = get_full_graph()
        
        if financial_graph is None:
                st.warning("Knowledge graph is empty. Please run `worker.py` (locally) to populate the *cloud* database.")
        else:
            st.write(f"Finds the shortest path between two nodes in the pre-filtered graph (the {financial_graph.number_of_edges()} strongest relationships).")
            
            # Prepare Data
            all_nodes = sorted(list(financial_graph.nodes()))
            all_sectors = sorted(list(set(financial_graph.nodes[n].get('sector', 'Unknown') for n in all_nodes)))
            company_map = load_company_names()
            
            def format_ticker(ticker):
                name = company_map.get(ticker)
                if not name and financial_graph.has_node(ticker):
                    name = financial_graph.nodes[ticker].get('name')
                return f"{name} ({ticker})" if name else ticker

            path_col1, path_col2 = st.columns(2)
            
            # --- COLUMN 1: START NODE ---
            with path_col1:
                st.subheader("1. The Trigger")
                # Filter Start
                start_sector = st.selectbox("Filter Start Sector:", ["All Sectors"] + all_sectors, key="path_start_filter")
                start_options = all_nodes
                if start_sector != "All Sectors":
                    start_options = [n for n in all_nodes if financial_graph.nodes[n].get('sector') == start_sector]

                start_node = st.selectbox(
                    "Start Company:", 
                    start_options, 
                    format_func=format_ticker, 
                    key="path_start"
                )

                # DIAGNOSTICS
                is_sink_node = False 
                if start_node and start_node in financial_graph:
                    out_degree = financial_graph.out_degree(start_node)
                    in_degree = financial_graph.in_degree(start_node)
                    st.caption(f"📊 **Node Stats:** `{out_degree} Outgoing` | `{in_degree} Incoming`")
                    
                    if out_degree == 0:
                        is_sink_node = True
                        st.error(f"🚫 **Sink Node:** {start_node} has no outgoing connections.")
                        if in_degree > 0:
                            if st.button("🔄 Reverse Analysis (What drives this?)"):
                                st.divider()
                                st.subheader(f"🕵️‍♀️ Drivers of {start_node}")
                                parents = list(financial_graph.predecessors(start_node))
                                parent_risks = []
                                for p in parents:
                                    r_score = financial_graph.nodes[p].get('raw_risk_score', 0)
                                    p_name = company_map.get(p, p)
                                    parent_risks.append({'Ticker': p, 'Name': p_name, 'Risk': r_score})
                                st.dataframe(pd.DataFrame(parent_risks).sort_values("Risk", ascending=False), hide_index=True)
                                st.stop() 

            # --- DYNAMIC REACHABILITY CALCULATION ---
            reachable_nodes = {}
            if start_node and not is_sink_node:
                try:
                    # Find everything reachable within 5 hops
                    reachable_nodes = nx.single_source_shortest_path_length(financial_graph, start_node, cutoff=5)
                except: reachable_nodes = {}
            
            # Valid targets are nodes that are reachable AND match the end filter
            valid_targets_raw = sorted([n for n in reachable_nodes.keys() if n != start_node])

            # --- COLUMN 2: END NODE ---
            with path_col2:
                st.subheader("2. The Target")
                
                if is_sink_node:
                    st.info("🚫 Cannot select target (Dead End).")
                    end_node = None
                elif not valid_targets_raw:
                    st.warning("⚠️ No companies reachable within 5 steps.")
                    end_node = None
                else:
                    # Filter End
                    end_sector = st.selectbox("Filter End Sector:", ["All Sectors"] + all_sectors, key="path_end_filter")
                    
                    # Apply both Reachability AND Sector filters
                    final_end_options = valid_targets_raw
                    if end_sector != "All Sectors":
                        final_end_options = [
                            n for n in valid_targets_raw 
                            if financial_graph.nodes[n].get('sector') == end_sector
                        ]

                    st.caption(f"✅ {len(final_end_options)} reachable targets in this sector.")
                    
                    end_node = st.selectbox(
                        "End Company:", 
                        final_end_options, 
                        format_func=format_ticker, 
                        key="path_end"
                    )

            # --- EXECUTE PATHFINDING ---
            st.divider()
            if not is_sink_node and st.button("🗺️ Find Riskiest Path"):
                if start_node and end_node:
                    try:
                        all_paths = list(nx.all_simple_paths(financial_graph, source=start_node, target=end_node, cutoff=5))
                        
                        if not all_paths:
                            st.error("No path found.")
                        else:
                            # Find path with highest cumulative risk
                            best_path = None
                            max_risk_score = -1
                            for path in all_paths:
                                current_risk_score = sum([financial_graph.nodes[nid].get('predicted_risk', 0) for nid in path])
                                if current_risk_score > max_risk_score:
                                    max_risk_score = current_risk_score
                                    best_path = path
                            
                            st.success(f"**Highest-Risk Path Found:**")
                            st.code(' -> '.join(best_path))
                            
                            # AI Explanation
                            if gemini_active:
                                with st.expander("🧠 Explain this Path Logic", expanded=True):
                                    with st.spinner("Analyzing..."):
                                        prompt = f"Explain the economic contagion logic: {' -> '.join(best_path)}."
                                        st.write(generate_ai_analysis(prompt))

                            # Visualization
                            path_graph = financial_graph.subgraph(best_path)
                            graph_for_pyvis = path_graph.copy()
                            for u, v, data in graph_for_pyvis.edges(data=True):
                                data.pop('source', None); data.pop('target', None)
                            
                            net = Network(height="400px", width="100%", notebook=True, directed=True, bgcolor="#222222", font_color="white")
                            net.from_nx(graph_for_pyvis) 
                            net = apply_ai_visual_styles(net, graph_for_pyvis)
                            
                            # Color nodes
                            risk_map = {0: "#66bb6a", 1: "#ffa726", 2: "#ef5350"}
                            for node_id in best_path:
                                node = net.get_node(node_id)
                                node_risk = financial_graph.nodes[node_id].get('predicted_risk', 0)
                                node['color'] = risk_map.get(node_risk, "#66bb6a")
                                node['label'] = node_id
                                if node_id == start_node: node['size'] = 30
                                if node_id == end_node: node['shape'] = 'star'

                            net.set_options('{"physics": {"enabled": false}}')
                            
                            html_file = f"temp_graph_{uuid.uuid4().hex}.html"
                            net.save_graph(html_file)
                            with open(html_file, 'r', encoding='utf-8') as f:
                                st.components.v1.html(f.read(), height=420, scrolling=True)
                            os.remove(html_file)

                    except Exception as e:
                        st.error(f"Pathfinding Error: {e}")

    # ==============================================================================
    # --- TAB 5: GNN X-RAY (EXPLAINABILITY) ---
    # ==============================================================================
    with tab_xray:
        st.header("🔮 GNN X-Ray: Why did the model predict this?")
        st.markdown("""
        This tool uses **GNNExplainer** to mathematically identify the specific sub-graph that contributed most to a prediction.
        It answers: *"Did the model flag this company because of its own financials, or because of a supplier crash?"*
        """)

        # 1. Load Data for Filters & Indexing
        financial_graph = get_full_graph()
        full_sorted_tickers = get_sorted_tickers() 
        
        if financial_graph is None:
            st.warning("Knowledge graph is empty. Please ensure data is loaded.")
        else:
            # 2. Prepare Sectors
            all_sectors = sorted(list(set(
                financial_graph.nodes[n].get('sector', 'Unknown') 
                for n in financial_graph.nodes()
                if financial_graph.nodes[n].get('sector')
            )))

            # 3. Filter Layout
            col_xr_filter, col_xr_sel, col_xr_act = st.columns([1, 2, 1])

            with col_xr_filter:
                selected_sector_xr = st.selectbox("Filter Sector:", ["All Sectors"] + all_sectors, key="xray_sector_filter")

            # Filter Logic
            display_tickers = full_sorted_tickers
            if selected_sector_xr != "All Sectors":
                display_tickers = [
                    n for n in full_sorted_tickers 
                    if financial_graph.has_node(n) and financial_graph.nodes[n].get('sector') == selected_sector_xr
                ]

            company_map = load_company_names()
            def format_xray_ticker(ticker):
                name = company_map.get(ticker)
                if not name and financial_graph.has_node(ticker):
                    name = financial_graph.nodes[ticker].get('name')
                return f"{name} ({ticker})" if name else ticker

            with col_xr_sel:
                xray_ticker = st.selectbox("Select Company to Explain:", display_tickers, format_func=format_xray_ticker, key="xray_ticker_box")
            
            with col_xr_act:
                st.write("") 
                st.write("") 
                run_xray = st.button("🔍 Run Explainability Engine", type="primary")

            # --- EXECUTION LOGIC (WITH SESSION STATE) ---
            
            # If button clicked, run math and SAVE to session state
            if run_xray and xray_ticker:
                try:
                    if xray_ticker not in full_sorted_tickers:
                        st.error("Ticker not found in global mapping.")
                    else:
                        target_idx = full_sorted_tickers.index(xray_ticker)
                        
                        with st.status("Running Mathematical Explanation...", expanded=True) as status:
                            status.write("🧠 Loading Neural Network & Graph Data...")
                            model, data, config = get_inference_resources()
                            
                            if model is None:
                                status.update(label="Error loading model.", state="error")
                                st.stop()
                                
                            status.write("⚙️ Configuring GNNExplainer Mask...")
                            explainer = setup_explainer(model)
                            
                            status.write(f"📉 Optimizing Mask for Node {target_idx} ({xray_ticker})...")
                            # Step 2: Run explanation (It now returns Top-20 edges only)
                            subgraph = explain_prediction(explainer, data, target_idx)
                            
                            status.update(label="Explanation Complete!", state="complete")
                        
                        # SAVE RESULTS TO STATE
                        st.session_state['xray_subgraph'] = subgraph
                        st.session_state['xray_target_ticker'] = xray_ticker

                except Exception as e:
                    st.error(f"X-Ray Failed: {e}")
                    logger.error(f"X-Ray Error: {e}")

            # --- DISPLAY LOGIC (CHECKS SESSION STATE) ---
            # This runs on every reload, keeping the graph visible
            if 'xray_subgraph' in st.session_state and st.session_state['xray_subgraph'] is not None:
                
                subgraph = st.session_state['xray_subgraph']
                target = st.session_state['xray_target_ticker']
                
                if subgraph.number_of_nodes() == 0:
                    st.warning("The model did not find specific neighbors that strongly influenced this prediction.")
                else:
                    st.divider()
                    st.subheader(f"The 'Why' for {target}")
                    st.caption("Showing the **Top 20** mathematical factors driving the risk score.")
                    
                    html_file = visualize_explanation(subgraph, full_sorted_tickers)
                    with open(html_file, 'r', encoding='utf-8') as f:
                        st.components.v1.html(f.read(), height=600, scrolling=True)
                    if os.path.exists(html_file): os.remove(html_file)

                    # --- AI INTERPRETATION ---
                    st.divider()
                    if gemini_active:
                        if st.button("🤖 Interpret this Subgraph"):
                            with st.spinner("Analyzing causal structure..."):
                                factor_nodes = [n for n in subgraph.nodes if "Company" in str(n) or "Event" in str(n)]
                                readable_factors = []
                                for f in factor_nodes:
                                    if "Company_" in f:
                                        try:
                                            idx = int(f.split("_")[1])
                                            if idx < len(full_sorted_tickers): 
                                                readable_factors.append(full_sorted_tickers[idx])
                                        except: pass
                                    else:
                                        readable_factors.append(f)
                                
                                prompt = f"""
                                The GNN model predicted a risk score for {target}. 
                                The 'Explainability' engine identified these specific nodes as the mathematical drivers of that decision: {readable_factors}.
                                Explain WHY these specific connections likely drove the risk calculation.
                                """
                                explanation = generate_ai_analysis(prompt)
                                st.info("🤖 **Gemini Analysis**")
                                st.write(explanation)

                except Exception as e:
                    st.error(f"X-Ray Failed: {e}")
                    logger.error(f"X-Ray Error: {e}")

    # ==============================================================================
    # --- TAB 6: INSTRUCTIONS & GUIDE ---
    # ==============================================================================
    with tab_help:
        st.header("📘 User Guide & Documentation")
        
        # Sidebar for quick navigation within the Help tab
        help_choice = st.radio(
            "Navigate Guide:",
            ["Introduction", "Recent Events", "Explore Graph", "Simulate Scenarios", "Causal Pathfinding", "GNN X-Ray"],
            horizontal=True
        )
        
        st.divider()

        if help_choice == "Introduction":
            st.markdown("""
            ### Welcome to the Causal Inference Engine
            This application moves beyond simple price correlations to map the **causal structure** of the financial markets.

            **New in this Version:**
            * **🔮 GNN X-Ray:** A "Glass Box" tool that explains *why* the AI predicted a specific risk score.
            * **🏭 Supply Chain Mining:** The system reads **SEC 10-K Filings** to find hard supply chain dependencies.
            * **🌍 Macro Intelligence:** The graph includes "Super Nodes" like **Crude Oil** and **10-Year Treasury Yields**.

            **Visual Legend:**
            * 🔴 **Red Node:** High Risk (Top 5% Volatility).
            * 🟢 **Green Node:** Low Risk (Stable).
            * 🔵 **Blue Line:** 📄 **SEC 10-K Filing** (Hard Supply Chain Evidence).
            * **Solid Line:** ✅ Verified data (Standard).
            * **Dashed Grey Line:** 🤖 AI Inferred (High Probability).
            """)
        
        elif help_choice == "Recent Events":
            st.markdown("""
            ### 🔔 Recent Events Tab
            **Purpose:** A feed of market "shocks" (News, Earnings, Macro Data) detected by the daily ingestion engine.

            **How to Use:**
            * **Refresh Events:** Reloads the latest data from the database.
            * **Inspect Source:** Click the link to read the original article.

            **Interpreting the Data:**
            * 🔴 **Negative (-1.0):** Bearish news (Lawsuits, Missed Earnings, Rate Hikes).
            * 🟢 **Positive (+1.0):** Bullish news (Mergers, Record Profits, Rate Cuts).
            """)
        
        elif help_choice == "Explore Graph":
            st.markdown("""
            ### 🗺️ Explore Graph Tab
            **Purpose:** Visual map of your financial universe.

            **New Feature: Sector Filters**
            * Use the **"Filter by Sector"** dropdown to isolate specific industries.
            * Select **"Macro"** to see how global factors like **Oil** or **Bond Yields** connect to the stock market.

            **Interpreting the Lines:**
            * **Thick Blue Line:** A relationship explicitly stated in a company's annual report (10-K).
            * **Dashed Grey Line:** A relationship inferred by Gemini AI based on news context.
            """)
        
        elif help_choice == "Simulate Scenarios":
            st.markdown("""
            ### 🔬 Simulate Scenarios Tab
            **Purpose:** A "Stress Test" lab. Ask "What If?" questions to forecast contagion.

            **Tools:**
            1.  **🌪️ Systemic Vulnerability:** Identifies **"Super Spreaders"**—companies whose failure would cause the most total damage to the network.
            2.  **🧪 Manual Simulation:** Select a trigger asset and a shock score to see the "Blast Radius" and generate an AI Crisis Report.
            """)

        elif help_choice == "Causal Pathfinding":
            st.markdown("""
            ### ↔️ Causal Pathfinding Tab
            **Purpose:** Find the hidden transmission chains between two assets (e.g., *How does a Bond Yield spike affect Nvidia?*).

            **Smart Diagnostics:**
            * **Dynamic Filtering:** Only shows nodes that are *actually reachable* from your start node.
            * **🚫 Sink Node Detection:** Warns you if a company is a "dead end" (absorbs shocks but doesn't pass them on).
            """)

        elif help_choice == "GNN X-Ray":
            st.markdown("""
            ### 🔮 GNN X-Ray (Explainability) Tab
            **Purpose:** The "Why" button for Artificial Intelligence. 
            
            Standard AI gives you a score (e.g., "High Risk"), but acts like a Black Box. This tool turns it into a **Glass Box**.

            **How it Works (The Math):**
            * It uses **GNNExplainer**, a state-of-the-art algorithm that mathematically "masks" parts of the graph to see what changes the prediction.
            * It asks: *"If I hide this Supplier relationship, does the risk score change?"* If yes, that supplier is critical.

            **Visual Legend:**
            * 🟡 **Gold / 🔴 Red Edges:** The **"Smoking Gun."** These are the critical connections that drove the AI's decision.
            * ⚪ **Grey/Thin Edges:** Background noise. The AI ignored these relationships for this specific prediction.
            """)

if __name__ == "__main__":
    main()