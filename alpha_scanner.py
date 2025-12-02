# ==============================================================================
# --- ALPHA SCANNER (The Opportunity Hunter) ---
# ==============================================================================
import yfinance as yf
import pandas as pd
import numpy as np
import logging
import os
import json
import networkx as nx
from database_manager import DatabaseManager

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("AlphaScanner")

# --- CONFIGURATION ---
LOOKBACK_WINDOW = "5d"      # Download last 5 days of data
RETURN_PERIOD = 3           # Compare returns over the last 3 days (Rolling Lag)
DIVERGENCE_THRESHOLD = 0.03 # 3% Gap required to trigger a signal

def load_config():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        with open(config_path, 'r') as f: return json.load(f)
    except: return {}

def build_graph(config):
    """Rebuilds the graph structure from Neo4j to analyze connections."""
    db = DatabaseManager(config)
    
    # Fetch all edges with high confidence
    query = """
    MATCH (source:Company)-[r]->(target:Company)
    WHERE r.weight > 0.5
    RETURN source.ticker as source, target.ticker as target, r.weight as weight
    """
    results = db.execute_read(query)
    db.close()
    
    G = nx.DiGraph()
    for r in results:
        G.add_edge(r['source'], r['target'], weight=r['weight'])
    
    return G

def get_market_returns(tickers):
    """Fetches price data and calculates cumulative returns."""
    if not tickers: return pd.Series()
    
    # Clean tickers for Yahoo (Dot to Dash)
    yahoo_tickers = [t.replace('.', '-') for t in tickers]
    
    try:
        # Fetch data
        data = yf.download(yahoo_tickers, period=LOOKBACK_WINDOW, progress=False)['Close']
        
        # Calculate the % return over the configured period (e.g., last 3 days)
        # Formula: (Current Price - Price N days ago) / Price N days ago
        if len(data) < RETURN_PERIOD:
            logger.warning("Not enough data for return calculation.")
            return pd.Series()
            
        # Get percentage change from start of window to today
        returns = data.pct_change(periods=RETURN_PERIOD - 1).iloc[-1]
        
        # Map back to DB tickers (Dash to Dot)
        returns.index = returns.index.str.replace('-', '.')
        return returns
        
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        return pd.Series()

def scan_for_alpha():
    config = load_config()
    if not config: return

    logger.info("🕸️  Building Causal Graph...")
    G = build_graph(config)
    all_nodes = list(G.nodes())
    
    if not all_nodes:
        logger.warning("Graph is empty. Run ingestion first.")
        return

    logger.info(f"📉 Fetching price data for {len(all_nodes)} assets...")
    returns_map = get_market_returns(all_nodes)
    
    logger.info("🔍 Scanning for Divergences (Mispriced Assets)...")
    logger.info(f"   Parameters: Lookback={RETURN_PERIOD} days | Threshold={DIVERGENCE_THRESHOLD*100}%")
    print("\n" + "="*60)
    print(f"{'TICKER':<8} | {'ACTUAL %':<10} | {'EXPECTED %':<10} | {'DIVERGENCE':<10} | {'SIGNAL'}")
    print("="*60)
    
    opportunities = 0

    for target in all_nodes:
        # 1. Who drives this target? (Get Incoming Edges / Predecessors)
        drivers = list(G.predecessors(target))
        
        if not drivers: continue # Skip nodes with no parents
        if target not in returns_map or np.isnan(returns_map[target]): continue

        # 2. Calculate "Expected Return" based on Drivers
        expected_return = 0.0
        total_weight = 0.0
        driver_count = 0
        
        for driver in drivers:
            if driver in returns_map and not np.isnan(returns_map[driver]):
                weight = G[driver][target]['weight']
                driver_ret = returns_map[driver]
                
                expected_return += (driver_ret * weight)
                total_weight += weight
                driver_count += 1
        
        if total_weight == 0: continue
        
        # Weighted Average of Driver Performance
        expected_return /= total_weight
        
        # 3. Calculate Divergence (The "Lag")
        actual_return = returns_map[target]
        divergence = actual_return - expected_return
        
        # 4. Generate Signals
        # Condition: Target is DOWN (or flat), but Drivers are UP -> BUY
        # Divergence will be NEGATIVE (e.g., Actual 0% - Expected 5% = -5%)
        
        if divergence < -DIVERGENCE_THRESHOLD:
            print(f"{target:<8} | {actual_return:>7.2%}    | {expected_return:>7.2%}    | {divergence:>7.2%}    | 🟢 BUY (Lagging)")
            opportunities += 1
            
        # Condition: Target is UP, but Drivers are DOWN -> SHORT (Mean Reversion)
        elif divergence > DIVERGENCE_THRESHOLD:
            print(f"{target:<8} | {actual_return:>7.2%}    | {expected_return:>7.2%}    | {divergence:>7.2%}    | 🔴 SELL (Overextended)")
            opportunities += 1

    print("="*60)
    if opportunities == 0:
        logger.info("No significant divergences found. Market is efficient today.")
    else:
        logger.info(f"Found {opportunities} potential alpha opportunities.")

if __name__ == "__main__":
    scan_for_alpha()