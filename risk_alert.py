# ==============================================================================
# --- RISK ALERT SYSTEM ("The Watchman") ---
# ==============================================================================
import json
import logging
import pandas as pd
import os
from database_manager import DatabaseManager

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("RiskAlert")

# 1. Define Thresholds
RISK_THRESHOLD = 0.85 

def load_live_risk_scores(csv_path="live_risk_scores.csv"):
    """Loads the CSV and returns a dictionary: {'TICKER': 0.95, ...}"""
    if not os.path.exists(csv_path):
        logger.warning(f"⚠️ Risk CSV not found at {csv_path}. Assuming no external risks.")
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        # Clean and map
        return pd.Series(df.Risk_Score.values, index=df.Ticker).to_dict()
    except Exception as e:
        logger.error(f"❌ Failed to load risk scores: {e}")
        return {}

def run_risk_scan():
    # 1. Load Config
    try:
        # Determine path relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # --- DYNAMIC WATCHLIST ---
        # Reads the list you added to config.json
        watchlist = config.get("target_tickers", [])
        
        if not watchlist:
            logger.warning("⚠️ 'target_tickers' list is empty or missing in config.json. Nothing to scan.")
            return

    except Exception as e:
        logger.error(f"❌ Config error: {e}")
        return

    # 2. Connect to Database
    db = DatabaseManager(config)
    
    # 3. Load Risk Data
    csv_path = os.path.join(script_dir, "live_risk_scores.csv")
    risk_map = load_live_risk_scores(csv_path)
    
    logger.info("---------------------------------------------------")
    logger.info(f"🛡️  STARTING WATCHMAN SCAN FOR: {len(watchlist)} TICKERS")
    logger.info("---------------------------------------------------")
    
    alerts_triggered = 0

    for ticker in watchlist:
        ticker = ticker.upper().strip() # Safety clean
        
        # Get immediate neighbors (1-hop)
        graph = db.get_neighborhood_graph(ticker)
        
        if graph.number_of_nodes() == 0:
            continue
            
        # Check every neighbor in the graph
        for neighbor in graph.nodes():
            if neighbor == ticker: continue # Skip self
            
            # Get the risk score for this neighbor from our CSV map
            neighbor_risk = risk_map.get(neighbor, 0.0)
            
            if neighbor_risk >= RISK_THRESHOLD:
                # 🚨 HIGH RISK NEIGHBOR FOUND
                
                # Get the edge details (Why are they connected?)
                edge_data = graph.get_edge_data(ticker, neighbor)
                if not edge_data:
                    edge_data = graph.get_edge_data(neighbor, ticker)
                
                if edge_data:
                    alerts_triggered += 1
                    mechanism = edge_data.get('mechanism', 'Direct Correlation')
                    status = edge_data.get('verification_status', 'VERIFIED')
                    
                    # Log to File/Console
                    print(f"\n🚨 [RISK ALERT] {ticker} is exposed to {neighbor}!")
                    print(f"   🔥 Risk Score: {neighbor_risk:.4f}")
                    print(f"   🔗 Link Type:  {status}")
                    print(f"   📝 Mechanism:  {mechanism}")
                    print("   ---------------------------------------------------")

    if alerts_triggered == 0:
        logger.info("✅ Scan Complete. No immediate contagion risks found above threshold.")
    else:
        logger.info(f"\n⚠️ Scan Complete. {alerts_triggered} Alerts Triggered.")

    db.close()

if __name__ == "__main__":
    run_risk_scan()