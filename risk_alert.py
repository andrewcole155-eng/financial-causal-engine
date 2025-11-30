# ==============================================================================
# --- RISK ALERT SYSTEM ("The Watchman") ---
# ==============================================================================
import json
import logging
import pandas as pd
import os
from database_manager import DatabaseManager

# --- CONFIGURATION ---
# 1. Define your portfolio/watchlist here
WATCHLIST = ["INTC", "SNAP", "IONQ", "KR", "KO", "OXY", "SIRI", "AMD"]

# 2. Risk Threshold (0.0 to 1.0)
# Alert if a neighbor's risk score is above this number.
RISK_THRESHOLD = 0.85 

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("RiskAlert")

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
        with open("config.json", 'r') as f:
            config = json.load(f)
    except:
        logger.error("❌ Config not found.")
        return

    # 2. Connect to Database
    db = DatabaseManager(config)
    
    # 3. Load Risk Data
    risk_map = load_live_risk_scores()
    
    logger.info("---------------------------------------------------")
    logger.info(f"🛡️  STARTING WATCHMAN SCAN FOR: {len(WATCHLIST)} TICKERS")
    logger.info("---------------------------------------------------")
    
    alerts_triggered = 0

    for ticker in WATCHLIST:
        # Get immediate neighbors (1-hop)
        # The Manager handles the "AI Priority" logic automatically now!
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
                # Try both directions
                edge_data = graph.get_edge_data(ticker, neighbor)
                if not edge_data:
                    edge_data = graph.get_edge_data(neighbor, ticker)
                
                if edge_data:
                    alerts_triggered += 1
                    mechanism = edge_data.get('mechanism', 'Direct Correlation')
                    status = edge_data.get('verification_status', 'VERIFIED')
                    
                    # Formatting the Alert
                    icon = "🤖" if status == "AI_PROPOSED" else "🔗"
                    
                    print(f"\n🚨 [RISK ALERT] {ticker} is exposed to {neighbor}!")
                    print(f"   🔥 Risk Score: {neighbor_risk:.4f}")
                    print(f"   {icon} Link Type:  {status}")
                    print(f"   📝 Mechanism:  {mechanism}")
                    print("   ---------------------------------------------------")

    if alerts_triggered == 0:
        logger.info("✅ Scan Complete. No immediate contagion risks found above threshold.")
    else:
        logger.info(f"\n⚠️ Scan Complete. {alerts_triggered} Alerts Triggered.")

    db.close()

if __name__ == "__main__":
    run_risk_scan()