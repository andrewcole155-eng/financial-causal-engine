# ==============================================================================
# --- RISK ALERT SYSTEM ("The Watchman") ---
# ==============================================================================
import json
import logging
import pandas as pd
import os
import requests  # <--- Required for Discord
import time
from database_manager import DatabaseManager

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("RiskAlert")

# 1. Define Thresholds
RISK_THRESHOLD = 0.85 

def send_discord_alert(webhook_url, ticker, neighbor, score, mechanism, status):
    """Sends a formatted alert to Discord."""
    if not webhook_url:
        logger.warning("⚠️ Alert triggered, but no Discord Webhook URL found in config.")
        return

    # Visual cues for the message
    icon = "🤖" if status == "AI_PROPOSED" else "🔗"
    color_bar = 15158332 # Red color code for Discord embed
    
    # Discord Embed Structure (Looks professional on mobile)
    payload = {
        "username": "The Watchman 👁️",
        "embeds": [
            {
                "title": f"🚨 Contagion Alert: {ticker}",
                "description": f"**{ticker}** is exposed to high-risk asset **{neighbor}**.",
                "color": color_bar,
                "fields": [
                    {"name": "🔥 Threat Level", "value": f"`{score:.4f}` (High Volatility)", "inline": True},
                    {"name": "🔗 Link Type", "value": f"{icon} {status}", "inline": True},
                    {"name": "📝 Mechanism", "value": mechanism, "inline": False}
                ],
                "footer": {"text": "Knowledge Graph Surveillance System"}
            }
        ]
    }

    try:
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"❌ Failed to send Discord alert: {e}")

def load_live_risk_scores(csv_path="live_risk_scores.csv"):
    """Loads the CSV and returns a dictionary: {'TICKER': 0.95, ...}"""
    if not os.path.exists(csv_path):
        logger.warning(f"⚠️ Risk CSV not found at {csv_path}. Assuming no external risks.")
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        return pd.Series(df.Risk_Score.values, index=df.Ticker).to_dict()
    except Exception as e:
        logger.error(f"❌ Failed to load risk scores: {e}")
        return {}

def run_risk_scan():
    # 1. Load Config
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        watchlist = config.get("target_tickers", [])
        webhook_url = config.get("DISCORD_WEBHOOK_URL")
        
        if not watchlist:
            logger.warning("⚠️ 'target_tickers' list is empty. Nothing to scan.")
            return

    except Exception as e:
        logger.error(f"❌ Config error: {e}")
        return

    # 2. Connect to Database
    try:
        db = DatabaseManager(config)
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return
    
    # 3. Load Risk Data
    csv_path = os.path.join(script_dir, "live_risk_scores.csv")
    risk_map = load_live_risk_scores(csv_path)
    
    logger.info("---------------------------------------------------")
    logger.info(f"🛡️  STARTING WATCHMAN SCAN FOR: {len(watchlist)} TICKERS")
    logger.info("---------------------------------------------------")
    
    alerts_triggered = 0

    for ticker in watchlist:
        ticker = ticker.upper().strip()
        
        # Get immediate neighbors (1-hop)
        graph = db.get_neighborhood_graph(ticker)
        
        if graph.number_of_nodes() == 0:
            continue
            
        # Check every neighbor
        for neighbor in graph.nodes():
            if neighbor == ticker: continue 
            
            neighbor_risk = risk_map.get(neighbor, 0.0)
            
            if neighbor_risk >= RISK_THRESHOLD:
                # 🚨 HIGH RISK NEIGHBOR FOUND
                
                edge_data = graph.get_edge_data(ticker, neighbor)
                if not edge_data:
                    edge_data = graph.get_edge_data(neighbor, ticker)
                
                if edge_data:
                    alerts_triggered += 1
                    mechanism = edge_data.get('mechanism', 'Direct Correlation')
                    status = edge_data.get('verification_status', 'VERIFIED')
                    
                    # 1. Log to File
                    print(f"\n🚨 [RISK ALERT] {ticker} is exposed to {neighbor}!")
                    print(f"   🔥 Risk Score: {neighbor_risk:.4f}")
                    
                    # 2. Send to Discord
                    send_discord_alert(webhook_url, ticker, neighbor, neighbor_risk, mechanism, status)
                    
                    # Sleep briefly to avoid Discord rate limits if multiple alerts fire at once
                    time.sleep(1) 

    if alerts_triggered == 0:
        logger.info("✅ Scan Complete. No immediate contagion risks found.")
    else:
        logger.info(f"\n⚠️ Scan Complete. {alerts_triggered} Alerts Sent to Discord.")

    db.close()

if __name__ == "__main__":
    run_risk_scan()