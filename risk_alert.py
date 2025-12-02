# ==============================================================================
# --- RISK ALERT SYSTEM ("The Watchman") ---
# ==============================================================================
import json
import logging
import os
import requests
import time
from datetime import datetime
from database_manager import DatabaseManager

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("RiskAlert")

# 1. Define Thresholds
RISK_THRESHOLD = 0.85

# 2. Status File Path (Shared with Trading Bot)
STATUS_FILE = "trading_status.json"

def update_trading_status(status, reason=None):
    """Writes the DEFCON level to a JSON file for the trading bot."""
    data = {
        "status": status,  # "GREEN" or "RED"
        "last_updated": datetime.now().isoformat(),
        "reason": reason,
        "risk_threshold": RISK_THRESHOLD
    }
    
    # Write safely (atomic write pattern could be used, but this is sufficient for now)
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, STATUS_FILE)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
            
        if status == "RED":
            logger.warning(f"⛔ KILL SWITCH ACTIVATED: {reason}")
        else:
            logger.info("✅ Trading Status: GREEN")
            
    except Exception as e:
        logger.error(f"❌ Failed to update trading status file: {e}")

def send_discord_alert(webhook_url, ticker, neighbor, score, mechanism, status):
    """Sends a formatted alert to Discord."""
    if not webhook_url:
        return

    icon = "🤖" if status == "AI_PROPOSED" else "🔗"
    color_bar = 15158332 # Red
    
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
                    {"name": "📝 Mechanism", "value": mechanism, "inline": False},
                    {"name": "⛔ ACTION TAKEN", "value": "Trading Bot HALTED via Kill Switch.", "inline": False}
                ],
                "footer": {"text": "Knowledge Graph Surveillance System"}
            }
        ]
    }

    try:
        requests.post(webhook_url, json=payload)
    except Exception as e:
        logger.error(f"❌ Failed to send Discord alert: {e}")

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
            logger.warning("⚠️ Watchlist empty.")
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
    
    logger.info("---------------------------------------------------")
    logger.info(f"🛡️  STARTING WATCHMAN SCAN FOR: {len(watchlist)} TICKERS")
    logger.info("---------------------------------------------------")
    
    alerts_triggered = 0
    kill_switch_reason = ""

    for ticker in watchlist:
        ticker = ticker.upper().strip()
        
        # Get immediate neighbors
        graph = db.get_neighborhood_graph(ticker)
        
        if graph.number_of_nodes() == 0:
            continue
            
        for neighbor in graph.nodes():
            if neighbor == ticker: continue 
            
            # Get Risk Score directly from DB property
            neighbor_risk = graph.nodes[neighbor].get('raw_risk_score', 0.0)
            
            if neighbor_risk >= RISK_THRESHOLD:
                # 🚨 HIGH RISK FOUND
                edge_data = graph.get_edge_data(ticker, neighbor)
                if not edge_data:
                    edge_data = graph.get_edge_data(neighbor, ticker)
                
                if edge_data:
                    alerts_triggered += 1
                    mechanism = edge_data.get('mechanism', 'Direct Correlation')
                    status = edge_data.get('verification_status', 'VERIFIED')
                    
                    msg = f"{ticker} exposed to {neighbor} (Risk: {neighbor_risk:.2f})"
                    kill_switch_reason = msg # Save the first reason for the status file
                    
                    print(f"\n🚨 [RISK ALERT] {msg}")
                    send_discord_alert(webhook_url, ticker, neighbor, neighbor_risk, mechanism, status)
                    time.sleep(1) 

    # --- KILL SWITCH LOGIC ---
    if alerts_triggered > 0:
        update_trading_status("RED", kill_switch_reason)
        logger.info(f"\n⚠️ Scan Complete. {alerts_triggered} Alerts. Kill Switch ENGAGED.")
    else:
        update_trading_status("GREEN", "System Nominal")
        logger.info("✅ Scan Complete. System Nominal.")

    db.close()

if __name__ == "__main__":
    run_risk_scan()