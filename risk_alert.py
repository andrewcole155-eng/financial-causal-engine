# ==============================================================================
# --- RISK ALERT SYSTEM ("The Watchman") ---
# ==============================================================================
# Handles immediate risk alerts, kill switches, and contagion scanning.
# Can be run standalone or imported by sentiment_pulse.py for instant triggers.
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
PANIC_THRESHOLD = -0.80 # Sentiment scores below this trigger immediate panic

# 2. Status File Path (Shared with Trading Bot)
STATUS_FILE = "trading_status.json"

def load_config():
    """Helper to load config from the same directory."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        with open(config_path, 'r') as f: return json.load(f)
    except: return {}

def update_trading_status(status, reason=None):
    """
    Writes the DEFCON level to a JSON file for the trading bot.
    status: "GREEN" or "RED"
    """
    data = {
        "status": status,  # "GREEN" or "RED"
        "last_updated": datetime.now().isoformat(),
        "reason": reason,
        "risk_threshold": RISK_THRESHOLD
    }
    
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

# --- NEW: IMMEDIATE PANIC TRIGGER ---
def trigger_panic_mode(ticker, headline, score, url, config=None):
    """
    Callable from external scripts (like sentiment_pulse.py) to immediately
    halt trading and fire a Discord alert.
    """
    if config is None:
        config = load_config()

    webhook_url = config.get("DISCORD_WEBHOOK_URL")
    
    # 1. Activate Kill Switch locally
    reason = f"PANIC: {ticker} Sentiment {score} ({headline[:30]}...)"
    update_trading_status("RED", reason)
    
    # 2. Fire Discord Alert
    if webhook_url:
        payload = {
            "username": "The Watchman 👁️",
            "embeds": [{
                "title": f"🚨 MARKET PANIC: {ticker}",
                "description": f"**CRITICAL NEGATIVE NEWS DETECTED**\n\n**Headline:** {headline}\n**Analysis:** Extreme bearish sentiment identified by AI.",
                "color": 16711680, # Bright Red (0xFF0000)
                "fields": [
                    {"name": "📉 Sentiment Score", "value": f"`{score}`", "inline": True},
                    {"name": "📰 Source", "value": f"[Read Story]({url})", "inline": True},
                    {"name": "⛔ ACTION TAKEN", "value": "TRADING HALTED (KILL SWITCH ENGAGED)", "inline": False}
                ],
                "footer": {"text": f"Triggered at {datetime.now().strftime('%H:%M:%S')}"}
            }]
        }
        try:
            requests.post(webhook_url, json=payload)
            logger.info(f"🚨 Panic Alert Sent to Discord for {ticker}")
        except Exception as e:
            logger.error(f"Failed to send Panic Alert: {e}")

# --- EXISTING: CONTAGION ALERT (GRAPH SCAN) ---
def send_discord_alert(webhook_url, ticker, neighbor, score, mechanism, status):
    """Sends a contagion risk alert to Discord."""
    if not webhook_url:
        return

    icon = "🤖" if status == "AI_PROPOSED" else "🔗"
    color_bar = 15158332 # Orange/Red
    
    payload = {
        "username": "The Watchman 👁️",
        "embeds": [
            {
                "title": f"⚠️ Contagion Warning: {ticker}",
                "description": f"**{ticker}** is exposed to high-risk asset **{neighbor}**.",
                "color": color_bar,
                "fields": [
                    {"name": "🔥 Risk Exposure", "value": f"`{score:.4f}`", "inline": True},
                    {"name": "🔗 Link Type", "value": f"{icon} {status}", "inline": True},
                    {"name": "📝 Mechanism", "value": mechanism, "inline": False},
                    {"name": "🛡️ Recommendation", "value": "Reduce position sizing or engage Kill Switch.", "inline": False}
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
    """
    Main routine: Scans the database for contagion risks (neighbors of watchlist tokens).
    """
    config = load_config()
    watchlist = config.get("target_tickers", [])
    webhook_url = config.get("DISCORD_WEBHOOK_URL")
    
    if not watchlist:
        logger.warning("⚠️ Watchlist empty.")
        return

    # Connect to Database
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
                    kill_switch_reason = msg
                    
                    print(f"\n🚨 [RISK ALERT] {msg}")
                    send_discord_alert(webhook_url, ticker, neighbor, neighbor_risk, mechanism, status)
                    time.sleep(1) 

    # --- KILL SWITCH LOGIC (SCAN BASED) ---
    if alerts_triggered > 0:
        update_trading_status("RED", kill_switch_reason)
        logger.info(f"\n⚠️ Scan Complete. {alerts_triggered} Alerts. Kill Switch ENGAGED.")
    else:
        # Only reset to GREEN if we are running the scan and finding nothing
        # We might optionally check if a "Manual" or "Panic" lock is in place before resetting
        update_trading_status("GREEN", "System Nominal")
        logger.info("✅ Scan Complete. System Nominal.")

    db.close()

if __name__ == "__main__":
    run_risk_scan()