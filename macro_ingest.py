# ==============================================================================
# --- MACRO-ECONOMIC INGESTION ENGINE (GLOBAL STATE UPDATE) ---
# ==============================================================================
# Injects Indices as "Super Nodes" that influence the entire graph.
# ==============================================================================

import yfinance as yf
import json
import logging
import os
from database_manager import DatabaseManager

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("MacroIngest")

# --- MACRO DEFINITIONS ---
# Ticker: (Name, Affected_Sector, Relationship_Type)
# UPDATED: We set these to "All" to create the Global State Super-Nodes.
MACRO_ASSETS = {
    "^TNX":      ("10-Year Treasury Yield", "All", "GLOBAL_RATE_SENSITIVITY"),
    "CL=F":      ("Crude Oil", "All", "GLOBAL_COMMODITY_SENSITIVITY"),
    "GC=F":      ("Gold", "All", "GLOBAL_RISK_OFF_SENTIMENT"),
    "DX-Y.NYB":  ("US Dollar Index", "All", "GLOBAL_FX_IMPACT"),
    "^VIX":      ("Volatility Index", "All", "GLOBAL_MARKET_FEAR")
}

def load_config():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        with open(config_path, 'r') as f: return json.load(f)
    except: return {}

def ingest_macro_nodes():
    config = load_config()
    if not config: return
    
    db = DatabaseManager(config)
    
    logger.info(f"🌍 Ingesting {len(MACRO_ASSETS)} Macro-Economic Assets...")

    # Collect all macro tickers so we don't link macros to each other
    all_macro_tickers = list(MACRO_ASSETS.keys())

    for ticker, (name, target_sector, rel_type) in MACRO_ASSETS.items():
        
        # 1. Create the Node (ensure it is labeled correctly)
        query_node = """
        MERGE (m:Company {ticker: $ticker})
        SET m.name = $name,
            m.sector = 'Macro',
            m.is_macro = true
        WITH m
        CALL apoc.create.addLabels(m, ['Macro']) YIELD node
        RETURN node
        """
        db.execute_write(query_node, ticker=ticker, name=name)
        
        # 2. Logic Branching
        if target_sector == "All":
            # --- NEW GLOBAL STATE ARCHITECTURE ---
            # Connects this Macro Node to EVERY single Company node (Asset).
            # We filter out other Macro nodes to prevent cycles.
            logger.info(f"   -> ⚡ Wiring {ticker} as a GLOBAL SUPER-NODE...")
            
            query_global = f"""
            MATCH (m:Company {{ticker: $ticker}})
            MATCH (c:Company) 
            WHERE NOT c.ticker IN $macro_list AND c.ticker <> $ticker
            MERGE (m)-[r:{rel_type}]->(c)
            SET r.weight = 0.9, 
                r.mechanism = 'Global Macro Influence',
                r.verification_status = 'VERIFIED'
            """
            db.execute_write(query_global, ticker=ticker, macro_list=all_macro_tickers)

        else:
            # --- LEGACY SECTOR SPECIFIC LOGIC ---
            # Keeps the ability to link specific assets (like just Copper -> Materials)
            logger.info(f"   -> Linking {ticker} to specific sector '{target_sector}'...")
            
            query_sector = f"""
            MATCH (m:Company {{ticker: $ticker}})
            MATCH (c:Company) 
            WHERE c.sector CONTAINS $sector AND c.ticker <> $ticker
            MERGE (m)-[r:{rel_type}]->(c)
            SET r.weight = 0.75,
                r.mechanism = 'Sector Correlation'
            """
            db.execute_write(query_sector, ticker=ticker, sector=target_sector)

    logger.info("✅ Macro Ingestion Complete.")
    db.close()

if __name__ == "__main__":
    ingest_macro_nodes()