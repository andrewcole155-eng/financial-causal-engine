# ==============================================================================
# --- MACRO-ECONOMIC INGESTION ENGINE ---
# ==============================================================================
# Injects Indices, Commodities, and Yields into the Graph as "Super Nodes".
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
# If Sector is "All", it affects the whole market.
MACRO_ASSETS = {
    "^TNX":  ("10-Year Treasury Yield", "Technology", "NEGATIVELY_CORRELATED_WITH"), # Rates up, Tech down
    "CL=F":  ("Crude Oil", "Energy", "POSITIVELY_CORRELATED_WITH"),                 # Oil up, Energy up
    "GC=F":  ("Gold", "Materials", "POSITIVELY_CORRELATED_WITH"),
    "DX-Y.NYB": ("US Dollar Index", "All", "AFFECTS_DEMAND_FOR"),                   # Strong dollar affects exports
    "^VIX":  ("Volatility Index", "All", "INCREASES_RISK_FOR")                      # Fear gauge
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

    # 1. Create/Merge Macro Nodes
    for ticker, (name, target_sector, rel_type) in MACRO_ASSETS.items():
        
        # Upsert the Node
        # We give it the 'Company' label so market_data.py picks it up for risk scoring!
        # We also give it a 'Macro' label for filtering.
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
        
        # 2. Create Sector-Wide Relationships
        # This links the Macro Node to EVERY company in that sector.
        if target_sector == "All":
            # Link to major indices or just leave as global node
            pass 
        else:
            logger.info(f"   -> Linking {ticker} to all '{target_sector}' companies...")
            query_link = f"""
            MATCH (m:Company {{ticker: $ticker}})
            MATCH (c:Company) 
            WHERE c.sector CONTAINS $sector AND c.ticker <> $ticker
            MERGE (m)-[r:{rel_type}]->(c)
            SET r.weight = 0.7,
                r.mechanism = 'Macro-Economic Sector Correlation',
                r.verification_status = 'VERIFIED'
            """
            db.execute_write(query_link, ticker=ticker, sector=target_sector)

    logger.info("✅ Macro Ingestion Complete.")
    db.close()

if __name__ == "__main__":
    ingest_macro_nodes()