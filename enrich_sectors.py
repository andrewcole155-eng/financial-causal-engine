# ==============================================================================
# --- SECTOR ENRICHMENT SCRIPT ---
# ==============================================================================
import yfinance as yf
import json
import logging
import os
import time
from database_manager import DatabaseManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("SectorRepair")

def load_config():
    try:
        with open("config.json", 'r') as f: return json.load(f)
    except: return {}

def enrich_sectors():
    config = load_config()
    if not config: return
    db = DatabaseManager(config)
    
    # 1. Get all companies with missing or 'Unknown' sectors
    query = "MATCH (c:Company) RETURN c.ticker as ticker"
    results = db.execute_read(query)
    tickers = [r['ticker'] for r in results]
    
    logger.info(f"🔍 Scanning {len(tickers)} companies for missing Sector data...")
    
    updated_count = 0
    
    # Process in batches to be polite to Yahoo
    batch_size = 20
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        
        # Filter out non-stock tickers (Crypto/Forex)
        clean_batch = [t.replace('.', '-') for t in batch if not any(x in t for x in [':', '='])]
        
        if not clean_batch: continue

        try:
            # Fetch info for the batch
            tickers_obj = yf.Tickers(" ".join(clean_batch))
            
            for t_symbol in clean_batch:
                try:
                    info = tickers_obj.tickers[t_symbol].info
                    sector = info.get('sector')
                    industry = info.get('industry')
                    
                    if sector:
                        # Map back to DB symbol (swap - back to .)
                        db_symbol = t_symbol.replace('-', '.')
                        
                        # Update Neo4j
                        update_query = """
                        MATCH (c:Company {ticker: $ticker})
                        SET c.sector = $sector,
                            c.industry = $industry
                        """
                        db.execute_write(update_query, ticker=db_symbol, sector=sector, industry=industry)
                        updated_count += 1
                        print(f"   ✅ {db_symbol}: {sector}")
                        
                except Exception:
                    pass # Ticker might be delisted or data unavailable
                    
        except Exception as e:
            logger.error(f"Batch failed: {e}")
            
    logger.info(f"🎉 Finished. Updated sectors for {updated_count} companies.")
    db.close()

if __name__ == "__main__":
    enrich_sectors()