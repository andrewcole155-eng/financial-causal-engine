# ==============================================================================
# --- SECTOR ENRICHMENT SCRIPT (Targeted Fix) ---
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
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        with open(config_path, 'r') as f: return json.load(f)
    except: return {}

def clean_ticker_for_yahoo(ticker):
    """
    Converts DB format to Yahoo format.
    Example: 'BRK.B' -> 'BRK-B', 'OTCMKTS:APO' -> 'APO'
    """
    # Remove Exchange Prefixes (e.g., "OTCMKTS:APO" -> "APO")
    if ":" in ticker:
        ticker = ticker.split(":")[-1]
    
    # Fix Berkshire and others with dots
    ticker = ticker.replace('.', '-')
    
    # Specific Manual Fixes
    if ticker == 'BRK': return 'BRK-B'
    
    return ticker

def enrich_sectors():
    config = load_config()
    if not config: return
    db = DatabaseManager(config)
    
    # 1. Target ONLY the problem nodes
    logger.info("🔍 finding companies with 'Discovered' or 'Unknown' sectors...")
    query = """
    MATCH (c:Company) 
    WHERE c.sector IN ['Discovered', 'Unknown', 'null'] OR c.sector IS NULL
    RETURN c.ticker as ticker
    """
    results = db.execute_read(query)
    tickers = [r['ticker'] for r in results]
    
    if not tickers:
        logger.info("✅ No missing sectors found! Your database is clean.")
        return

    logger.info(f"📉 Found {len(tickers)} companies to fix. Starting batch processing...")
    
    updated_count = 0
    
    # Process in batches of 20
    batch_size = 20
    for i in range(0, len(tickers), batch_size):
        batch_db_tickers = tickers[i:i+batch_size]
        
        # Create a map of YahooTicker -> DBTicker so we can save it back correctly
        yahoo_map = {clean_ticker_for_yahoo(t): t for t in batch_db_tickers}
        yahoo_list = list(yahoo_map.keys())
        
        try:
            # Fetch info for the batch
            tickers_obj = yf.Tickers(" ".join(yahoo_list))
            
            for y_sym in yahoo_list:
                try:
                    # Check if we got data
                    info = tickers_obj.tickers[y_sym].info
                    
                    # Yahoo returns 'sector' and 'industry'
                    sector = info.get('sector')
                    industry = info.get('industry')
                    
                    if sector:
                        # Retrieve original DB ticker to ensure we update the right node
                        original_ticker = yahoo_map[y_sym]
                        
                        update_query = """
                        MATCH (c:Company {ticker: $ticker})
                        SET c.sector = $sector,
                            c.industry = $industry
                        """
                        db.execute_write(update_query, ticker=original_ticker, sector=sector, industry=industry)
                        updated_count += 1
                        print(f"   ✅ Fixed {original_ticker}: {sector}")
                    else:
                        print(f"   ⚠️ No data for {y_sym}")
                        
                except Exception:
                    pass 
                    
        except Exception as e:
            logger.error(f"Batch failed: {e}")
            
        # Be polite to Yahoo API
        time.sleep(1)
            
    logger.info(f"🎉 Finished. Repaired {updated_count} companies.")
    db.close()

if __name__ == "__main__":
    enrich_sectors()