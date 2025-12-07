# ==============================================================================
# --- MARKET DATA ENGINE (Real-World History Version) ---
# ==============================================================================
import yfinance as yf
import pandas as pd
import numpy as np
import logging
import os
import json
from database_manager import DatabaseManager

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("MarketData")

# --- CONFIGURATION ---
HISTORY_PERIOD = "6mo"  # General lookback for risk scores
DETAILED_HISTORY = "3mo" # Lookback for GNN training sequences (Need ~60 days)
RISK_LOOKBACK = 20      # Calculate volatility over the last 20 trading days

def load_config():
    """Loads configuration from the same directory."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Error loading config: {e}")
        return {}

def load_tickers_from_db(config):
    """Fetches all tickers currently in your Neo4j Graph."""
    try:
        db = DatabaseManager(config)
        query = "MATCH (n:Company) RETURN DISTINCT n.ticker as ticker"
        results = db.execute_read(query)
        db.close()
        
        tickers = [r['ticker'] for r in results if r['ticker']]
        logger.info(f"✅ Found {len(tickers)} tickers in the Knowledge Graph.")
        return tickers
    except Exception as e:
        logger.error(f"❌ DB Error: {e}")
        return []

def calculate_technical_risk(df):
    """
    Calculates a 'Risk Score' (0-1) based on technical indicators.
    """
    # 1. Calculate Daily Returns
    df['Returns'] = df['Close'].pct_change()
    
    # 2. Annualized Volatility (Standard Deviation * Sqrt(252))
    current_volatility = df['Returns'].tail(RISK_LOOKBACK).std() * np.sqrt(252)
    
    # 3. RSI (Relative Strength Index) - 14 Day
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    current_rsi = 100 - (100 / (1 + rs)).iloc[-1]
    
    return current_volatility, current_rsi

# ==============================================================================
# --- NEW FUNCTION: FETCH FULL HISTORY FOR GNN ---
# ==============================================================================
def fetch_historical_data(config=None, period=DETAILED_HISTORY):
    """
    Downloads detailed daily OHLCV data for ALL tickers.
    Returns a pandas DataFrame indexed by Date with MultiIndex columns (Ticker, Price).
    Used by 'generate_real_history.py'.
    """
    if config is None: config = load_config()
    raw_tickers = load_tickers_from_db(config)
    
    if not raw_tickers: return None

    # Clean Tickers
    yahoo_tickers = []
    ticker_map = {}
    for t in raw_tickers:
        if any(prefix in t for prefix in ["X:", "C:", "I:"]): continue # Skip non-stocks
        yahoo_t = t.replace('.', '-')
        yahoo_tickers.append(yahoo_t)
        ticker_map[yahoo_t] = t

    logger.info(f"⬇️  Fetching {period} of daily history for {len(yahoo_tickers)} assets...")
    
    try:
        # Download all at once
        data = yf.download(yahoo_tickers, period=period, group_by='ticker', threads=True)
        
        # Validation
        if data.empty:
            logger.error("❌ YFinance returned empty data.")
            return None
            
        logger.info(f"✅ Downloaded data shape: {data.shape}")
        return data, ticker_map

    except Exception as e:
        logger.error(f"❌ History Download failed: {e}")
        return None, None

# ==============================================================================
# --- EXISTING FUNCTION: UPDATE CURRENT RISK (Unchanged logic) ---
# ==============================================================================
def update_risk_scores():
    # 1. Load Config
    config = load_config()
    if not config: return

    # 2. Get Tickers
    raw_tickers = load_tickers_from_db(config)
    if not raw_tickers: return

    # --- SYNTAX CLEANER & MAPPING ---
    yahoo_tickers = []
    ticker_map = {} 
    
    skipped_count = 0
    for t in raw_tickers:
        if any(prefix in t for prefix in ["X:", "C:", "I:"]):
            skipped_count += 1
            continue 
            
        yahoo_t = t.replace('.', '-')
        yahoo_tickers.append(yahoo_t)
        ticker_map[yahoo_t] = t 
        
    logger.info(f"⬇️  Downloading market data for {len(yahoo_tickers)} companies (Skipped {skipped_count} non-stocks)...")
    
    # 3. Batch Download
    try:
        data = yf.download(yahoo_tickers, period=HISTORY_PERIOD, group_by='ticker', threads=True)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return

    # 4. Calculate Risk
    logger.info("🧮 Calculating Real-World Volatility & RSI...")
    risk_results = []
    
    for yahoo_t in yahoo_tickers:
        try:
            if len(yahoo_tickers) == 1:
                df = data
            else:
                if yahoo_t not in data.columns.levels[0]: continue
                df = data[yahoo_t]
            
            if df.empty or 'Close' not in df.columns: continue
                
            df = df.dropna()
            if len(df) < 30: continue 
            
            vol, rsi = calculate_technical_risk(df)
            
            if np.isnan(vol): vol = 0.0
            if np.isnan(rsi): rsi = 50.0 

            original_db_ticker = ticker_map.get(yahoo_t, yahoo_t)

            risk_results.append({
                "Ticker": original_db_ticker, 
                "Volatility": vol,
                "RSI": rsi
            })
        except Exception:
            continue

    results_df = pd.DataFrame(risk_results)
    if results_df.empty:
        logger.error("No valid risk data calculated.")
        return

    # 5. Normalize Scores (0 to 1)
    min_vol = results_df['Volatility'].min()
    max_vol = results_df['Volatility'].max()
    
    if max_vol == min_vol:
        results_df['Vol_Score'] = 0.5
    else:
        results_df['Vol_Score'] = (results_df['Volatility'] - min_vol) / (max_vol - min_vol)
    
    results_df['RSI_Risk'] = abs(results_df['RSI'] - 50) / 50.0
    results_df['Risk_Score'] = (results_df['Vol_Score'] * 0.7) + (results_df['RSI_Risk'] * 0.3)
    results_df['Risk_Score'] = results_df['Risk_Score'].clip(0, 1)

    # 6. WRITE TO NEO4J
    logger.info("💾 Writing risk scores directly to Neo4j...")
    
    batch_data = results_df[['Ticker', 'Risk_Score']].rename(
        columns={'Ticker': 'ticker', 'Risk_Score': 'score'}
    ).to_dict('records')
    
    db = DatabaseManager(config)
    
    query = """
    UNWIND $batch AS row
    MATCH (c:Company {ticker: row.ticker})
    SET c.raw_risk_score = row.score,
        c.risk_last_updated = datetime()
    """
    
    try:
        db.execute_write(query, batch=batch_data)
        logger.info(f"✅ Successfully updated {len(batch_data)} nodes in Neo4j.")
    except Exception as e:
        logger.error(f"❌ Failed to write to Neo4j: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    update_risk_scores()