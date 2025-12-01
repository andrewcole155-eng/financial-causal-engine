# ==============================================================================
# --- MARKET DATA ENGINE (Real Math Update + Syntax Fix) ---
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
HISTORY_PERIOD = "6mo"  # Look back 6 months to judge volatility
RISK_LOOKBACK = 20      # Calculate volatility over the last 20 trading days

def load_tickers_from_db():
    """Fetches all tickers currently in your Neo4j Graph."""
    try:
        # Load Config relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        db = DatabaseManager(config)
        # Quick query to get all unique tickers
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
    Math:
    1. Volatility (StdDev of returns): High Volatility = High Risk.
    2. RSI (Relative Strength): Very Low RSI (<30) = Crash Mode = High Risk.
    """
    # 1. Calculate Daily Returns
    df['Returns'] = df['Close'].pct_change()
    
    # 2. Annualized Volatility (Standard Deviation * Sqrt(252))
    # We take the rolling volatility of the last RISK_LOOKBACK days
    current_volatility = df['Returns'].tail(RISK_LOOKBACK).std() * np.sqrt(252)
    
    # 3. RSI (Relative Strength Index) - 14 Day
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    current_rsi = 100 - (100 / (1 + rs)).iloc[-1]
    
    return current_volatility, current_rsi

def update_risk_scores():
    raw_tickers = load_tickers_from_db()
    if not raw_tickers:
        return

    # --- SYNTAX CLEANER & MAPPING ---
    # We need to map 'BRK.B' (DB) -> 'BRK-B' (Yahoo) -> 'BRK.B' (CSV)
    # This ensures yfinance finds the data, but the Graph can still read the result.
    yahoo_tickers = []
    ticker_map = {} # Key: Yahoo Ticker, Value: DB Ticker (e.g., {'BRK-B': 'BRK.B'})
    
    skipped_count = 0
    
    for t in raw_tickers:
        # 1. Skip non-stock assets to clean up logs (Crypto X:, Forex C:, Indices I:)
        if any(prefix in t for prefix in ["X:", "C:", "I:"]):
            skipped_count += 1
            continue 
            
        # 2. Fix Dot to Dash (BRK.B -> BRK-B) for Yahoo
        yahoo_t = t.replace('.', '-')
        
        yahoo_tickers.append(yahoo_t)
        ticker_map[yahoo_t] = t # Remember the original name
        
    logger.info(f"⬇️  Downloading market data for {len(yahoo_tickers)} companies (Skipped {skipped_count} non-stocks)...")
    
    # Batch download
    try:
        data = yf.download(yahoo_tickers, period=HISTORY_PERIOD, group_by='ticker', threads=True)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return

    risk_results = []

    logger.info("🧮 Calculating Real-World Volatility & RSI...")
    
    for yahoo_t in yahoo_tickers:
        try:
            # Handle Single Ticker vs Multi-Ticker dataframe structure
            if len(yahoo_tickers) == 1:
                df = data
            else:
                # If ticker data is missing, this key won't exist
                if yahoo_t not in data.columns.levels[0]: 
                    continue
                df = data[yahoo_t]
            
            # Skip empty or malformed data
            if df.empty or 'Close' not in df.columns:
                continue
                
            # Drop NaN to perform math
            df = df.dropna()
            if len(df) < 30: continue # Not enough history
            
            vol, rsi = calculate_technical_risk(df)
            
            if np.isnan(vol): vol = 0.0
            if np.isnan(rsi): rsi = 50.0 # Neutral if math fails

            # IMPORTANT: Map back to the DB Ticker Name!
            original_db_ticker = ticker_map.get(yahoo_t, yahoo_t)

            risk_results.append({
                "Ticker": original_db_ticker, 
                "Volatility": vol,
                "RSI": rsi
            })
            
        except Exception:
            continue

    # --- THE "REAL MATH" SCORING ---
    results_df = pd.DataFrame(risk_results)
    
    if results_df.empty:
        logger.error("No valid risk data calculated.")
        return

    # Normalize Volatility (0 to 1) relative to the rest of the market
    min_vol = results_df['Volatility'].min()
    max_vol = results_df['Volatility'].max()
    
    # Avoid divide by zero if market is flat (unlikely)
    if max_vol == min_vol:
        results_df['Vol_Score'] = 0.5
    else:
        results_df['Vol_Score'] = (results_df['Volatility'] - min_vol) / (max_vol - min_vol)
    
    # RSI Risk: 
    # RSI < 30 is PANIC (High Risk)
    # RSI > 70 is GREED (High Risk of correction)
    # RSI 50 is Calm (Low Risk)
    results_df['RSI_Risk'] = abs(results_df['RSI'] - 50) / 50.0
    
    # FINAL WEIGHTED SCORE
    # 70% Weight on Volatility (Structural Instability)
    # 30% Weight on RSI (Current Price Action)
    results_df['Risk_Score'] = (results_df['Vol_Score'] * 0.7) + (results_df['RSI_Risk'] * 0.3)
    
    # Clip to ensure 0-1
    results_df['Risk_Score'] = results_df['Risk_Score'].clip(0, 1)

    # Save
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_risk_scores.csv")
    results_df[['Ticker', 'Risk_Score']].to_csv(csv_path, index=False)
    
    logger.info("------------------------------------------------")
    logger.info(f"✅ UPDATED {len(results_df)} RISK SCORES")
    
    # Sort for logging (Safeguard against empty result)
    if not results_df.empty:
        riskiest = results_df.sort_values('Risk_Score', ascending=False).iloc[0]['Ticker']
        safest = results_df.sort_values('Risk_Score', ascending=True).iloc[0]['Ticker']
        logger.info(f"🔥 Riskiest Asset: {riskiest}")
        logger.info(f"🛡️  Safest Asset:   {safest}")
    logger.info("------------------------------------------------")

if __name__ == "__main__":
    update_risk_scores()