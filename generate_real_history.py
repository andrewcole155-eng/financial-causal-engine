import torch
import os
import logging
import json
import pandas as pd
import numpy as np
from datetime import timedelta
from gnn_pipeline import GNNPipeline
from market_data import fetch_historical_data

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# --- HELPER FUNCTION ---
def load_local_config():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError: return {}
    except Exception as e: return {}

def calculate_technical_indicators(df):
    """
    Calculates RSI, Momentum, and Volatility using pure Pandas.
    No external 'ta' library needed.
    """
    # 1. Momentum (Returns)
    df['R_5'] = df['Close'].pct_change(5).fillna(0)   # 1 Week Trend
    df['R_20'] = df['Close'].pct_change(20).fillna(0) # 1 Month Trend
    
    # 2. Volatility (Standard Deviation of Price)
    # Log-normalized to keep scale consistent
    rolling_std = df['Close'].rolling(window=20).std().fillna(0)
    df['Vol_20'] = np.log1p(rolling_std)

    # 3. Relative Strength Index (RSI) - 14 Day
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Fill NaN (RSI is NaN for first 14 days)
    df['RSI'] = df['RSI'].fillna(50) # Default to neutral
    
    # Normalize RSI to 0-1 range for the Neural Network
    df['RSI_Norm'] = df['RSI'] / 100.0
    
    return df

def generate_real_history():
    logger.info("🚀 Starting Real-World History Generation (Quant Mode)...")

    # 1. Initialize
    config = load_local_config()
    if "neo4j" not in config: config = {"neo4j": config}
    pipeline = GNNPipeline(config)
    
    try:
        base_data = pipeline.get_graph_data()
    except Exception as e:
        logger.error(f"Error connecting to Neo4j: {e}")
        return
    
    if base_data is None:
        logger.error("❌ Could not fetch base graph.")
        return

    # 2. Download Data (3 Months)
    market_df, ticker_map = fetch_historical_data(config, period="3mo")
    
    if market_df is None or market_df.empty:
        logger.error("❌ Failed to download market history.")
        return

    id_to_ticker = {v: k for k, v in pipeline.ticker_to_id.items()}
    num_nodes = base_data['Company'].num_nodes
    dates = market_df.index
    logger.info(f"📅 Processing {len(dates)} trading days...")

    generated_count = 0
    
    # 3. Pre-Calculate Indicators for ALL Tickers
    # This is much faster than doing it inside the loop
    logger.info("🧮 Pre-calculating Technical Indicators...")
    processed_data = {}
    
    for ticker_yahoo in market_df.columns.levels[0]:
        try:
            df_ticker = market_df[ticker_yahoo].copy()
            # Handle missing data
            if df_ticker['Close'].isnull().all(): continue
            
            # Forward fill missing prices
            df_ticker['Close'] = df_ticker['Close'].ffill()
            
            # Calc Indicators
            df_ticker = calculate_technical_indicators(df_ticker)
            processed_data[ticker_yahoo] = df_ticker
        except Exception:
            continue

    # 4. Generate Snapshots
    for i in range(len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i+1]
        date_str = current_date.strftime("%Y-%m-%d")
        
        daily_snapshot = base_data.clone()
        
        new_features = []
        targets_return = []
        targets_risk = []
        
        for node_idx in range(num_nodes):
            ticker_db = id_to_ticker.get(node_idx)
            ticker_yahoo = ticker_db.replace('.', '-') if ticker_db else None
            
            # Defaults
            feat_mom5 = 0.0; feat_mom20 = 0.0; feat_vol = 0.0; feat_rsi = 0.5
            ret_next = 0.0; risk_class = 0; close = 100.0

            # --- EXTRACT FEATURES ---
            if ticker_yahoo and ticker_yahoo in processed_data:
                try:
                    df = processed_data[ticker_yahoo]
                    
                    if current_date in df.index and next_date in df.index:
                        today = df.loc[current_date]
                        nxt = df.loc[next_date]
                        
                        # Features
                        feat_mom5 = float(today['R_5'])
                        feat_mom20 = float(today['R_20'])
                        feat_vol = float(today['Vol_20'])
                        feat_rsi = float(today['RSI_Norm'])
                        close = float(today['Close'])
                        
                        # Target
                        if close > 0:
                            ret_next = (float(nxt['Close']) - close) / close
                        
                        # Risk Class Logic
                        if ret_next >= 0: risk_class = 0      # Bullish
                        elif ret_next > -0.02: risk_class = 1 # Neutral/Small Dip
                        else: risk_class = 2                  # Crash (<-2%)

                except Exception: pass

            # Static Features from Neo4j (Market Cap, Macro Flag)
            if base_data['Company'].x is not None:
                base_feats = base_data['Company'].x[node_idx]
                cap_norm = float(base_feats[0])
                is_macro_val = float(base_feats[1])
            else:
                cap_norm = 0.0; is_macro_val = 0.0
            
            # *** NEW FEATURE VECTOR (Size 6) ***
            # [LogCap, IsMacro, Momentum5, Momentum20, Volatility, RSI]
            new_features.append([cap_norm, is_macro_val, feat_mom5, feat_mom20, feat_vol, feat_rsi])
            targets_return.append([ret_next])
            targets_risk.append(risk_class)

        # Save
        daily_snapshot['Company'].x = torch.tensor(new_features, dtype=torch.float)
        daily_snapshot['Company'].y = torch.tensor(targets_return, dtype=torch.float)
        daily_snapshot['Company'].y_class = torch.tensor(targets_risk, dtype=torch.long)
        
        save_path = os.path.join(pipeline.snapshot_dir, f"graph_snapshot_{date_str}.pt")
        torch.save(daily_snapshot, save_path)
        generated_count += 1
        
        if generated_count % 10 == 0:
            logger.info(f"   Saved {date_str} ({generated_count}/{len(dates)-1})")

    logger.info(f"✅ Generated {generated_count} SMART snapshots with Technical Indicators.")

if __name__ == "__main__":
    generate_real_history()