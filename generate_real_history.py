import torch
import os
import logging
import json
import pandas as pd
import numpy as np
from datetime import timedelta
from gnn_pipeline import GNNPipeline
from market_data import fetch_historical_data
from feature_engineering import enrich_data  # <--- IMPORTED YOUR NEW LOGIC

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

def generate_real_history():
    logger.info("🚀 Starting Real-World History Generation (Volatility & Shock Mode)...")

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
    # We fetch 3 months to allow the rolling windows (7d, 30d) in feature_engineering to warm up
    market_df, ticker_map = fetch_historical_data(config, period="3mo")
    
    if market_df is None or market_df.empty:
        logger.error("❌ Failed to download market history.")
        return

    id_to_ticker = {v: k for k, v in pipeline.ticker_to_id.items()}
    num_nodes = base_data['Company'].num_nodes
    dates = market_df.index
    logger.info(f"📅 Processing {len(dates)} trading days...")

    generated_count = 0
    
    # 3. Pre-Calculate Features using SHARED LOGIC
    logger.info("⚡ Applying feature_engineering.py to all assets...")
    processed_data = {}
    
    for ticker_yahoo in market_df.columns.levels[0]:
        try:
            df_ticker = market_df[ticker_yahoo].copy()
            
            # Data Cleaning for Feature Engineering
            if df_ticker['Close'].isnull().all(): continue
            
            # 1. Standardize Columns (enrich_data expects lowercase 'close')
            df_ticker = df_ticker.rename(columns={'Close': 'close', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Volume': 'volume'})
            
            # 2. Forward fill missing prices to prevent NaN gaps
            df_ticker['close'] = df_ticker['close'].ffill()
            
            # 3. Handle Missing Sentiment
            # If your market_data fetcher doesn't provide 'sentiment_score', we mock it
            # so the script runs. (Ideally, fetch real sentiment here).
            if 'sentiment_score' not in df_ticker.columns:
                df_ticker['sentiment_score'] = 0.0 
            
            # 4. APPLY THE NEW LOGIC
            # This adds: volatility_shock, sentiment_shock, trend_deviation, etc.
            df_ticker = enrich_data(df_ticker)
            
            processed_data[ticker_yahoo] = df_ticker
            
        except Exception as e:
            # logger.warning(f"Skipping {ticker_yahoo}: {e}")
            continue

    # 4. Generate Snapshots
    # We skip the first 30 days to ensure the rolling windows have valid data
    start_index = 30 
    if len(dates) < start_index + 1:
        logger.error("Not enough data points for rolling windows.")
        return

    for i in range(start_index, len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i+1]
        date_str = current_date.strftime("%Y-%m-%d")
        
        daily_snapshot = base_data.clone()
        
        new_features = []
        targets_return = []
        targets_risk = []
        
        for node_idx in range(num_nodes):
            ticker_db = id_to_ticker.get(node_idx)
            # Normalize Ticker format (Neo4j 'BRK.B' -> Yahoo 'BRK-B')
            ticker_yahoo = ticker_db.replace('.', '-') if ticker_db else None
            
            # Defaults (Neutral values)
            feat_vol_shock = 0.0
            feat_sent_shock = 0.0
            feat_trend = 0.0
            feat_vol_mag = 0.0
            
            ret_next = 0.0
            risk_class = 1 # Default to Neutral Risk
            
            # --- EXTRACT FEATURES ---
            if ticker_yahoo and ticker_yahoo in processed_data:
                try:
                    df = processed_data[ticker_yahoo]
                    
                    if current_date in df.index and next_date in df.index:
                        today = df.loc[current_date]
                        nxt = df.loc[next_date]
                        
                        # Extract the exact features created by enrich_data
                        feat_vol_shock = float(today.get('volatility_shock', 0))
                        feat_sent_shock = float(today.get('sentiment_shock', 0))
                        feat_trend = float(today.get('trend_deviation', 0))
                        feat_vol_mag = float(today.get('volatility_7d', 0))
                        
                        close = float(today['close'])
                        
                        # Calculate Target (Next Day Return)
                        if close > 0:
                            ret_next = (float(nxt['close']) - close) / close
                        
                        # Risk Label Logic (Crash Detection)
                        # We use -2% as the threshold for "High Risk" (Class 2)
                        if ret_next < -0.02: 
                            risk_class = 2  # Crash / High Risk
                        elif ret_next > 0.01: 
                            risk_class = 0  # Bullish / Low Risk
                        else: 
                            risk_class = 1  # Neutral

                except Exception: pass

            # Static Features from Neo4j (Preserve these context features)
            if base_data['Company'].x is not None:
                base_feats = base_data['Company'].x[node_idx]
                cap_norm = float(base_feats[0])  # Market Cap
                is_macro_val = float(base_feats[1]) # Macro Flag
            else:
                cap_norm = 0.0; is_macro_val = 0.0
            
            # *** NEW FEATURE VECTOR (Size 6) ***
            # Matches what we defined in train.py logic
            # [MarketCap, IsMacro, VolatilityShock, SentimentShock, TrendDeviation, VolatilityMagnitude]
            new_features.append([
                cap_norm, 
                is_macro_val, 
                feat_vol_shock, 
                feat_sent_shock, 
                feat_trend, 
                feat_vol_mag
            ])
            
            targets_return.append([ret_next])
            targets_risk.append(risk_class)

        # Save to Snapshot Object
        daily_snapshot['Company'].x = torch.tensor(new_features, dtype=torch.float)
        daily_snapshot['Company'].y = torch.tensor(targets_return, dtype=torch.float)
        daily_snapshot['Company'].y_class = torch.tensor(targets_risk, dtype=torch.long)
        
        # Ensure directory exists
        os.makedirs(pipeline.snapshot_dir, exist_ok=True)
        save_path = os.path.join(pipeline.snapshot_dir, f"graph_snapshot_{date_str}.pt")
        
        torch.save(daily_snapshot, save_path)
        generated_count += 1
        
        if generated_count % 10 == 0:
            logger.info(f"   Saved {date_str} ({generated_count} snapshots generated)")

    logger.info(f"✅ Generated {generated_count} synchronized snapshots ready for training.")

if __name__ == "__main__":
    generate_real_history()