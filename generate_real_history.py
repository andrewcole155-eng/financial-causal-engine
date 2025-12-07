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

# --- HELPER FUNCTION ADDED LOCALLY ---
def load_local_config():
    """Loads config.json from the same directory as the script."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning(f"Error reading config.json: {e}")
        return {}

def generate_real_history():
    logger.info("🚀 Starting Real-World History Generation...")

    # 1. Initialize Pipeline & Fetch Structure
    config = load_local_config()
    if "neo4j" not in config: config = {"neo4j": config}
    
    pipeline = GNNPipeline(config)
    
    # Get the "Skeleton" graph (Nodes & Edges from Neo4j)
    try:
        base_data = pipeline.get_graph_data()
    except Exception as e:
        logger.error(f"Error connecting to Neo4j: {e}")
        return
    
    if base_data is None:
        logger.error("❌ Could not fetch base graph.")
        return

    # 2. Download Real History (The "Time Machine")
    # We ask for 3 months to ensure we have enough data for a 60-day window
    market_df, ticker_map = fetch_historical_data(config, period="3mo")
    
    if market_df is None or market_df.empty:
        logger.error("❌ Failed to download market history. Exiting.")
        return

    # Map Neo4j ID -> Ticker
    # We need to know which node index corresponds to which ticker string
    # pipeline.ticker_to_id is populated by get_graph_data()
    id_to_ticker = {v: k for k, v in pipeline.ticker_to_id.items()}
    
    if 'Company' not in base_data.node_types:
        logger.error("❌ Graph missing Company nodes.")
        return

    num_nodes = base_data['Company'].num_nodes

    # 3. Iterate Through Time
    # market_df index is the Date. We loop through every date.
    dates = market_df.index
    logger.info(f"📅 Processing {len(dates)} trading days...")

    generated_count = 0
    
    # We need at least 2 days to calculate a "Target" (Next Day Return)
    # So we stop 1 day before the end.
    for i in range(len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i+1]
        
        date_str = current_date.strftime("%Y-%m-%d")
        
        # Clone the skeleton
        daily_snapshot = base_data.clone()
        
        # Prepare Feature Matrices
        new_features = []
        targets_return = []
        targets_risk = []
        
        # We need to handle the case where YFinance returns a MultiIndex (Ticker, Field) 
        # or just (Field) if one ticker.
        # market_df usually: Columns = MultiIndex([('AAPL', 'Close'), ...])
        
        for node_idx in range(num_nodes):
            ticker_db = id_to_ticker.get(node_idx)
            
            # Find the Yahoo Ticker (convert '.' to '-')
            ticker_yahoo = ticker_db.replace('.', '-') if ticker_db else None
            
            # Default values if data missing
            close = 100.0
            volume = 0.0
            ret_next = 0.0
            risk_class = 0 # Default Low Risk
            
            # --- EXTRACT REAL DATA ---
            # Check if ticker exists in the downloaded data columns
            has_data = False
            if ticker_yahoo:
                 # Check if the ticker is in the top level of columns
                 if isinstance(market_df.columns, pd.MultiIndex):
                     if ticker_yahoo in market_df.columns.levels[0]:
                         has_data = True
                 elif ticker_yahoo in market_df.columns:
                     has_data = True

            if has_data:
                try:
                    # Access data for this specific ticker
                    if isinstance(market_df.columns, pd.MultiIndex):
                        ticker_data = market_df[ticker_yahoo]
                    else:
                        ticker_data = market_df # Single ticker case (rare here)

                    # Get row for this date
                    if current_date in ticker_data.index and next_date in ticker_data.index:
                        daily_data = ticker_data.loc[current_date]
                        next_data_row = ticker_data.loc[next_date]
                        
                        # 1. Features (Today)
                        val_close = float(daily_data['Close'])
                        # Handle NaN
                        if np.isnan(val_close): val_close = 100.0
                        close = val_close
                        
                        # 2. Target (Tomorrow's Return)
                        val_next = float(next_data_row['Close'])
                        if not np.isnan(val_next) and close > 0:
                            ret_next = (val_next - close) / close
                        
                        # 3. Risk Class
                        # 0 = Up, 1 = Small Drop, 2 = Big Drop (<-2%)
                        if ret_next >= 0:
                            risk_class = 0
                        elif ret_next > -0.02:
                            risk_class = 1
                        else:
                            risk_class = 2
                except Exception:
                    pass

            # Re-use static features from base graph (Market Cap, Is_Macro) if available
            # base_data.x shape: [NumNodes, 3] -> [Cap, Macro, Price]
            # We preserve Cap and Macro, update Price
            if base_data['Company'].x is not None:
                base_feats = base_data['Company'].x[node_idx]
                cap_norm = float(base_feats[0])
                is_macro_val = float(base_feats[1])
            else:
                cap_norm = 0.0
                is_macro_val = 0.0
            
            new_features.append([cap_norm, is_macro_val, np.log1p(close)])
            targets_return.append([ret_next])
            targets_risk.append(risk_class)

        # Assign to Snapshot
        daily_snapshot['Company'].x = torch.tensor(new_features, dtype=torch.float)
        daily_snapshot['Company'].y = torch.tensor(targets_return, dtype=torch.float)
        daily_snapshot['Company'].y_class = torch.tensor(targets_risk, dtype=torch.long)
        
        # Save to Disk
        save_path = os.path.join(pipeline.snapshot_dir, f"graph_snapshot_{date_str}.pt")
        torch.save(daily_snapshot, save_path)
        generated_count += 1
        
        if generated_count % 10 == 0:
            logger.info(f"   Saved {date_str} (Snapshot {generated_count}/{len(dates)-1})")

    logger.info(f"✅ Successfully generated {generated_count} real-world snapshots.")
    logger.info("   Now run 'python train.py' to train on this REAL data!")

if __name__ == "__main__":
    generate_real_history()