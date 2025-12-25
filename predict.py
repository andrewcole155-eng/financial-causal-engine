import json
import logging
import torch
import torch.nn.functional as F
import pandas as pd
import os
import sys
from typing import Tuple, List, Dict
from gnn_pipeline import GNNPipeline
from gnn_model import create_hetero_model
from torch_geometric.data import HeteroData

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- HELPER: CONFIG LOADER ---
def load_local_config():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, "config.json"), 'r') as f: return json.load(f)
    except: return {}

# ==============================================================================
# --- CORE INFERENCE LOGIC ---
# ==============================================================================
def get_inference_resources():
    """
    Loads Pipeline, Model, and the SEQUENCE of data needed for the temporal model.
    """
    logger.info("🔌 App requested inference resources...")

    # 1. Load Config
    neo_uri = os.environ.get('NEO4J_URI')
    neo_user = os.environ.get('NEO4J_USER')
    neo_pass = os.environ.get('NEO4J_PASSWORD')

    config = {}
    if neo_uri and neo_user and neo_pass:
        config = {"neo4j": {"uri": neo_uri, "user": neo_user, "password": neo_pass}}
    else:
        loaded = load_local_config()
        config = loaded if "neo4j" in loaded else {"neo4j": loaded}
    
    if "neo4j" not in config: raise ValueError("Configuration missing.")

    # 2. Initialize Pipeline
    pipeline = GNNPipeline(config)

    # 3. Load Historical Sequence (Last 5 Days)
    # We need at least 5 days to make a prediction
    WINDOW_SIZE = 5
    snapshots = pipeline.load_historical_sequence(days=WINDOW_SIZE)
    
    if len(snapshots) < WINDOW_SIZE:
        raise ValueError(f"❌ Not enough history! Found {len(snapshots)}, need {WINDOW_SIZE}. Run generate_fake_history.py?")

    # 4. Load Model
    # We use the metadata from the last snapshot to initialize the model structure
    last_snapshot = snapshots[-1]
    model = create_hetero_model(last_snapshot, hidden_dim=64) # Must match train.py hidden_dim
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "gnn_multitask_model.pth")
    
    if os.path.exists(model_path):
        # weights_only=False required for complex model structures in newer PyTorch
        try:
            state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        except:
             # Fallback if saved without weights_only support
             state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
             
        model.load_state_dict(state_dict)
        model.eval()
        logger.info(f"✅ Loaded Multi-Task Model from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

    return pipeline, model, snapshots

# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================
def run_inference():
    logger.info("🚀 Starting TEMPORAL FORECAST script...")

    try:
        pipeline, model, snapshots = get_inference_resources()
    except Exception as e:
        logger.critical(f"❌ Initialization failed: {e}")
        sys.exit(1)

    # 5. Prepare Input Sequence
    # The model expects a LIST of graphs.
    # Since we are running inference for ONE window (the most recent one),
    # we just pass the list directly.
    # IMPORTANT: If your model expects a batch dimension (e.g., list of lists), handle it.
    # Our `TemporalHeteroGNN` handles a single list [t-4, t-3, ..., t] fine.
    
    logger.info("🔮 Running model to forecast Tomorrow...")
    
    with torch.no_grad():
        # returns: (price_forecast, risk_logits)
        pred_price, pred_risk_logits, _ = model(snapshots)
        
        # Process Outputs
        # A. Price Forecast (Raw Float)
        price_forecasts = pred_price.squeeze().numpy()
        
        # B. Risk Score (Probabilities)
        probs = F.softmax(pred_risk_logits, dim=1)
        # Risk Score Formula: 0.5 * Medium + 1.0 * High
        risk_scores = (probs[:, 1] * 0.5) + (probs[:, 2] * 1.0)
        risk_scores = risk_scores.numpy()

    # 6. Map to Tickers & Save
    # We use the ticker map from the pipeline (populated during load)
    # Note: pipeline.ticker_to_id might be stale if we only loaded from disk.
    # We should grab it from the snapshot if possible, but the snapshot doesn't store the map easily.
    # Robust approach: The pipeline.ticker_to_id is populated when get_graph_data() is called.
    # Since we loaded from disk, we might need to refresh the map or trust the order is same.
    # For now, let's assume Ticker Order is sorted alphabetically in Neo4j query (ORDER BY c.ticker), 
    # so it should be consistent across all snapshots.
    
    # We re-fetch the map from Neo4j just to be safe and get the ticker names
    logger.info("Fetching fresh ticker map from Neo4j to ensure alignment...")
    _ = pipeline.get_graph_data() # This populates pipeline.ticker_to_id
    id_to_ticker = {v: k for k, v in pipeline.ticker_to_id.items()}

    results = {}
    
    # Safety check
    if len(price_forecasts) != len(id_to_ticker):
        logger.warning(f"⚠️ Shape mismatch! Model pred {len(price_forecasts)} vs DB Tickers {len(id_to_ticker)}")
        # We proceed carefully, mapping by index
    
    for idx, (price, risk) in enumerate(zip(price_forecasts, risk_scores)):
        if idx in id_to_ticker:
            ticker = id_to_ticker[idx]
            results[ticker] = {
                "price_forecast": float(price),
                "risk_score": float(risk)
            }

    # 7. Save to Neo4j
    pipeline.save_predictions(results)

    # 8. Export to CSV (Dual Signal)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "live_forecasts.csv")
    
    # Convert dict to DataFrame
    df_data = []
    for ticker, vals in results.items():
        df_data.append({
            "ticker": ticker,
            "forecast_return": vals['price_forecast'],
            "risk_score": vals['risk_score']
        })
    
    df = pd.DataFrame(df_data)
    
    if not df.empty:
        df.to_csv(csv_path, index=False)
        logger.info(f"✅ Exported Forecasts to: {csv_path}")
        
        # Show exciting results
        print("\n--- 🚀 TOP BULLISH FORECASTS ---")
        print(df.sort_values("forecast_return", ascending=False).head(3))
        
        print("\n--- ⚠️ HIGHEST RISK ALERTS ---")
        print(df.sort_values("risk_score", ascending=False).head(3))
    else:
        logger.warning("No results to export.")

if __name__ == "__main__":
    run_inference()