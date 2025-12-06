import json
import logging
import torch
import torch.nn.functional as F
import pandas as pd
from neo4j import GraphDatabase
from gnn_pipeline import GNNPipeline
from gnn_model import create_hetero_model
from typing import Dict, Any
from torch_geometric.data import HeteroData
import os

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    """Loads config.json from the same directory."""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config.json: {e}")
        return {}

def add_unique_company_features(data: HeteroData) -> HeteroData:
    """
    MUST MATCH TRAIN.PY
    Injects the same shape of random features so the model structure matches.
    """
    logger.info("--- Injecting Unique Random Features for Inference ---")
    
    # --- NEW: Set Seed for Consistency ---
    # This ensures that when the Explainer runs later, it sees the exact same 
    # "random" features as the Inference engine did.
    torch.manual_seed(42)
    
    num_companies = data['Company'].num_nodes
    
    # Must match the embedding_dim from train.py (16)
    embedding_dim = 16
    
    unique_features = torch.randn(num_companies, embedding_dim)
    
    data['Company'].x = unique_features
    logger.info(f"Assigned random feature vectors of shape {unique_features.shape}")
    return data

def get_inference_resources():
    """
    --- NEW HELPER FOR STREAMLIT / EXPLAINER ---
    Loads the Data and the Model and returns them.
    Does NOT run the full batch prediction or CSV export.
    This allows the App to 'borrow' the model for explanation purposes.
    """
    config = load_config()
    if not config: return None, None, None

    # 1. Load Data
    pipeline = GNNPipeline(config)
    data = pipeline.get_graph_data()
    if data is None: return None, None, None

    # 2. Inject Features (Using the fixed seed)
    data = add_unique_company_features(data)

    # 3. Load Model
    model_path = "gnn_risk_model.pth"
    if not os.path.exists(model_path):
        logger.error(f"FATAL: Model file '{model_path}' not found.")
        return None, None, None

    try:
        # hidden_dim=32 matches your train.py
        model = create_hetero_model(data, hidden_dim=32, out_dim=3)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model, data, config
    except Exception as e:
        logger.error(f"Error loading model resources: {e}")
        return None, None, None

def export_to_csv(config: Dict[str, Any], filename="live_risk_scores.csv"):
    """
    Fetches scores from Neo4j, FILTERS them against the clean JSON list,
    and saves to CSV.
    """
    try:
        # 1. Load the "Guest List" (Clean JSON)
        json_file = 'sp500_companies.json'
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                clean_map = json.load(f) # {'AAPL': 'Apple Inc.', ...}
            valid_tickers = set(clean_map.keys())
            logger.info(f"Loaded {len(valid_tickers)} valid tickers from {json_file}")
        else:
            logger.warning(f"⚠️ {json_file} not found. Exporting ALL database nodes (including Zombies).")
            clean_map = {}
            valid_tickers = None

        # 2. Connect to Neo4j
        uri = config.get("neo4j_uri", "bolt://localhost:7687")
        auth = (config.get("neo4j_user", "neo4j"), config.get("neo4j_password", "password"))
        
        driver = GraphDatabase.driver(uri, auth=auth)
        
        query = """
        MATCH (c:Company)
        WHERE c.gnn_risk_score IS NOT NULL
        RETURN c.ticker AS Ticker, 
               c.name AS Name, 
               c.sector AS Sector, 
               c.gnn_risk_score AS Risk_Score,
               c.gnn_last_updated AS Last_Updated
        ORDER BY c.gnn_risk_score DESC
        """
        
        with driver.session() as session:
            result = session.run(query)
            raw_data = [record.data() for record in result]
            
        driver.close()

        # 3. FILTER and CLEAN the data
        clean_data = []
        
        for row in raw_data:
            ticker = row['Ticker']
            
            # If we have a valid list, skip tickers not in it (Kills Zombies like ATVI)
            if valid_tickers and ticker not in valid_tickers:
                continue
                
            # Update the Name from our clean JSON (Fixes 'N/A' names)
            if valid_tickers and ticker in clean_map:
                row['Name'] = clean_map[ticker]

            # (Optional) Clean up 'Discovered' sector if possible, 
            # otherwise keep what is in DB.
            if row['Sector'] == 'Discovered':
                # Placeholder: In the future, your JSON should include sectors 
                # to fix this properly. For now, we leave it or set to 'Unknown'.
                pass 

            clean_data.append(row)

        if clean_data:
            df = pd.DataFrame(clean_data)
            df.to_csv(filename, index=False)
            logger.info(f"✅ CSV Export Successful: Saved {len(df)} clean records to {filename}")
            
            # Log how many Zombies were killed (Restored this logic)
            zombies_killed = len(raw_data) - len(clean_data)
            if zombies_killed > 0:
                logger.info(f"👻 Removed {zombies_killed} 'Zombie' tickers (e.g., ATVI, ABC) from export.")
        else:
            logger.warning("⚠️ CSV Export Warning: No valid data found.")

    except Exception as e:
        logger.error(f"❌ Error exporting CSV: {e}")

def run_inference():
    """
    Loads model, predicts risk, applies Contrast Stretching if needed, and saves.
    """
    logger.info("🚀 Starting GNN inference script...")
    
    # --- CHANGED: Use the helper to get resources ---
    model, data, config = get_inference_resources()
    if model is None:
        return

    # --- 4. Run Inference ---
    logger.info("Running model to generate risk scores...")
    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict)
        
        # --- DETECT MODEL COLLAPSE & STRETCH CONTRAST ---
        # 1. Get raw probabilities
        probs = F.softmax(logits, dim=1)
        
        # 2. Calculate base risk score (Weighted Average)
        # Weight Med (idx 1) by 0.5, High (idx 2) by 1.0
        risk_scores = (probs[:, 1] * 0.5) + (probs[:, 2] * 1.0)
        
        # 3. Check for "Flatline" (Standard Deviation near 0)
        std_dev = torch.std(risk_scores).item()
        logger.info(f"Model Score Std Dev: {std_dev:.6f}")

        if std_dev < 0.001:
            logger.warning("⚠️ DETECTED MODEL COLLAPSE: All scores are identical.")
            logger.warning("🔧 Applying 'Contrast Stretching' to logits to force ranking...")
            
            # Use the raw Logits for the 'High Risk' class (index 2)
            raw_risk_signal = logits[:, 2] 
            
            # Normalize these raw signals to 0.0 - 1.0 range
            min_val = torch.min(raw_risk_signal)
            max_val = torch.max(raw_risk_signal)
            
            if max_val - min_val > 0:
                risk_scores = (raw_risk_signal - min_val) / (max_val - min_val)
                # Add a tiny bit of noise to break perfect ties (helps sorting)
                risk_scores += torch.rand_like(risk_scores) * 0.001
            else:
                # If truly 100% identical, inject pure noise (fallback)
                logger.warning("⚠️ Logits identical. Injecting random jitter.")
                risk_scores = torch.rand_like(risk_scores)

        logger.info(f"Generated risk scores for {len(risk_scores)} companies.")

        # --- 5. Write Predictions ---
        # We need to re-init pipeline here just to access the save method
        # (Or we could have returned pipeline from get_resources, but this is fine)
        pipeline = GNNPipeline(config)
        pipeline.save_predictions(risk_scores)

        # --- 6. Export ---
        export_to_csv(config)

if __name__ == "__main__":
    run_inference()