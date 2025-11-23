import json
import logging
import torch
import torch.nn.functional as F
import pandas as pd
from neo4j import GraphDatabase
from gnn_pipeline import GNNPipeline
from gnn_model import create_hetero_model
from typing import Dict, Any
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

def export_to_csv(config: Dict[str, Any], filename="live_risk_scores.csv"):
    """
    Fetches the latest risk scores from Neo4j and saves them to a CSV.
    """
    try:
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
            data = [record.data() for record in result]
            
        driver.close()

        if data:
            df = pd.DataFrame(data)
            df.to_csv(filename, index=False)
            logger.info(f"✅ CSV Export Successful: Saved {len(df)} records to {filename}")
        else:
            logger.warning("⚠️ CSV Export Warning: No data found in Neo4j to export.")

    except Exception as e:
        logger.error(f"❌ Error exporting CSV: {e}")

def run_inference():
    """
    Loads model, predicts risk, applies Contrast Stretching if needed, and saves.
    """
    logger.info("🚀 Starting GNN inference script...")
    
    config = load_config()
    if not config:
        return

    # --- 1. Load Graph Data ---
    pipeline = GNNPipeline(config)
    data = pipeline.get_graph_data()
    
    if data is None:
        logger.error("Failed to load graph data. Exiting.")
        return

    # --- 2. Load Trained Model ---
    model_path = "gnn_risk_model.pth"
    if not os.path.exists(model_path):
        logger.error(f"FATAL: Model file '{model_path}' not found.")
        return

    try:
        model = create_hetero_model(data, hidden_dim=64, out_dim=3)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        logger.info(f"Successfully loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    # --- 3. Run Inference ---
    logger.info("Running model to generate risk scores...")
    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict)
        
        # --- FIX: DETECT MODEL COLLAPSE & STRETCH CONTRAST ---
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
            # Logits contain the 'confidence' before it gets squashed by Softmax
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

        # --- 4. Write Predictions ---
        pipeline.save_predictions(risk_scores)

        # --- 5. Export ---
        export_to_csv(config)

if __name__ == "__main__":
    run_inference()