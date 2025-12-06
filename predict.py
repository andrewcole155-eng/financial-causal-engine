import json
import logging
import torch
import torch.nn.functional as F
import pandas as pd
import os
import sys
from gnn_pipeline import GNNPipeline
from gnn_model import create_hetero_model
from torch_geometric.data import HeteroData

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_local_config():
    """Loads config.json from the same directory as the script."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def inject_inference_features(data: HeteroData) -> HeteroData:
    """
    Must match train.py logic!
    Injects random noise features to match the model's input dimension (16).
    """
    logger.info("--- Injecting Unique Random Features for Inference ---")
    # Use a fixed seed to ensure consistent features if run multiple times
    torch.manual_seed(42)
    
    num_companies = data['Company'].num_nodes
    embedding_dim = 16
    
    # Generate random features [N, 16]
    unique_features = torch.randn(num_companies, embedding_dim)
    data['Company'].x = unique_features
    logger.info(f"Assigned random feature vectors of shape {unique_features.shape}")
    return data

def run_inference():
    logger.info("🚀 Starting GNN inference script...")

    # 1. Load Configuration
    config = {}
    neo4j_uri = os.environ.get('NEO4J_URI')
    neo4j_user = os.environ.get('NEO4J_USER')
    neo4j_password = os.environ.get('NEO4J_PASSWORD')

    if neo4j_uri and neo4j_user and neo4j_password:
        config = {"neo4j": {"uri": neo4j_uri, "user": neo4j_user, "password": neo4j_password}}
    else:
        loaded = load_local_config()
        config = loaded if "neo4j" in loaded else {"neo4j": loaded}

    if not config or "neo4j" not in config:
        logger.critical("❌ No valid configuration found.")
        sys.exit(1)

    # 2. Initialize Pipeline ONCE (Critical for ID mapping)
    pipeline = GNNPipeline(config)
    data = pipeline.get_graph_data()

    if data is None:
        logger.error("Failed to fetch graph data.")
        return

    # 3. Inject Features (Must match training)
    data = inject_inference_features(data)

    # 4. Load Model
    # Hidden dim must match train.py (32)
    model = create_hetero_model(data, hidden_dim=32, out_dim=3)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "gnn_risk_model.pth")
    
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        logger.info(f"✅ Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return

    # 5. Run Prediction
    logger.info("🔮 Running model to generate risk scores...")
    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict)
        probs = F.softmax(logits, dim=1)
        
        # Risk Score Formula: 0.5 * Medium + 1.0 * High
        risk_scores = (probs[:, 1] * 0.5) + (probs[:, 2] * 1.0)
        
        std_dev = risk_scores.std().item()
        logger.info(f"Model Score Std Dev: {std_dev:.6f} (Higher is better)")

    # 6. Save to Neo4j AND Get Results
    # This uses the SAME pipeline instance, so ticker_to_id is valid
    updates = pipeline.save_predictions(risk_scores)

    # 7. Export to CSV (Robust)
    if updates:
        csv_path = os.path.join(script_dir, "live_risk_scores.csv")
        try:
            df = pd.DataFrame(updates)
            df.to_csv(csv_path, index=False)
            logger.info(f"✅ Exported CSV to: {csv_path}")
            logger.info(f"Top 3 Riskiest Assets:\n{df.sort_values('score', ascending=False).head(3)}")
        except Exception as e:
            logger.error(f"❌ Error exporting CSV: {e}")
    else:
        logger.warning("⚠️ No updates returned from pipeline (List is empty).")

if __name__ == "__main__":
    run_inference()