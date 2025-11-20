import json
import logging
import torch
import torch.nn.functional as F
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

def run_inference():
    """
    Loads the trained GNN model, predicts risk scores for all companies, 
    and writes them back to Neo4j safely.
    """
    logger.info("🚀 Starting GNN inference script...")
    
    config = load_config()
    if not config:
        return

    # --- 1. Load Graph Data & Pipeline ---
    # The pipeline now handles the safe mapping of Index -> Ticker
    pipeline = GNNPipeline(config)
    data = pipeline.get_graph_data()
    
    if data is None:
        logger.error("Failed to load graph data. Exiting.")
        return
    logger.info("Graph data loaded successfully.")

    # --- 2. Load Trained Model ---
    model_path = "gnn_risk_model.pth"
    
    if not os.path.exists(model_path):
        logger.error(f"FATAL: Model file '{model_path}' not found. Please run train.py first.")
        return

    try:
        # Initialize model architecture (must match training)
        model = create_hetero_model(data, hidden_dim=64, out_dim=3)
        
        # Load the trained weights
        model.load_state_dict(torch.load(model_path))
        model.eval() # Set to evaluation mode
        logger.info(f"Successfully loaded trained model from {model_path}")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return

    # --- 3. Run Inference ---
    logger.info("Running model to generate risk scores...")
    with torch.no_grad():
        # Forward pass
        logits = model(data.x_dict, data.edge_index_dict)
        
        # Convert Logits -> Probabilities (Softmax)
        # shape: [num_companies, 3] -> [Prob(Low), Prob(Med), Prob(High)]
        probs = F.softmax(logits, dim=1)
        
        # Calculate Continuous Risk Score (0.0 to 1.0)
        # This ensures the frontend gets a gradient (Blue -> Red), not just 3 chunk colors.
        # Formula: 50% weight on Medium, 100% weight on High.
        risk_scores = (probs[:, 1] * 0.5) + (probs[:, 2] * 1.0)
        
        logger.info(f"Generated risk scores for {len(risk_scores)} companies.")

        # --- 4. Write Safe Predictions ---
        # We use the pipeline's built-in method to ensure tickers match indices perfectly.
        pipeline.save_predictions(risk_scores)

if __name__ == "__main__":
    run_inference()