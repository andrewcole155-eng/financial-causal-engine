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
    This CSV is what the Streamlit app will read from GitHub.
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
            # Save to the current directory (which is mounted to the host)
            df.to_csv(filename, index=False)
            logger.info(f"✅ CSV Export Successful: Saved {len(df)} records to {filename}")
        else:
            logger.warning("⚠️ CSV Export Warning: No data found in Neo4j to export.")

    except Exception as e:
        logger.error(f"❌ Error exporting CSV: {e}")

def run_inference():
    """
    Loads the trained GNN model, predicts risk scores for all companies, 
    writes them back to Neo4j, and exports a CSV for the frontend.
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

        # --- 4. Write Safe Predictions to Neo4j ---
        # We use the pipeline's built-in method to ensure tickers match indices perfectly.
        pipeline.save_predictions(risk_scores)

        # --- 5. Export for Streamlit/GitHub ---
        # This creates the file that your git_update.sh script will push
        export_to_csv(config)

if __name__ == "__main__":
    run_inference()