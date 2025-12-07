import torch
import os
import json
import logging
from datetime import datetime, timedelta
from gnn_pipeline import GNNPipeline
from torch_geometric.data import HeteroData

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
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning(f"Error reading config.json: {e}")
        return {}

def generate_fake_history():
    # 1. Initialize Pipeline
    logger.info("🔌 Connecting to Neo4j to fetch BASE graph...")
    config = load_local_config()
    
    if "neo4j" not in config: 
        config = {"neo4j": config}
    
    pipeline = GNNPipeline(config)
    
    # 2. Get the "Anchor" Graph
    try:
        base_data = pipeline.get_graph_data()
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return
    
    if base_data is None:
        logger.error("❌ Could not fetch base graph.")
        return

    # --- DEBUGGING OUTPUT ---
    logger.info(f"🔍 Graph Metadata: {base_data.metadata()}")
    logger.info(f"🔍 Node Types found: {base_data.node_types}")
    
    # Robust Check for Company Nodes
    has_companies = False
    num_companies = 0
    
    if 'Company' in base_data.node_types:
        # Check explicit num_nodes or infer from feature matrix x
        if hasattr(base_data['Company'], 'num_nodes') and base_data['Company'].num_nodes > 0:
            num_companies = base_data['Company'].num_nodes
            has_companies = True
        elif hasattr(base_data['Company'], 'x') and base_data['Company'].x is not None:
            num_companies = base_data['Company'].x.shape[0]
            has_companies = True
            
    if not has_companies:
        logger.error(f"❌ No Company nodes detected in object. Found types: {base_data.node_types}")
        return

    logger.info(f"✅ Validation Passed. Found {num_companies} companies.")

    # Ensure output directory exists
    output_dir = pipeline.snapshot_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. Generate 30 Days of "Fake History"
    logger.info("⏳ Generating 30 days of synthetic history...")
    
    today = datetime.now()
    
    for i in range(30, 0, -1):
        past_date = today - timedelta(days=i)
        date_str = past_date.strftime("%Y-%m-%d")
        
        # Clone the base data
        fake_snapshot = base_data.clone()
        
        # --- ADD NOISE ---
        if fake_snapshot['Company'].x is not None:
            noise = torch.randn_like(fake_snapshot['Company'].x) * 0.01 
            fake_snapshot['Company'].x += noise
            
        # Fake Returns (-3% to +3%)
        fake_returns = torch.randn(num_companies, 1) * 0.03
        fake_snapshot['Company'].y = fake_returns
        
        # Fake Risk Labels (0, 1, 2)
        fake_risks = torch.randint(0, 3, (num_companies,), dtype=torch.long)
        fake_snapshot['Company'].y_class = fake_risks

        # --- SAVE ---
        filename = f"graph_snapshot_{date_str}.pt"
        save_path = os.path.join(output_dir, filename)
        
        torch.save(fake_snapshot, save_path)
        logger.info(f"   -> Saved snapshot: {filename}")

    logger.info("✅ Fake history generation complete.")
    logger.info(f"   Now run: python train.py")

if __name__ == "__main__":
    generate_fake_history()