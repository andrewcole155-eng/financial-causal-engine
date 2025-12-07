import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from gnn_model import create_hetero_model
from gnn_pipeline import GNNPipeline
from torch_geometric.data import HeteroData
from typing import List
import os
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIG LOADER ---
def load_local_config():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, "config.json"), 'r') as f: return json.load(f)
    except: return {}

# ---------------------------------------------------------
# 1. MULTI-TASK DATASET
# ---------------------------------------------------------
class MultiTaskDataset(Dataset):
    def __init__(self, snapshots: List[HeteroData], window_size: int = 5):
        self.snapshots = snapshots
        self.window_size = window_size
        if len(self.snapshots) <= self.window_size:
            logger.warning(f"⚠️ Not enough snapshots! Has {len(snapshots)}, needs > {window_size}")

    def __len__(self):
        return max(0, len(self.snapshots) - self.window_size)

    def __getitem__(self, idx):
        # Input: Sequence
        window_graphs = self.snapshots[idx : idx + self.window_size]
        
        # Target: The NEXT day
        target_graph = self.snapshots[idx + self.window_size]
        
        # Safety check for missing Company nodes
        if 'Company' not in target_graph.node_types:
             # Return dummy data if graph is broken to prevent crash
             logger.warning(f"Snapshot {idx} missing Company nodes. returning None.")
             return None 

        company_data = target_graph['Company']
        
        # 1. Return (Continuous)
        if hasattr(company_data, 'y') and company_data.y is not None:
             y_return = company_data.y
        else:
             y_return = torch.zeros(company_data.num_nodes, 1)

        # 2. Risk Label (Class)
        if hasattr(company_data, 'y_class'):
            y_risk = company_data.y_class
        else:
            y_risk = torch.zeros(company_data.num_nodes, dtype=torch.long)

        return window_graphs, y_return, y_risk

# ---------------------------------------------------------
# 2. CUSTOM COLLATE FUNCTION (The Fix for the TypeError)
# ---------------------------------------------------------
def custom_collate(batch):
    """
    Since batch_size=1, 'batch' is a list with 1 item: [(window_graphs, y_return, y_risk)]
    We just unwrap it.
    """
    elem = batch[0]
    if elem is None: return None # Handle broken graphs
    return elem

# ---------------------------------------------------------
# 3. TRAINING EXECUTION
# ---------------------------------------------------------
def run_training():
    logger.info("🚀 Starting MULTI-TASK GNN training...")
    
    # Config Setup
    config = load_local_config()
    if "neo4j" not in config: config = {"neo4j": config}
    
    pipeline = GNNPipeline(config)
    
    # --- LOAD REAL SNAPSHOTS FROM DISK ---
    # This uses the files you generated with 'generate_fake_history.py'
    snapshots = pipeline.load_historical_sequence(days=60)
    
    if len(snapshots) < 6:
        logger.error("❌ Not enough history files found in 'graph_snapshots/'. Run generate_fake_history.py first!")
        return

    # Dataset
    WINDOW_SIZE = 5
    split_idx = int(len(snapshots) * 0.8)
    
    train_snapshots = snapshots[:split_idx]
    test_snapshots = snapshots[split_idx:]
    
    logger.info(f"Train size: {len(train_snapshots)} snapshots | Test size: {len(test_snapshots)} snapshots")

    train_dataset = MultiTaskDataset(train_snapshots, window_size=WINDOW_SIZE)
    test_dataset = MultiTaskDataset(test_snapshots, window_size=WINDOW_SIZE)
    
    # We use custom_collate here!
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

    # Model
    # Use the first valid snapshot to initialize model dimensions
    model = create_hetero_model(snapshots[0], hidden_dim=64)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)
    logger.info(f"Model initialized on {DEVICE}")

    # Optimizers
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn_forecast = nn.MSELoss()        
    loss_fn_risk = nn.CrossEntropyLoss()  

    # Loop
    EPOCHS = 10
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        count = 0
        
        for batch in train_loader:
            if batch is None: continue # Skip bad data
            
            window_graphs, y_return, y_risk = batch
            
            # Move list of graphs to device manually
            window_graphs = [g.to(DEVICE) for g in window_graphs]
            y_return = y_return.to(DEVICE).squeeze(0) # [Num_Companies, 1]
            y_risk = y_risk.to(DEVICE).squeeze(0)     # [Num_Companies]
            
            optimizer.zero_grad()

            # Forward Pass
            pred_return, pred_risk = model(window_graphs)
            
            # Mask NaNs
            mask = ~torch.isnan(y_return).squeeze()
            if mask.sum() > 0:
                l1 = loss_fn_forecast(pred_return[mask], y_return[mask])
                l2 = loss_fn_risk(pred_risk[mask], y_risk[mask])
                
                loss = l1 + l2 
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                count += 1
        
        avg_loss = total_loss / count if count > 0 else 0
        logger.info(f"Epoch {epoch:03d} | Total Loss: {avg_loss:.6f}")

    # Save
    torch.save(model.state_dict(), "gnn_multitask_model.pth")
    logger.info("💾 Model saved as 'gnn_multitask_model.pth'")

if __name__ == "__main__":
    run_training()