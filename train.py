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

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIG LOADER ---
def load_local_config():
    """Loads configuration from config.json."""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f: return json.load(f)
        return {}
    except Exception as e:
        logger.warning(f"Config load failed: {e}")
        return {}

# ---------------------------------------------------------
# 1. MULTI-TASK DATASET
# ---------------------------------------------------------
class MultiTaskDataset(Dataset):
    """
    Dataset that returns a sequence of 'window_size' snapshots as input,
    and the subsequent snapshot as the target.
    """
    def __init__(self, snapshots: List[HeteroData], window_size: int = 5):
        self.snapshots = snapshots
        self.window_size = window_size
        if len(self.snapshots) <= self.window_size:
            logger.warning(f"⚠️ Not enough snapshots! Has {len(snapshots)}, needs > {window_size}")

    def __len__(self):
        return max(0, len(self.snapshots) - self.window_size)

    def __getitem__(self, idx):
        # Input: Sequence of Graphs (t, t+1, ... t+w)
        window_graphs = self.snapshots[idx : idx + self.window_size]
        
        # Target: The NEXT day (t+w+1)
        target_graph = self.snapshots[idx + self.window_size]
        
        # Safety check for missing Company nodes in target
        if 'Company' not in target_graph.node_types:
             logger.warning(f"Snapshot {idx} broken (missing Company nodes). Returning None.")
             return None 

        company_data = target_graph['Company']
        
        # 1. Regression Target (Continuous Return)
        if hasattr(company_data, 'y') and company_data.y is not None:
             y_return = company_data.y
        else:
             y_return = torch.zeros(company_data.num_nodes, 1)

        # 2. Classification Target (Risk Class)
        if hasattr(company_data, 'y_class'):
            y_risk = company_data.y_class
        else:
            y_risk = torch.zeros(company_data.num_nodes, dtype=torch.long)

        return window_graphs, y_return, y_risk

# ---------------------------------------------------------
# 2. CUSTOM COLLATE FUNCTION
# ---------------------------------------------------------
def custom_collate(batch):
    """
    Handles batching of graph sequences.
    Since we use batch_size=1 for temporal sequences, we just unwrap the list.
    """
    elem = batch[0]
    if elem is None: return None # Filter out broken graphs
    return elem

# ---------------------------------------------------------
# 3. TRAINING EXECUTION
# ---------------------------------------------------------
def run_training():
    logger.info("🚀 Starting MULTI-TASK GNN training...")
    
    # 1. Config Setup
    config = load_local_config()
    if "neo4j" not in config: config = {"neo4j": config}
    
    pipeline = GNNPipeline(config)
    
    # 2. Load Snapshots
    # Note: These snapshots must have been created by 'generate_real_history.py'
    # which uses 'feature_engineering.py' to add the volatility features.
    snapshots = pipeline.load_historical_sequence(days=60)
    
    if len(snapshots) < 6:
        logger.error("❌ Not enough history files found in 'graph_snapshots/'. Run generate_real_history.py first!")
        return

    # 3. Dataset Splitting
    WINDOW_SIZE = 5
    split_idx = int(len(snapshots) * 0.8)
    
    train_snapshots = snapshots[:split_idx]
    test_snapshots = snapshots[split_idx:]
    
    logger.info(f"Train size: {len(train_snapshots)} snapshots | Test size: {len(test_snapshots)} snapshots")

    train_dataset = MultiTaskDataset(train_snapshots, window_size=WINDOW_SIZE)
    test_dataset = MultiTaskDataset(test_snapshots, window_size=WINDOW_SIZE)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)

    # 4. Model Initialization
    first_snapshot = snapshots[0]
    
    # --- CRITICAL DIMENSION CHECK ---
    # We now expect 6 features: [Close, Sentiment, Volatility, Vol_Shock, Sent_Shock, Trend]
    input_dim = first_snapshot['Company'].x.shape[1]
    EXPECTED_DIM = 6
    
    if input_dim != EXPECTED_DIM:
        logger.warning(f"⚠️ Feature Dimension Mismatch! Found {input_dim}, expected {EXPECTED_DIM}.")
        logger.warning("Did you re-run 'generate_real_history.py' after creating 'feature_engineering.py'?")
    else:
        logger.info(f"✅ Input features verified: {input_dim} (Matches Volatility Logic)")
    
    # Initialize Model
    model = create_hetero_model(first_snapshot, hidden_dim=64)
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(DEVICE)
    logger.info(f"Model initialized on {DEVICE}")

    # 5. Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Decay LR by 0.5 every 20 epochs to fine-tune convergence
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    loss_fn_forecast = nn.MSELoss()        
    loss_fn_risk = nn.CrossEntropyLoss()   

    # 6. Training Loop
    EPOCHS = 100 
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        count = 0
        
        for batch in train_loader:
            if batch is None: continue 
            
            window_graphs, y_return, y_risk = batch
            
            # Move data to GPU/CPU
            window_graphs = [g.to(DEVICE) for g in window_graphs]
            y_return = y_return.to(DEVICE).squeeze(0) # Shape: [Num_Companies, 1]
            y_risk = y_risk.to(DEVICE).squeeze(0)     # Shape: [Num_Companies]
            
            optimizer.zero_grad()

            # Forward Pass
            pred_return, pred_risk = model(window_graphs)
            
            # Mask NaNs (Handle companies missing data for specific days)
            mask = ~torch.isnan(y_return).squeeze()
            
            if mask.sum() > 0:
                # Multi-Task Loss: Return Error + Risk Classification Error
                l1 = loss_fn_forecast(pred_return[mask], y_return[mask])
                l2 = loss_fn_risk(pred_risk[mask], y_risk[mask])
                
                loss = l1 + l2 
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                count += 1
        
        # Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        avg_loss = total_loss / count if count > 0 else 0
        
        # Log progress
        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:03d} | LR: {current_lr:.6f} | Total Loss: {avg_loss:.6f}")

    # 7. Save Model
    torch.save(model.state_dict(), "gnn_multitask_model.pth")
    logger.info("💾 Model saved as 'gnn_multitask_model.pth'")

if __name__ == "__main__":
    run_training()