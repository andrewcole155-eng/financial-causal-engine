import logging
import torch
import os
import glob
from datetime import datetime, timedelta
from neo4j import GraphDatabase
from torch_geometric.data import HeteroData
import numpy as np
import sys

logger = logging.getLogger(__name__)

class GNNPipeline:
    def __init__(self, config):
        # 1. Robust Credential Extraction
        neo_conf = config.get('neo4j', config)
        self.uri = neo_conf.get('uri') or neo_conf.get('neo4j_uri')
        self.user = neo_conf.get('user') or neo_conf.get('neo4j_user')
        self.password = neo_conf.get('password') or neo_conf.get('neo4j_password')

        if not all([self.uri, self.user, self.password]):
            logger.error(f"❌ Missing Neo4j credentials! Found: URI={self.uri}, User={self.user}")
            raise ValueError("Invalid Configuration: Missing Neo4j connection details.")

        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.ticker_to_id = {} 
        
        self.snapshot_dir = os.path.join(os.path.dirname(__file__), "graph_snapshots")
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def get_graph_data(self, target_date=None):
        logger.info("✅ GNN Pipeline connected to Neo4j (Official Driver).")
        logger.info("Building HeteroData object for GNN...")
        
        data = HeteroData()

        with self.driver.session() as session:
            # 1. Fetch ALL Company Nodes with COALESCE for Safety
            query_nodes = """
            MATCH (c:Company)
            RETURN c.ticker as ticker, 
                   COALESCE(c.market_cap, 0.0) as market_cap, 
                   COALESCE(c.is_macro, false) as is_macro,
                   COALESCE(c.last_close, 100.0) as close_price,     
                   COALESCE(c.daily_return, 0.0) as daily_return,    
                   COALESCE(c.risk_label, 0) as risk_label           
            ORDER BY c.ticker ASC
            """
            results = session.run(query_nodes).data()
            
            if not results:
                logger.error("No nodes found in Neo4j!")
                return None

            self.ticker_to_id = {row['ticker']: i for i, row in enumerate(results)}
            
            features = []
            y_returns = []
            y_risks = []

            for row in results:
                # --- UPDATED FEATURE CONSTRUCTION (Size 6) ---
                # Matches the logic in generate_real_history.py & train.py
                
                # 1. Market Cap (Log Normalized)
                cap = float(row['market_cap'])
                cap_norm = np.log1p(cap)
                
                # 2. Macro Flag
                is_macro = 1.0 if row['is_macro'] else 0.0
                
                # 3-6. DYNAMIC PLACEHOLDERS (Vol, Shock, Trend)
                # These are 0.0 by default here. 
                # They MUST be populated by 'enrich_data' in predict.py or generate_history.py
                # We pad them here so the tensor shape (N, 6) is correct immediately.
                feat_vol_shock = 0.0
                feat_sent_shock = 0.0
                feat_trend = 0.0
                feat_vol_mag = 0.0
                
                # Final Vector: [Cap, Macro, VolShock, SentShock, Trend, VolMag]
                features.append([cap_norm, is_macro, feat_vol_shock, feat_sent_shock, feat_trend, feat_vol_mag])
                
                # Targets (Y)
                ret = float(row['daily_return'])
                y_returns.append([ret])
                
                risk = int(row['risk_label'])
                y_risks.append(risk)

            data['Company'].x = torch.tensor(features, dtype=torch.float)
            data['Company'].y = torch.tensor(y_returns, dtype=torch.float)
            data['Company'].y_class = torch.tensor(y_risks, dtype=torch.long)
            
            data['Company'].num_nodes = len(results)

            # 2. Fetch Event Nodes
            query_events = "MATCH (e:Event) RETURN elementId(e) as id, e.score as score"
            event_results = session.run(query_events).data()
            
            if event_results:
                event_id_map = {row['id']: i for i, row in enumerate(event_results)}
                event_features = [[float(r['score'])] for r in event_results]
                data['Event'].x = torch.tensor(event_features, dtype=torch.float)
                
                query_rels = """
                MATCH (c:Company)-[:HAD_EVENT]->(e:Event)
                RETURN c.ticker as c_ticker, elementId(e) as e_id
                """
                rel_results = session.run(query_rels).data()
                
                sources, targets = [], []
                for r in rel_results:
                    if r['c_ticker'] in self.ticker_to_id and r['e_id'] in event_id_map:
                        sources.append(self.ticker_to_id[r['c_ticker']])
                        targets.append(event_id_map[r['e_id']])
                
                if sources:
                    data['Company', 'had_event', 'Event'].edge_index = torch.tensor([sources, targets], dtype=torch.long)

            # 3. Macro Edges
            query_macro = """
            MATCH (m:Company)-[r]->(c:Company) WHERE m.is_macro = true 
            RETURN m.ticker as source, c.ticker as target
            """
            macro_results = session.run(query_macro).data()
            m_sources, m_targets = [], []
            for r in macro_results:
                if r['source'] in self.ticker_to_id and r['target'] in self.ticker_to_id:
                    m_sources.append(self.ticker_to_id[r['source']])
                    m_targets.append(self.ticker_to_id[r['target']])
            
            if m_sources:
                 data['Company', 'macro_influence', 'Company'].edge_index = torch.tensor([m_sources, m_targets], dtype=torch.long)

        return data

    def save_daily_snapshot(self):
        data = self.get_graph_data()
        if data:
            today_str = datetime.now().strftime("%Y-%m-%d")
            filename = f"graph_snapshot_{today_str}.pt"
            path = os.path.join(self.snapshot_dir, filename)
            torch.save(data, path)
            logger.info(f"💾 Snapshot saved: {path}")
            return path
        return None

    def load_historical_sequence(self, days=30):
        """
        Loads the last N days of snapshots from the disk.
        """
        pattern = os.path.join(self.snapshot_dir, "graph_snapshot_*.pt")
        files = glob.glob(pattern)
        files.sort()
        recent_files = files[-days:]
        
        snapshots = []
        for f in recent_files:
            try:
                # --- FIX FOR PYTORCH 2.6+ SECURITY UPDATE ---
                # We generated these files locally, so we trust them.
                # We set weights_only=False to allow loading complex HeteroData objects.
                data = torch.load(f, weights_only=False)
                snapshots.append(data)
            except Exception as e:
                logger.error(f"Failed to load snapshot {f}: {e}")
                
        logger.info(f"📚 Loaded {len(snapshots)} historical snapshots for training.")
        return snapshots

    def save_predictions(self, predictions_dict):
        logger.info("💾 Writing Multi-Task Predictions back to Neo4j...")
        
        batch_data = []

        if isinstance(predictions_dict, list):
             pass
        elif isinstance(predictions_dict, dict):
            for ticker, vals in predictions_dict.items():
                batch_data.append({
                    "ticker": ticker,
                    "risk_score": float(vals.get('risk_score', 0.0)),
                    "price_forecast": float(vals.get('price_forecast', 0.0))
                })

        query = """
        UNWIND $batch as item
        MATCH (c:Company {ticker: item.ticker})
        SET c.gnn_risk_score = item.risk_score,
            c.price_forecast_next_day = item.price_forecast,
            c.last_inference_date = datetime()
        """
        with self.driver.session() as session:
             session.run(query, batch=batch_data)
             
        logger.info(f"✅ Updated DB with {len(batch_data)} predictions.")