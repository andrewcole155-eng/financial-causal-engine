import logging
import torch
from neo4j import GraphDatabase
from torch_geometric.data import HeteroData
import numpy as np
import sys

logger = logging.getLogger(__name__)

class GNNPipeline:
    def __init__(self, config):
        # 1. Robust Credential Extraction
        # We look for the 'neo4j' block, but default to the root config if missing
        neo_conf = config.get('neo4j', config)
        
        # Try both naming conventions ('uri' vs 'neo4j_uri')
        self.uri = neo_conf.get('uri') or neo_conf.get('neo4j_uri')
        self.user = neo_conf.get('user') or neo_conf.get('neo4j_user')
        self.password = neo_conf.get('password') or neo_conf.get('neo4j_password')

        # validation
        if not all([self.uri, self.user, self.password]):
            logger.error(f"❌ Missing Neo4j credentials! Found: URI={self.uri}, User={self.user}")
            raise ValueError("Invalid Configuration: Missing Neo4j connection details.")

        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
        # Mapping tickers to integer IDs for PyTorch
        self.ticker_to_id = {} 

    def get_graph_data(self):
        logger.info("✅ GNN Pipeline connected to Neo4j (Official Driver).")
        logger.info("Building HeteroData object for GNN...")
        
        data = HeteroData()

        with self.driver.session() as session:
            # 1. Fetch ALL Company Nodes (Including Macro Nodes)
            logger.info("Fetching Company nodes and features...")
            query_nodes = """
            MATCH (c:Company)
            RETURN c.ticker as ticker, 
                   c.market_cap as market_cap, 
                   c.is_macro as is_macro
            ORDER BY c.ticker ASC
            """
            results = session.run(query_nodes).data()
            
            if not results:
                logger.error("No nodes found in Neo4j!")
                return None

            # Create mappings
            self.ticker_to_id = {row['ticker']: i for i, row in enumerate(results)}
            
            # Create Features
            # We use Market Cap + Is_Macro flag as basic features
            features = []
            for row in results:
                cap = float(row['market_cap']) if row['market_cap'] else 0.0
                is_macro = 1.0 if row.get('is_macro') else 0.0
                # Normalize cap (log scale) to keep it in range
                cap_norm = np.log1p(cap) 
                features.append([cap_norm, is_macro])

            data['Company'].x = torch.tensor(features, dtype=torch.float)
            data['Company'].num_nodes = len(results)
            logger.info(f"   -> Processed {len(results)} Company nodes.")

            # 2. Fetch Event Nodes & Relationships (Existing Logic)
            logger.info("Fetching Event nodes and features...")
            query_events = """
            MATCH (e:Event)
            RETURN elementId(e) as id, e.score as score
            """
            event_results = session.run(query_events).data()
            
            # Map Event IDs
            event_id_map = {row['id']: i for i, row in enumerate(event_results)}
            event_features = [[float(r['score'])] for r in event_results]
            
            if event_features:
                data['Event'].x = torch.tensor(event_features, dtype=torch.float)
                data['Event'].num_nodes = len(event_results)
            
                # Fetch HAD_EVENT edges
                logger.info("Fetching (Company)-[:HAD_EVENT]->(Event) edges...")
                query_rels = """
                MATCH (c:Company)-[:HAD_EVENT]->(e:Event)
                RETURN c.ticker as c_ticker, elementId(e) as e_id
                """
                rel_results = session.run(query_rels).data()
                
                sources = []
                targets = []
                for r in rel_results:
                    if r['c_ticker'] in self.ticker_to_id and r['e_id'] in event_id_map:
                        sources.append(self.ticker_to_id[r['c_ticker']])
                        targets.append(event_id_map[r['e_id']])
                
                edge_index = torch.tensor([sources, targets], dtype=torch.long)
                data['Company', 'had_event', 'Event'].edge_index = edge_index
                logger.info(f"   -> Processed {len(rel_results)} news relationships.")

            # ------------------------------------------------------------------
            # 3. NEW: Fetch GLOBAL MACRO EDGES
            # ------------------------------------------------------------------
            logger.info("Fetching Global Macro Influence edges...")
            
            # We fetch ANY relationship where the source is a Macro node
            # and the target is a regular Company.
            # We group them all under a single edge type 'macro_influence' for the GNN.
            query_macro = """
            MATCH (m:Company)-[r]->(c:Company)
            WHERE m.is_macro = true 
            RETURN m.ticker as source, c.ticker as target
            """
            macro_results = session.run(query_macro).data()
            
            m_sources = []
            m_targets = []
            
            for r in macro_results:
                if r['source'] in self.ticker_to_id and r['target'] in self.ticker_to_id:
                    m_sources.append(self.ticker_to_id[r['source']])
                    m_targets.append(self.ticker_to_id[r['target']])

            if m_sources:
                macro_edge_index = torch.tensor([m_sources, m_targets], dtype=torch.long)
                # We define this new edge type in the HeteroData object
                data['Company', 'macro_influence', 'Company'].edge_index = macro_edge_index
                logger.info(f"   -> Processed {len(macro_results)} macro relationships.")
            else:
                logger.warning("   -> ⚠️ No Macro edges found in Neo4j!")

        logger.info("✅ HeteroData object successfully created.")
        return data

    def save_predictions(self, risk_scores):
        """
        Writes the calculated risk scores back to Neo4j AND returns them.
        """
        logger.info("💾 Writing GNN Risk Scores back to Neo4j...")
        
        # Convert tensor to list
        scores_list = risk_scores.tolist()
        
        updates = []
        # Reverse map ID -> Ticker
        id_to_ticker = {v: k for k, v in self.ticker_to_id.items()}
        
        for idx, score in enumerate(scores_list):
            if idx in id_to_ticker:
                updates.append({
                    "ticker": id_to_ticker[idx], 
                    "score": float(score)
                })

        # Batch write
        batch_size = 500
        with self.driver.session() as session:
            for i in range(0, len(updates), batch_size):
                batch = updates[i:i+batch_size]
                session.run("""
                UNWIND $batch as item
                MATCH (c:Company {ticker: item.ticker})
                SET c.gnn_risk_score = item.score,
                    c.last_risk_update = datetime()
                """, batch=batch)
                logger.info(f"    -> Updated batch {i//batch_size + 1}")
        
        logger.info(f"✅ Successfully updated risk scores for {len(updates)} companies.")
        
        # --- NEW LINE HERE ---
        return updates