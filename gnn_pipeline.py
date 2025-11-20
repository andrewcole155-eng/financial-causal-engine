import logging
from typing import Dict, Any, List, Optional
from py2neo import Graph
from torch_geometric.data import HeteroData
import torch
import os

# --- Setup structured logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GNNPipeline:
    """
    Manages the extraction of graph data from Neo4j and its conversion 
    into a PyTorch Geometric HeteroData object for the GNN.
    Also handles writing prediction scores back to Neo4j.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the connection to Neo4j.
        """
        self.neo4j_graph: Optional[Graph] = None
        self.company_ticker_map: List[str] = [] # Stores mapping of Index -> Ticker
        
        try:
            neo4j_uri = os.getenv("NEO4J_URI", config.get("neo4j_uri"))
            neo4j_user = os.getenv("NEO4J_USER", config.get("neo4j_user"))
            neo4j_password = os.getenv("NEO4J_PASSWORD", config.get("neo4j_password"))

            if not all([neo4j_uri, neo4j_user, neo4j_password]):
                # Fallback for Docker secrets or alternative config paths
                logger.warning("Env vars missing, checking passed config dict...")
                neo4j_uri = config.get("uri") or config.get("neo4j_uri")
                neo4j_user = config.get("user") or config.get("neo4j_user")
                neo4j_password = config.get("password") or config.get("neo4j_password")

            if not all([neo4j_uri, neo4j_user, neo4j_password]):
                 raise ValueError("Neo4j connection details not found.")

            self.neo4j_graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
            # Quick connectivity check
            self.neo4j_graph.run("MATCH (n) RETURN count(n) LIMIT 1")
            logger.info("✅ GNN Pipeline connected to Neo4j.")
            
        except Exception as e:
            logger.critical(f"❌ GNN Pipeline failed to connect to Neo4j: {e}")
            self.neo4j_graph = None

    def get_graph_data(self) -> Optional[HeteroData]:
        """
        Fetches all graph data from Neo4j and constructs a HeteroData object.
        """
        if not self.neo4j_graph:
            logger.error("No Neo4j connection.")
            return None

        logger.info("Building HeteroData object for GNN...")
        data = HeteroData()

        # --- 1. Process Company Nodes ---
        logger.info("Fetching Company nodes and features...")
        # Query to get Company features and a new, 0-indexed ID for PyG
        company_query = """
        MATCH (c:Company)
        RETURN c.ticker AS ticker, c.market_cap AS market_cap, id(c) AS neo4j_id
        """
        
        company_nodes = self.neo4j_graph.run(company_query).data()
        
        if not company_nodes:
            logger.warning("No Company nodes found in Neo4j. GNN Data will be empty.")
            return None

        # SAVE THE MAPPING: Index -> Ticker
        # This is crucial for writing results back to the correct company later
        self.company_ticker_map = [node['ticker'] for node in company_nodes]
        
        # Create a mapping: {neo4j_id: pyg_index}
        company_id_map = {node['neo4j_id']: i for i, node in enumerate(company_nodes)}
        
        # Create the feature tensor `x`
        # Using market_cap as the first feature. Handling None values.
        company_features = [
            [float(node['market_cap']) if node['market_cap'] is not None else 0.0] 
            for node in company_nodes
        ]
        
        data['Company'].x = torch.tensor(company_features, dtype=torch.float)
        data['Company'].num_nodes = len(company_nodes)
        logger.info(f" -> Processed {data['Company'].num_nodes} Company nodes.")


        # --- 2. Process Event Nodes ---
        logger.info("Fetching Event nodes and features...")
        event_query = """
        MATCH (e:Event)
        RETURN e.score AS score, id(e) AS neo4j_id
        """
        
        event_nodes = self.neo4j_graph.run(event_query).data()
        
        # Mapping: {neo4j_id: pyg_index}
        event_id_map = {node['neo4j_id']: i for i, node in enumerate(event_nodes)}
        
        # Feature tensor `x`
        event_features = [
            [float(node['score']) if node['score'] is not None else 0.0] 
            for node in event_nodes
        ]
        
        # Handle edge case where there are no events yet
        if event_nodes:
            data['Event'].x = torch.tensor(event_features, dtype=torch.float)
        else:
            # If no events, create empty tensor to prevent crash
            data['Event'].x = torch.zeros((0, 1), dtype=torch.float)
            
        data['Event'].num_nodes = len(event_nodes)
        logger.info(f" -> Processed {data['Event'].num_nodes} Event nodes.")


        # --- 3. Process Relationships (Edges) ---
        logger.info("Fetching (Company)-[:HAD_EVENT]->(Event) edges...")
        had_event_query = """
        MATCH (c:Company)-[:HAD_EVENT]->(e:Event)
        RETURN id(c) AS source_id, id(e) AS target_id
        """
        
        edges = self.neo4j_graph.run(had_event_query).data()
        
        source_indices = []
        target_indices = []
        
        for edge in edges:
            source_pyg_id = company_id_map.get(edge['source_id'])
            target_pyg_id = event_id_map.get(edge['target_id'])
            
            if source_pyg_id is not None and target_pyg_id is not None:
                source_indices.append(source_pyg_id)
                target_indices.append(target_pyg_id)

        if source_indices:
            edge_index_forward = torch.tensor([source_indices, target_indices], dtype=torch.long)
            data['Company', 'had_event', 'Event'].edge_index = edge_index_forward
            
            # Create Reverse Edges for Message Passing (Event -> Company)
            edge_index_reverse = torch.tensor([target_indices, source_indices], dtype=torch.long)
            data['Event', 'is_event_of', 'Company'].edge_index = edge_index_reverse
            
            logger.info(f" -> Processed {len(source_indices)} relationships.")
        else:
            logger.warning("No relationships found between Companies and Events.")
            # Initialize empty edge indices to prevent GNN crash
            empty_edge = torch.empty((2, 0), dtype=torch.long)
            data['Company', 'had_event', 'Event'].edge_index = empty_edge
            data['Event', 'is_event_of', 'Company'].edge_index = empty_edge

        logger.info("✅ HeteroData object successfully created.")
        return data

    def save_predictions(self, predictions: torch.Tensor):
        """
        Writes the calculated Risk Scores back to the Neo4j database.
        
        Args:
            predictions (torch.Tensor): A tensor of shape [num_companies] or [num_companies, 1]
                                        containing the risk scores (0.0 to 1.0).
        """
        if not self.neo4j_graph:
            logger.error("Cannot save predictions: No database connection.")
            return

        if not self.company_ticker_map:
            logger.error("Cannot save predictions: Node mapping is empty. Did you run get_graph_data()?")
            return

        logger.info("💾 Writing GNN Risk Scores back to Neo4j...")
        
        # Ensure tensor is on CPU and flattened to a list
        scores_list = predictions.detach().cpu().flatten().tolist()
        
        if len(scores_list) != len(self.company_ticker_map):
            logger.error(f"Mismatch: Generated {len(scores_list)} scores but mapped {len(self.company_ticker_map)} companies.")
            return

        # Prepare batch for Neo4j
        batch_data = []
        for i, score in enumerate(scores_list):
            ticker = self.company_ticker_map[i]
            batch_data.append({"ticker": ticker, "score": float(score)})

        # Cypher query to update nodes in bulk
        update_query = """
        UNWIND $batch AS row
        MATCH (c:Company {ticker: row.ticker})
        SET c.gnn_risk_score = row.score,
            c.last_risk_update = datetime()
        """
        
        try:
            # Write in chunks to avoid memory issues if graph is huge
            batch_size = 500
            for k in range(0, len(batch_data), batch_size):
                chunk = batch_data[k:k+batch_size]
                self.neo4j_graph.run(update_query, batch=chunk)
                logger.info(f"   -> Updated batch {k // batch_size + 1}")
                
            logger.info(f"✅ Successfully updated risk scores for {len(batch_data)} companies.")
            
        except Exception as e:
            logger.error(f"❌ Failed to write risk scores to Neo4j: {e}")