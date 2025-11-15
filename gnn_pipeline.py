# gnn_pipeline.py

import logging
from typing import Dict, Any, Tuple
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
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the connection to Neo4j.
        """
        self.neo4j_graph: Graph | None = None
        try:
            neo4j_uri = os.getenv("NEO4J_URI", config.get("neo4j_uri"))
            neo4j_user = os.getenv("NEO4J_USER", config.get("neo4j_user"))
            neo4j_password = os.getenv("NEO4J_PASSWORD", config.get("neo4j_password"))

            if not all([neo4j_uri, neo4j_user, neo4j_password]):
                raise ValueError("Neo4j connection details not found.")

            self.neo4j_graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
            self.neo4j_graph.run("MATCH (n) RETURN count(n)")
            logger.info("✅ GNN Pipeline connected to Neo4j.")
        except Exception as e:
            logger.critical(f"❌ GNN Pipeline failed to connect to Neo4j: {e}")

    def get_graph_data(self) -> HeteroData | None:
        """
        Fetches all graph data from Neo4j and constructs a HeteroData object.
        This object is the primary input for our GNN.
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
        
        # We need to map the long, complex Neo4j ID to a simple 0, 1, 2... index
        # PyG models require this.
        company_nodes = self.neo4j_graph.run(company_query).data()
        
        # Create a mapping: {neo4j_id: pyg_index}
        company_id_map = {node['neo4j_id']: i for i, node in enumerate(company_nodes)}
        
        # Create the feature tensor `x`
        # Using market_cap as the first feature. We can add more.
        # We must handle missing (None) values, e.g., by replacing with 0
        company_features = [
            [node['market_cap'] if node['market_cap'] is not None else 0] 
            for node in company_nodes
        ]
        
        data['Company'].x = torch.tensor(company_features, dtype=torch.float)
        # We also store the number of nodes
        data['Company'].num_nodes = len(company_nodes)
        logger.info(f" -> Processed {data['Company'].num_nodes} Company nodes.")


        # --- 2. Process Event Nodes ---
        logger.info("Fetching Event nodes and features...")
        # Query to get Event features and a 0-indexed ID
        event_query = """
        MATCH (e:Event)
        RETURN e.score AS score, id(e) AS neo4j_id
        """
        
        event_nodes = self.neo4j_graph.run(event_query).data()
        
        # Create a mapping: {neo4j_id: pyg_index}
        event_id_map = {node['neo4j_id']: i for i, node in enumerate(event_nodes)}
        
        # Create the feature tensor `x`
        event_features = [
            [node['score'] if node['score'] is not None else 0] 
            for node in event_nodes
        ]
        
        data['Event'].x = torch.tensor(event_features, dtype=torch.float)
        data['Event'].num_nodes = len(event_nodes)
        logger.info(f" -> Processed {data['Event'].num_nodes} Event nodes.")


        # --- 3. Process Relationships (Edges) ---
        logger.info("Fetching (Company)-[:HAD_EVENT]->(Event) edges...")
        # Query to get the 'source' (Company) and 'target' (Event)
        had_event_query = """
        MATCH (c:Company)-[:HAD_EVENT]->(e:Event)
        RETURN id(c) AS source_id, id(e) AS target_id
        """
        
        edges = self.neo4j_graph.run(had_event_query).data()
        
        # We build the `edge_index` by converting Neo4j IDs to our new PyG indexes
        source_indices = []
        target_indices = []
        
        for edge in edges:
            source_pyg_id = company_id_map.get(edge['source_id'])
            target_pyg_id = event_id_map.get(edge['target_id'])
            
            # Only add the edge if both nodes were successfully mapped
            if source_pyg_id is not None and target_pyg_id is not None:
                source_indices.append(source_pyg_id)
                target_indices.append(target_pyg_id)

        # PyG requires edge_index in a [2, num_edges] shape
        edge_index_forward = torch.tensor([source_indices, target_indices], dtype=torch.long)
        
        # We store this edge_index in our HeteroData object
        data['Company', 'had_event', 'Event'].edge_index = edge_index_forward
        logger.info(f" -> Processed {len(source_indices)} ('Company', 'had_event', 'Event') relationships.")

        # --- THIS IS THE NEW LINE YOU MUST ADD ---
        # For message passing, we need Events to pass info TO Companies.
        # We create a new, reversed edge_index: [target, source]
        edge_index_reverse = torch.tensor([target_indices, source_indices], dtype=torch.long)
        
        # We store this under a new, "reverse" relationship key.
        data['Event', 'is_event_of', 'Company'].edge_index = edge_index_reverse
        logger.info(f" -> Processed {len(target_indices)} ('Event', 'is_event_of', 'Company') relationships.")

        # --- TODO: Process (Company)-[:REL_TYPE]->(Company) Edges ---
        # We would repeat the process above for your other company-to-company
        # relationships, which would be keyed as ('Company', 'SUPPLIES', 'Company') etc.
        # We'll add this later to keep this step simple.

        logger.info("✅ HeteroData object successfully created.")
        return data