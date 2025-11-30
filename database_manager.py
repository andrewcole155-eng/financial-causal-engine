# ==============================================================================
# --- IMPORTS ---
# ==============================================================================
import sqlite3
import logging
from typing import Dict, Any, List, Optional
import re
import os
import networkx as nx
import datetime
from neo4j import GraphDatabase, exceptions
from neo4j.time import Date, DateTime, Time

# --- Setup structured logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==============================================================================
# --- HELPER FUNCTION ---
# ==============================================================================

def _clean_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively converts Neo4j/Python complex types (like dates)
    into simple strings AND removes reserved keywords to prevent frontend errors.
    """
    clean_props = {}
    if not isinstance(properties, dict):
        return {}
        
    for key, value in properties.items():
        # pyvis/networkx will pass 'source' and 'target' as positional arguments
        if key.lower() in ["source", "target"]:
            continue
            
        if isinstance(value, (Date, DateTime, datetime.date, datetime.datetime)):
            clean_props[key] = value.isoformat()
        elif isinstance(value, (Time, datetime.time)):
            clean_props[key] = value.isoformat()
        elif isinstance(value, (int, float, str, bool)) or value is None:
            clean_props[key] = value
        elif isinstance(value, list):
            clean_props[key] = [str(item) for item in value]
        else:
            clean_props[key] = str(value)
            
    return clean_props

# ==============================================================================
# --- DATABASE MANAGER CLASS ---
# ==============================================================================

class DatabaseManager:
    """
    Manages all database interactions, providing a clean interface for connecting to,
    writing to, and reading from SQLite (for backup) and Neo4j (primary graph).
    """

    def __init__(self, config: Dict[str, Any]):
        """Initializes connections to SQLite and the Neo4j graph database."""
        logger.info("🗄️ Initializing Database Manager...")
        self.sqlite_conn: Optional[sqlite3.Connection] = None
        self.neo4j_driver: Optional[GraphDatabase.driver] = None 

        # --- Connect to SQLite (Backup/Fail-safe) ---
        try:
            db_dir = 'database'
            db_path = os.path.join(db_dir, 'financial_data.db')
            os.makedirs(db_dir, exist_ok=True) 
            self.sqlite_conn = sqlite3.connect(db_path, check_same_thread=False)
            self.sqlite_conn.row_factory = sqlite3.Row
            self._create_sqlite_tables()
            logger.info(f" -> ✅ Connected to local SQLite backup at {db_path}")
        except Exception as e:
            logger.warning(f" -> ⚠️ Failed to connect to local SQLite: {e}")
            self.sqlite_conn = None

        # --- Connect to Neo4j (Primary) ---
        try:
            # Handle nested config structure if present
            if "neo4j" in config and isinstance(config.get("neo4j"), dict):
                neo4j_config = config["neo4j"]
            else:
                neo4j_config = config

            neo4j_uri = neo4j_config.get("uri") or neo4j_config.get("neo4j_uri")
            neo4j_user = neo4j_config.get("user") or neo4j_config.get("neo4j_user")
            neo4j_password = neo4j_config.get("password") or neo4j_config.get("neo4j_password")

            if not all([neo4j_uri, neo4j_user, neo4j_password]):
                raise ValueError("Neo4j connection details are missing from configuration.")

            self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            self.neo4j_driver.verify_connectivity()
            logger.info(" -> ✅ Successfully connected to Neo4j.")
            
        except (exceptions.AuthError, exceptions.ServiceUnavailable, ValueError) as e:
            logger.critical(f" -> ❌ FATAL: Failed to connect to Neo4j: {e}")
            self.neo4j_driver = None
            # We raise here because the app cannot function correctly without the graph
            raise

    def is_connected(self) -> bool:
        """Checks if the connection to Neo4j is active."""
        return self.neo4j_driver is not None

    def _create_sqlite_tables(self):
        """Creates tables in SQLite if they don't exist."""
        if not self.sqlite_conn: return
        with self.sqlite_conn as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS companies (
                    ticker TEXT PRIMARY KEY, name TEXT, sector TEXT,
                    market_cap REAL, last_updated TEXT
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS significant_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, ticker TEXT,
                    headline TEXT, score REAL, link TEXT
                )
            ''')

    # ==========================================================================
    # --- NEO4J EXECUTION HELPERS ---
    # ==========================================================================

    def execute_write(self, query: str, **params: Any):
        """Runs a write Cypher query using a managed session."""
        if not self.is_connected(): return
        try:
            with self.neo4j_driver.session() as session:
                return session.execute_write(self._run_write_tx, query, **params)
        except Exception as e:
            logger.error(f"Failed to execute write query: {e}")

    @staticmethod
    def _run_write_tx(tx, query: str, **params: Any):
        result = tx.run(query, **params)
        return result.consume() 

    def execute_read(self, query: str, **params: Any) -> List[Dict[str, Any]]:
        """Runs a read-only Cypher query using a managed session."""
        if not self.is_connected(): return []
        try:
            with self.neo4j_driver.session() as session:
                result = session.execute_read(self._run_read_tx, query, **params)
                return result
        except Exception as e:
            logger.error(f"Failed to execute read query: {e}")
            return []
    
    @staticmethod
    def _run_read_tx(tx, query: str, **params: Any) -> List[Dict[str, Any]]:
        result = tx.run(query, **params)
        return result.data() 

    # ==========================================================================
    # --- CORE GRAPH METHODS ---
    # ==========================================================================

    def upsert_company_nodes_batch(self, nodes_data: List[Dict[str, Any]], batch_size: int = 100):
        """
        Inserts or updates a batch of company nodes in Neo4j.
        INCLUDES CHUNKING to prevent SSLEOFError on cloud instances.
        """
        if not self.is_connected() or not nodes_data:
            return

        logger.info(f" -> Upserting {len(nodes_data)} company nodes (in batches of {batch_size})...")
        
        query = """
        UNWIND $nodes_data AS node_props
        MERGE (c:Company {ticker: node_props.ticker})
        SET c.name = node_props.name,
            c.sector = node_props.sector,
            c.market_cap = node_props.market_cap
        """

        # --- CHUNKING LOGIC ---
        total = len(nodes_data)
        for i in range(0, total, batch_size):
            batch = nodes_data[i : i + batch_size]
            try:
                self.execute_write(query, nodes_data=batch)
                logger.info(f"    -> Wrote batch {i // batch_size + 1} ({len(batch)} nodes)")
            except Exception as e:
                logger.error(f"    -> ❌ Failed to write node batch {i}-{i+len(batch)}: {e}")

    def upsert_relationship(self, source_ticker: str, target_ticker: str, rel_type: str, properties: Dict[str, Any]):
        """
        Creates or updates a single relationship between two companies.
        """
        if not self.is_connected(): return
        
        # Sanitize relationship type (Cypher doesn't allow dynamic types easily)
        rel_type_sanitized = re.sub(r'[^A-Z0-9_]', '', rel_type.upper())
        if not rel_type_sanitized: return
            
        query = f"""
        MATCH (source:Company {{ticker: $source_ticker}})
        MATCH (target:Company {{ticker: $target_ticker}})
        MERGE (source)-[r:{rel_type_sanitized}]->(target)
        SET r += $properties
        """
        clean_props = _clean_properties(properties)
        self.execute_write(
            query, 
            source_ticker=source_ticker, 
            target_ticker=target_ticker, 
            properties=clean_props
        )

    def add_events_batch(self, events: List[Dict[str, Any]], batch_size: int = 50):
        """
        Adds a batch of events to Neo4j.
        INCLUDES CHUNKING to prevent connection timeouts.
        """
        if not events: return
        
        logger.info(f" -> Upserting {len(events)} events (in batches of {batch_size})...")

        query = """
        UNWIND $events AS event
        MERGE (c:Company {ticker: event.ticker})
        ON CREATE SET c.name = event.ticker, c.sector = 'Unknown'
        
        WITH c, event
        MERGE (e:Event {link: event.link})
        ON CREATE SET
            e.headline = event.headline,
            e.score = event.score,
            e.timestamp = toString(datetime()) 
        ON MATCH SET
            e.score = event.score
            
        MERGE (c)-[:HAD_EVENT]->(e)
        """
        
        # --- CHUNKING LOGIC ---
        total = len(events)
        for i in range(0, total, batch_size):
            batch = events[i : i + batch_size]
            try:
                self.execute_write(query, events=batch)
                logger.info(f"    -> Wrote event batch {i // batch_size + 1}")
            except Exception as e:
                logger.error(f"    -> ❌ Failed to write event batch {i}-{i+len(batch)}: {e}")
                
        logger.info(f" -> ✅ Processed all {total} events.")

    def get_all_events(self) -> List[Dict[str, Any]]:
        """Retrieves ALL significant events from Neo4j."""
        if not self.is_connected(): return []
        
        query = """
        MATCH (c:Company)-[:HAD_EVENT]->(e:Event)
        RETURN 
            c.ticker AS ticker, 
            e.headline AS headline, 
            e.score AS score, 
            e.link AS link, 
            e.timestamp AS timestamp
        ORDER BY e.timestamp ASC
        """
        results = self.execute_read(query)
        return [_clean_properties(event) for event in results]

    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves the most recent significant events from Neo4j."""
        if not self.is_connected(): return []
        
        query = """
        MATCH (c:Company)-[:HAD_EVENT]->(e:Event)
        WHERE e.timestamp IS NOT NULL
        RETURN 
            c.ticker AS ticker, 
            e.headline AS headline, 
            e.score AS score, 
            e.link AS link, 
            e.timestamp AS timestamp
        ORDER BY e.timestamp DESC
        LIMIT $limit
        """
        results = self.execute_read(query, limit=limit)
        return [_clean_properties(event) for event in results]

    def get_graph_from_db(self, weight_threshold: float = 0.1) -> nx.DiGraph:
        """Fetches the entire graph from Neo4j and converts it to a NetworkX DiGraph."""
        G = nx.DiGraph()
        if not self.is_connected(): return G
            
        # Get Nodes
        nodes_query = "MATCH (n:Company) RETURN n"
        nodes_result = self.execute_read(nodes_query)
        for record in nodes_result:
            node_data = _clean_properties(record["n"]) 
            G.add_node(node_data['ticker'], **node_data)
        
        # Get Edges
        edges_query = f"""
        MATCH (n:Company)-[r]->(m:Company)
        WHERE r.weight > $weight_threshold
        RETURN n.ticker AS source, m.ticker AS target, r
        """
        edges_result = self.execute_read(edges_query, weight_threshold=weight_threshold)
        for record in edges_result:
            rel_data = _clean_properties(record["r"]) 
            G.add_edge(record["source"], record["target"], **rel_data)
            
        return G

    def get_neighborhood_graph(self, company_ticker: str) -> nx.DiGraph:
        """Fetches a 1-hop neighborhood with STRICT AI prioritization."""
        G = nx.DiGraph()
        if not self.is_connected(): return G
        
        # 1. Fetch EVERYTHING (No Limit yet).
        # We fetch all raw connections so Python can decide the winner.
        query = """
        MATCH (c:Company {ticker: $ticker})-[r]-(neighbor:Company)
        RETURN c, neighbor, r, 
               startNode(r).ticker AS source_ticker, 
               endNode(r).ticker AS target_ticker
        """
        
        try:
            result_list = self.execute_read(query, ticker=company_ticker)
            
            # Handle isolated node case
            if not result_list:
                node_data_list = self.execute_read(
                    "MATCH (c:Company {ticker: $ticker}) RETURN c", ticker=company_ticker
                )
                if node_data_list:
                    node_data = _clean_properties(node_data_list[0]['c'])
                    G.add_node(node_data['ticker'], **node_data)
                return G 

            # 2. THE OVERRIDE LOGIC
            # We use a dictionary keyed by the relationship pair (Start, End).
            # This guarantees only ONE edge exists per pair.
            unique_edges = {}
            nodes_map = {}

            for record in result_list:
                source = record['source_ticker']
                target = record['target_ticker']
                rel_data = _clean_properties(record['r'])
                
                # Save node data for later
                nodes_map[record['c']['ticker']] = _clean_properties(record['c'])
                nodes_map[record['neighbor']['ticker']] = _clean_properties(record['neighbor'])

                edge_key = (source, target)
                
                # LOGIC: 
                # 1. If we haven't seen this pair, store it.
                # 2. If we HAVE seen it, but the NEW one is 'AI_PROPOSED', overwrite the old one.
                if edge_key not in unique_edges:
                    unique_edges[edge_key] = rel_data
                else:
                    current_status = unique_edges[edge_key].get('verification_status')
                    new_status = rel_data.get('verification_status')
                    
                    if new_status == 'AI_PROPOSED':
                        unique_edges[edge_key] = rel_data

            # 3. SORT & LIMIT (Python Side)
            # Now we have unique edges. We sort them so AI edges (Score 10) are at the top.
            sorted_items = sorted(
                unique_edges.items(), 
                key=lambda item: 10.0 if item[1].get('verification_status') == 'AI_PROPOSED' else item[1].get('weight', 0.0),
                reverse=True
            )
            
            # Keep top 75
            top_edges = sorted_items[:75]

            # 4. BUILD THE GRAPH
            # Add Nodes
            for ticker, data in nodes_map.items():
                G.add_node(ticker, **data)
                
            # Add Edges
            for (source, target), data in top_edges:
                G.add_edge(source, target, **data)
                
        except Exception as e:
            logger.error(f"Failed to build neighborhood graph for {company_ticker}: {e}")

        return G

    def clear_neo4j_database(self):
        """!! DANGEROUS !! Deletes all nodes and relationships."""
        if not self.is_connected(): return
        logger.warning("🚨 DELETING ALL DATA from the Neo4j database...")
        self.execute_write("MATCH (n) DETACH DELETE n")
        logger.warning(" -> ✅ Neo4j database has been cleared.")

    def close(self):
        """Closes all database connections."""
        if self.sqlite_conn:
            self.sqlite_conn.close()
        if self.neo4j_driver:
            self.neo4j_driver.close()
            logger.info(" -> Neo4j connection closed.")