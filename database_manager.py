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

def _clean_properties(properties: Any) -> Dict[str, Any]:
    """
    Recursively converts Neo4j/Python complex types (like dates)
    into simple strings AND removes reserved keywords.
    """
    # Force conversion to dict if it's a Neo4j Entity
    if hasattr(properties, 'items'):
        properties = dict(properties)
    
    if not isinstance(properties, dict):
        return {}
        
    clean_props = {}
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
                # .data() converts results to a list of dicts
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
        
        # NOTE: Updated to explicit properties return to be safe, though not strictly required for events
        query = """
        MATCH (c:Company)-[:HAD_EVENT]->(e:Event)
        WHERE e.timestamp IS NOT NULL
        RETURN properties(c) as c_props, properties(e) as e_props
        ORDER BY e.timestamp DESC
        LIMIT $limit
        """
        results = self.execute_read(query, limit=limit)
        
        clean_events = []
        for r in results:
            # Flatten structure
            evt = _clean_properties(r['e_props'])
            evt['ticker'] = r['c_props'].get('ticker')
            clean_events.append(evt)
            
        return clean_events

    def get_graph_from_db(self, weight_threshold: float = 0.1) -> nx.DiGraph:
        """Fetches the entire graph from Neo4j and converts it to a NetworkX DiGraph."""
        G = nx.DiGraph()
        if not self.is_connected(): return G
            
        # Get Nodes (Explicit Properties)
        nodes_query = "MATCH (n:Company) RETURN properties(n) as props"
        nodes_result = self.execute_read(nodes_query)
        for record in nodes_result:
            node_data = _clean_properties(record["props"]) 
            G.add_node(node_data['ticker'], **node_data)
        
        # Get Edges (Explicit Properties)
        edges_query = f"""
        MATCH (n:Company)-[r]->(m:Company)
        WHERE r.weight > $weight_threshold
        RETURN n.ticker AS source, m.ticker AS target, properties(r) as props
        """
        edges_result = self.execute_read(edges_query, weight_threshold=weight_threshold)
        for record in edges_result:
            rel_data = _clean_properties(record["props"]) 
            G.add_edge(record["source"], record["target"], **rel_data)
            
        return G

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

def _clean_properties(properties: Any) -> Dict[str, Any]:
    """
    Recursively converts Neo4j/Python complex types (like dates)
    into simple strings AND removes reserved keywords.
    """
    # Force conversion to dict if it's a Neo4j Entity
    if hasattr(properties, 'items'):
        properties = dict(properties)
    
    if not isinstance(properties, dict):
        return {}
        
    clean_props = {}
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
                # .data() converts results to a list of dicts
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
        
        # NOTE: Updated to explicit properties return to be safe, though not strictly required for events
        query = """
        MATCH (c:Company)-[:HAD_EVENT]->(e:Event)
        WHERE e.timestamp IS NOT NULL
        RETURN properties(c) as c_props, properties(e) as e_props
        ORDER BY e.timestamp DESC
        LIMIT $limit
        """
        results = self.execute_read(query, limit=limit)
        
        clean_events = []
        for r in results:
            # Flatten structure
            evt = _clean_properties(r['e_props'])
            evt['ticker'] = r['c_props'].get('ticker')
            clean_events.append(evt)
            
        return clean_events

    def get_graph_from_db(self, weight_threshold: float = 0.1) -> nx.DiGraph:
        """Fetches the entire graph from Neo4j and converts it to a NetworkX DiGraph."""
        G = nx.DiGraph()
        if not self.is_connected(): return G
            
        # Get Nodes (Explicit Properties)
        nodes_query = "MATCH (n:Company) RETURN properties(n) as props"
        nodes_result = self.execute_read(nodes_query)
        for record in nodes_result:
            node_data = _clean_properties(record["props"]) 
            G.add_node(node_data['ticker'], **node_data)
        
        # Get Edges (Explicit Properties)
        edges_query = f"""
        MATCH (n:Company)-[r]->(m:Company)
        WHERE r.weight > $weight_threshold
        RETURN n.ticker AS source, m.ticker AS target, properties(r) as props
        """
        edges_result = self.execute_read(edges_query, weight_threshold=weight_threshold)
        for record in edges_result:
            rel_data = _clean_properties(record["props"]) 
            G.add_edge(record["source"], record["target"], **rel_data)
            
        return G

    def get_neighborhood_graph(self, company_ticker: str) -> nx.DiGraph:
        """Fetches a 1-hop neighborhood with UNDIRECTED AI prioritization."""
        G = nx.DiGraph()
        if not self.is_connected(): return G
        
        # 1. Fetch EVERYTHING (Duplicates included).
        # We explicitly request 'properties(r)' to ensure Python gets a Dict, not an Object.
        query = """
        MATCH (c:Company {ticker: $ticker})-[r]-(neighbor:Company)
        RETURN properties(c) as c_props, 
               properties(neighbor) as n_props, 
               properties(r) as r_props, 
               startNode(r).ticker AS source_ticker, 
               endNode(r).ticker AS target_ticker
        """
        
        try:
            result_list = self.execute_read(query, ticker=company_ticker)
            
            if not result_list:
                # Handle isolated node case
                q_node = "MATCH (c:Company {ticker: $ticker}) RETURN properties(c) as c_props"
                node_data_list = self.execute_read(q_node, ticker=company_ticker)
                if node_data_list:
                    node_data = _clean_properties(node_data_list[0]['c_props'])
                    G.add_node(node_data['ticker'], **node_data)
                return G 

            # 2. THE UNDIRECTED OVERRIDE LOGIC
            unique_edges = {} # Key: tuple(sorted([A, B])) -> Value: (source, target, props)
            nodes_map = {}

            for record in result_list:
                source = record['source_ticker']
                target = record['target_ticker']
                
                # --- DIRECT PROPERTY ACCESS (Cleaned) ---
                rel_data = _clean_properties(record['r_props'])
                c_data = _clean_properties(record['c_props'])
                n_data = _clean_properties(record['n_props'])
                
                nodes_map[c_data['ticker']] = c_data
                nodes_map[n_data['ticker']] = n_data

                # Create undirected key (A-B is same as B-A)
                pair_key = tuple(sorted([source, target]))
                
                # LOGIC: Prioritize AI edges above all else
                is_new_ai = rel_data.get('verification_status') == 'AI_PROPOSED'
                
                if pair_key not in unique_edges:
                    unique_edges[pair_key] = (source, target, rel_data)
                else:
                    _, _, existing_data = unique_edges[pair_key]
                    is_existing_ai = existing_data.get('verification_status') == 'AI_PROPOSED'
                    
                    # Overwrite if new is AI and existing is NOT
                    if is_new_ai and not is_existing_ai:
                        unique_edges[pair_key] = (source, target, rel_data)
                    # If both are verified, usually keep the one with higher weight (optional)
                    elif not is_new_ai and not is_existing_ai:
                         if rel_data.get('weight', 0) > existing_data.get('weight', 0):
                             unique_edges[pair_key] = (source, target, rel_data)

            # 3. BUILD GRAPH
            # Add Nodes
            for ticker, data in nodes_map.items():
                G.add_node(ticker, **data)
                
            # Add Edges (Convert back to list to sort and limit)
            all_final_edges = list(unique_edges.values())
            
            # Sort: AI edges (10.0) first, then weight
            sorted_edges = sorted(
                all_final_edges, 
                key=lambda x: 10.0 if x[2].get('verification_status') == 'AI_PROPOSED' else x[2].get('weight', 0.0),
                reverse=True
            )
            
            # Limit to 75 edges to prevent UI lag
            for source, target, data in sorted_edges[:75]:
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