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
# --- HELPER FUNCTION (UPDATED) ---
# ==============================================================================

def _clean_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively converts Neo4j/Python complex types (like dates)
    into simple strings AND removes reserved keywords ('source', 'target') 
    to prevent pyvis errors.
    """
    clean_props = {}
    if not isinstance(properties, dict):
        return {}
        
    for key, value in properties.items():
        
        # --- NEW FIX: Check for reserved keys ---
        # pyvis/networkx will pass 'source' and 'target' as positional
        # arguments, so they MUST NOT be in the properties dictionary.
        if key.lower() in ["source", "target"]:
            continue  # Skip this key entirely
        # --- END FIX ---
            
        if isinstance(value, (Date, DateTime, datetime.date, datetime.datetime)):
            clean_props[key] = value.isoformat()
        elif isinstance(value, (Time, datetime.time)):
            clean_props[key] = value.isoformat()
        elif isinstance(value, (int, float, str, bool)) or value is None:
            clean_props[key] = value
        elif isinstance(value, list):
            # Recursively clean lists
            clean_props[key] = [str(item) for item in value]
        else:
            # Fallback for any other complex type
            clean_props[key] = str(value)
            
    return clean_props

# ==============================================================================
# --- DATABASE MANAGER CLASS ---
# ==============================================================================

class DatabaseManager:
    """
    Manages all database interactions, providing a clean interface for connecting to,
    writing to, and reading from SQLite (for event logs) and Neo4j (for the graph).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes connections to SQLite and the Neo4j graph database.
        """
        logger.info("🗄️ Initializing Database Manager...")
        self.sqlite_conn: Optional[sqlite3.Connection] = None
        self.neo4j_driver: Optional[GraphDatabase.driver] = None 

        # --- Connect to SQLite (fail-safe) ---
        try:
            db_dir = 'database'
            db_path = os.path.join(db_dir, 'financial_data.db')
            os.makedirs(db_dir, exist_ok=True) 
            self.sqlite_conn = sqlite3.connect(db_path, check_same_thread=False)
            self.sqlite_conn.row_factory = sqlite3.Row
            self._create_sqlite_tables()
            logger.info(f" -> ✅ Successfully connected to SQLite at {db_path}")
        except Exception as e:
            logger.warning(f" -> ⚠️ Failed to connect to local SQLite: {e}")
            self.sqlite_conn = None

        # --- Connect to Neo4j ---
        try:
            logger.info(" -> Reading Neo4j config from provided config dictionary...")
            
            if "neo4j" in config and isinstance(config.get("neo4j"), dict):
                neo4j_config = config["neo4j"]
            else:
                neo4j_config = config

            neo4j_uri = neo4j_config.get("uri") or neo4j_config.get("neo4j_uri")
            neo4j_user = neo4j_config.get("user") or neo4j_config.get("neo4j_user")
            neo4j_password = neo4j_config.get("password") or neo4j_config.get("neo4j_password")

            if not all([neo4j_uri, neo4j_user, neo4j_password]):
                raise ValueError("Neo4j connection details are missing from the config dictionary.")

            self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
            self.neo4j_driver.verify_connectivity()
            logger.info(" -> ✅ Successfully connected to Neo4j.")
            
        except (exceptions.AuthError, exceptions.ServiceUnavailable, ValueError) as e:
            logger.critical(f" -> ❌ FATAL: Failed to connect to Neo4j: {e}")
            self.neo4j_driver = None
            raise

    def is_connected(self) -> bool:
        """Checks if the connection to Neo4j is active."""
        return self.neo4j_driver is not None

    def _create_sqlite_tables(self):
        """Creates all necessary tables in the SQLite database if they don't exist."""
        if not self.sqlite_conn: 
            logger.info(" -> Skipping SQLite table creation (no connection).")
            return
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
    # --- NEO4J HELPER METHODS ---
    # ==========================================================================

    def execute_write(self, query: str, **params: Any):
        """
        Runs a write Cypher query (e.g., CREATE, MERGE, SET, DELETE)
        using a managed session and transaction.
        """
        if not self.is_connected(): return
        try:
            with self.neo4j_driver.session() as session:
                return session.execute_write(self._run_write_tx, query, **params)
        except Exception as e:
            logger.error(f"Failed to execute write query '{query[:50]}...': {e}")

    @staticmethod
    def _run_write_tx(tx, query: str, **params: Any):
        """Helper function passed to session.execute_write"""
        result = tx.run(query, **params)
        return result.consume() 

    def execute_read(self, query: str, **params: Any) -> List[Dict[str, Any]]:
        """
        Runs a read-only Cypher query (e.g., MATCH, RETURN)
        using a managed session and transaction.
        """
        if not self.is_connected(): return []
        try:
            with self.neo4j_driver.session() as session:
                result = session.execute_read(self._run_read_tx, query, **params)
                return result
        except Exception as e:
            logger.error(f"Failed to execute read query '{query[:50]}...': {e}")
            return []
    
    @staticmethod
    def _run_read_tx(tx, query: str, **params: Any) -> List[Dict[str, Any]]:
        """Helper function passed to session.execute_read"""
        result = tx.run(query, **params)
        # .data() converts nodes/rels to dicts of their properties
        return result.data() 

    # ==========================================================================
    # --- REFACTORED NEO4J METHODS ---
    # ==========================================================================

    def upsert_company_nodes_batch(self, nodes_data: List[Dict[str, Any]]):
        """
        Inserts or updates a batch of company nodes in Neo4j.
        """
        if not self.is_connected() or not nodes_data:
            return
        logger.info(f" -> Upserting batch of {len(nodes_data)} company nodes into Neo4j...")
        query = """
        UNWIND $nodes_data AS node_props
        MERGE (c:Company {ticker: node_props.ticker})
        SET c.name = node_props.name,
            c.sector = node_props.sector,
            c.market_cap = node_props.market_cap
        """
        self.execute_write(query, nodes_data=nodes_data)

    def upsert_relationship(self, source_ticker: str, target_ticker: str, rel_type: str, properties: Dict[str, Any]):
        """
        Creates or updates a single relationship between two companies in Neo4j.
        """
        if not self.is_connected(): return
        rel_type_sanitized = re.sub(r'[^A-Z0-9_]', '', rel_type.upper())
        if not rel_type_sanitized:
            logger.warning(f"Skipping invalid relationship type: {rel_type}")
            return
            
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

    def _add_event_node_to_graph(self, ticker: str, headline: str, score: float, link: str, timestamp: str):
        """
        (Private) Creates an Event node in Neo4j and links it to a Company.
        """
        if not self.is_connected(): return
        
        query = """
        MATCH (c:Company {ticker: $ticker})
        MERGE (e:Event {link: $link})
        ON CREATE SET
            e.headline = $headline,
            e.score = $score,
            e.timestamp = $timestamp
        ON MATCH SET
            e.headline = $headline,
            e.score = $score,
            e.timestamp = $timestamp
        MERGE (c)-[:HAD_EVENT]->(e)
        """
        try:
            from datetime import datetime
            ts = timestamp if timestamp else datetime.now().isoformat()
            
            self.execute_write(query, 
                ticker=ticker, 
                link=link, 
                timestamp=ts,
                headline=headline, 
                score=score
            )
        except Exception as e:
            logger.error(f"Failed to create Event node for {ticker} in Neo4j: {e}")

    def add_event(self, ticker: str, headline: str, score: float, link: str):
        """
        Adds an event to SQLite and the Neo4j graph.
        """
        from datetime import datetime
        event_timestamp = datetime.now()
        timestamp_str = event_timestamp.isoformat()

        if self.sqlite_conn:
            try:
                with self.sqlite_conn as conn:
                    conn.execute(
                        "INSERT INTO significant_events (timestamp, ticker, headline, score, link) VALUES (?, ?, ?, ?, ?)",
                        (event_timestamp, ticker, headline, score, link)
                    )
                logger.info(f" -> ✅ Event for {ticker} saved to SQLite.")
            except Exception as e:
                logger.error(f" -> Failed to save event for {ticker} to SQLite: {e}")
        else:
            logger.info(" -> Skipping SQLite event logging (no connection).")

        self._add_event_node_to_graph(ticker, headline, score, link, timestamp_str)
        logger.info(f" -> ✅ Event node for {ticker} added to Neo4j.")

    def get_all_events(self) -> List[Dict[str, Any]]:
        """
        Retrieves ALL significant events from the Neo4j database.
        """
        if not self.is_connected(): 
            logger.error("Neo4j connection not available.")
            return []
        
        logger.info("Retrieving all historical events from Neo4j...")
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
        logger.info(f" -> Found {len(results)} total events in Neo4j.")
        return [_clean_properties(event) for event in results]

    def get_recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieves the most recent significant events from the Neo4j database.
        """
        if not self.is_connected(): return []
        
        logger.info(f"Retrieving {limit} recent events from Neo4j...")
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


    def get_all_events_from_sqlite(self) -> List[Dict[str, Any]]:
        """
        Retrieves ALL significant events from the LOCAL SQLITE database.
        """
        if not self.sqlite_conn: 
            logger.error("SQLite connection not available. Cannot backfill.")
            return []
        
        logger.info("Retrieving all historical events from SQLite for backfill...")
        try:
            with self.sqlite_conn as conn:
                cursor = conn.execute("SELECT * FROM significant_events ORDER BY id ASC")
                events = [dict(row) for row in cursor.fetchall()]
                logger.info(f" -> Found {len(events)} total events in SQLite.")
                return events
        except Exception as e:
            logger.error(f"Failed to retrieve all events from SQLite: {e}")
            return []

    def get_graph_from_db(self, weight_threshold: float = 0.1) -> nx.DiGraph:
        """
        Fetches the entire graph from Neo4j and converts it to a NetworkX DiGraph.
        """
        G = nx.DiGraph()
        if not self.is_connected():
            logger.error("get_graph_from_db: No database connection. Returning empty graph.")
            return G
            
        nodes_query = "MATCH (n:Company) RETURN n"
        edges_query = f"""
        MATCH (n:Company)-[r]->(m:Company)
        WHERE r.weight > $weight_threshold
        RETURN n.ticker AS source, m.ticker AS target, r
        """
        
        try:
            nodes_result = self.execute_read(nodes_query)
            for record in nodes_result:
                node_data = _clean_properties(record["n"]) 
                G.add_node(node_data['ticker'], **node_data)
                
            edges_result = self.execute_read(edges_query, weight_threshold=weight_threshold)
            for record in edges_result:
                rel_data = _clean_properties(record["r"]) 
                G.add_edge(record["source"], record["target"], **rel_data)
        except Exception as e:
            logger.error(f"Failed to get graph from Neo4j: {e}")
        return G

    def get_neighborhood_graph(self, company_ticker: str) -> nx.DiGraph:
        """
        Fetches a 1-hop neighborhood for a given company and converts to NetworkX.
        """
        G = nx.DiGraph()
        if not self.is_connected():
            logger.error("get_neighborhood_graph: No database connection.")
            return G
        
        query = """
        MATCH (c:Company {ticker: $ticker})-[r]-(neighbor:Company)
        WITH c, r, neighbor
        ORDER BY r.weight DESC
        LIMIT 25
        RETURN c, neighbor, r, 
               startNode(r).ticker AS source_ticker, 
               endNode(r).ticker AS target_ticker
        """
        
        try:
            result_list = self.execute_read(query, ticker=company_ticker)
            nodes_added = set()

            if not result_list:
                node_data_list = self.execute_read(
                    "MATCH (c:Company {ticker: $ticker}) RETURN c", ticker=company_ticker
                )
                if node_data_list:
                    node_data = _clean_properties(node_data_list[0]['c'])
                    G.add_node(node_data['ticker'], **node_data)
                return G 

            for record in result_list:
                center_node_data = _clean_properties(record['c'])
                neighbor_node_data = _clean_properties(record['neighbor'])
                rel_data = _clean_properties(record['r'])
                source_ticker = record['source_ticker']
                target_ticker = record['target_ticker']
                
                if center_node_data['ticker'] not in nodes_added:
                    G.add_node(center_node_data['ticker'], **center_node_data)
                    nodes_added.add(center_node_data['ticker'])
                    
                if neighbor_node_data['ticker'] not in nodes_added:
                    G.add_node(neighbor_node_data['ticker'], **neighbor_node_data)
                    nodes_added.add(neighbor_node_data['ticker'])
                
                G.add_edge(source_ticker, target_ticker, **rel_data)
                
        except Exception as e:
            logger.error(f"Failed to build neighborhood graph for {company_ticker}: {e}")

        return G

    def clear_neo4j_database(self):
        """
        !! DANGEROUS !! Deletes all nodes and relationships from the Neo4j database.
        """
        if not self.is_connected(): return
        logger.warning("🚨 DELETING ALL DATA from the Neo4j database...")
        self.execute_write("MATCH (n) DETACH DELETE n")
        logger.warning(" -> ✅ Neo4j database has been cleared.")

    def clear_neo4j_events(self):
        """
        Deletes all :Event nodes and their :HAD_EVENT relationships from Neo4j.
        """
        if not self.is_connected(): return
        logger.warning("🚨 DELETING ALL :Event nodes and :HAD_EVENT relationships from Neo4j...")
        self.execute_write("MATCH (e:Event) DETACH DELETE e")
        logger.warning(" -> ✅ Neo4j events have been cleared.")

    def close(self):
        """Closes all database connections."""
        if self.sqlite_conn:
            self.sqlite_conn.close()
            logger.info(" -> SQLite connection closed.")
        if self.neo4j_driver:
            self.neo4j_driver.close()
            logger.info(" -> Neo4j connection closed.")


    def add_events_batch(self, events: List[Dict[str, Any]]):
        """
        Adds a list of event dictionaries to Neo4j using UNWIND for efficient batch creation.
        
        This replaces the previous per-event write and ensures atomicity.
        """
        if not events:
            logger.warning("Attempted to add an empty batch of events.")
            return
        
        # Cypher uses UNWIND to iterate over the list of events passed as a parameter ($events)
        query = """
        UNWIND $events AS event
        MERGE (c:Company {ticker: event.ticker})
        ON CREATE SET c.name = event.ticker, c.sector = 'Unknown'

        MERGE (e:Event {
            headline: event.headline
        })
        ON CREATE SET 
            e.score = event.score, 
            e.link = event.link,
            e.timestamp = datetime()
            
        MERGE (e)-[:MENTIONS_COMPANY {score: event.score}]->(c)
        """
        
        # Neo4j best practice: Execute write operations inside execute_write
        # --- FIX: Use self.neo4j_driver instead of self.driver ---
        with self.neo4j_driver.session() as session:
            session.execute_write(
                lambda tx: tx.run(query, events=events)
            )
        # --- FIX: Use global 'logger', not self.logger ---
        logger.info(f"Successfully processed {len(events)} events in a single batch.")