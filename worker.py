# ==============================================================================
# --- IMPORTS for Background Worker ---
# ==============================================================================
import json
import time
import re
import os
import glob
import logging
import sys
import smtplib
import yfinance as yf
from typing import Dict, Any, List
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime
from google import genai
from google.genai import types

# Third-party Imports
import feedparser
import spacy
import torch
from bs4 import BeautifulSoup
from polygon import RESTClient
from sec_edgar_downloader import Downloader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Local Imports
from database_manager import DatabaseManager

# ==============================================================================
# --- WORKER SETUP & CONFIGURATION ---
# ==============================================================================

# Setup professional logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout) # Ensure logs go to console/Docker logs
    ]
)
logger = logging.getLogger("FinancialWorker")

def get_worker_config() -> Dict[str, Any]:
    """
    Robust configuration loader.
    1. Defines defaults (prevents crashes).
    2. Loads config.json (if exists locally).
    3. Overrides with Environment Variables (for GitHub Actions/Docker).
    """
    # 1. Defaults
    config = {
        "news_sentiment_threshold": 0.7,
        "target_tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK.B", "LLY", "V"],
        "downloader_company": "OpenSourceProject",
        "downloader_email": "worker@example.com",
        "recipient_emails": [],
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "email_sender": "",
        "email_password": ""
    }

    # 2. Try loading local config.json (Merge if exists)
    try:
        if os.path.exists("config.json"):
            with open("config.json", "r", encoding='utf-8') as f:
                local_conf = json.load(f)
                config.update(local_conf)
                logger.info("📂 Loaded configuration from config.json")
    except Exception as e:
        logger.warning(f"⚠️ Error loading config.json: {e}. Proceeding with Defaults + Env Vars.")

    # 3. Override with Environment Variables (Docker/Cloud)
    # Database
    if os.environ.get("NEO4J_URI"):
        config["neo4j_uri"] = os.environ.get("NEO4J_URI")
    if os.environ.get("NEO4J_USER"):
        config["neo4j_user"] = os.environ.get("NEO4J_USER")
    if os.environ.get("NEO4J_PASSWORD"):
        config["neo4j_password"] = os.environ.get("NEO4J_PASSWORD")

    # APIs
    if os.environ.get("POLYGON_API_KEY"):
        config["polygon_api_key"] = os.environ.get("POLYGON_API_KEY")
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if api_key:
        config["google_api_key"] = api_key

    # Email - Server Config
    if os.environ.get("EMAIL_SERVER"):
        config["smtp_server"] = os.environ.get("EMAIL_SERVER")
    
    if os.environ.get("EMAIL_PORT"):
        try:
            config["smtp_port"] = int(os.environ.get("EMAIL_PORT"))
        except ValueError:
            logger.warning(f"⚠️ Invalid EMAIL_PORT '{os.environ.get('EMAIL_PORT')}'. Defaulting to {config['smtp_port']}.")

    # Email - Credentials
    if os.environ.get("EMAIL_SENDER"):
        config["email_sender"] = os.environ.get("EMAIL_SENDER")
    if os.environ.get("EMAIL_PASSWORD"):
        config["email_password"] = os.environ.get("EMAIL_PASSWORD")
    
    # Email - Recipients
    if os.environ.get("RECIPIENT_EMAILS"):
        # 1. Read the secret string (RECIPIENT_EMAILS)
        raw_recipients = os.environ.get("RECIPIENT_EMAILS")
        # 2. Split by comma and strip spaces to create the list (recipient_emails)
        config["recipient_emails"] = [email.strip() for email in raw_recipients.split(",") if email.strip()]

    return config

# ==============================================================================
# --- AI MODEL SETUP (Gemini) ---
# ==============================================================================
# Load config temporarily to get key for global setup
_temp_config = get_worker_config()
GOOGLE_KEY = _temp_config.get("GOOGLE_API_KEY") 

if GOOGLE_KEY:
    # UPDATED: Initialize the Client object (New SDK Syntax)
    gemini_client = genai.Client(api_key=GOOGLE_KEY)
    logger.info("✅ Gemini AI Client initialized for Relevance Analysis.")
else:
    gemini_client = None
    logger.warning("⚠️ No GOOGLE_API_KEY found. Worker will revert to basic FinBERT.")

# ==============================================================================
# --- GLOBAL NLP MODEL LOADING ---
# ==============================================================================

@torch.no_grad()
def get_financial_sentiment(text: str, tokenizer: Any, model: Any) -> float:
    """Analyzes text using a pre-loaded FinBERT model."""
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # FinBERT: 0=positive, 1=negative, 2=neutral
        # Result is Positive Score minus Negative Score
        return (scores[0][0] - scores[0][1]).item()
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {e}")
        return 0.0

# Load models once at startup
try:
    logger.info("Loading NLP models into memory...")
    finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    nlp_spacy = spacy.load("en_core_web_sm")
    logger.info("✅ NLP models loaded successfully.")
except Exception as e:
    logger.critical(f"FATAL: Could not load essential NLP models. Error: {e}")
    sys.exit(1)

# ==============================================================================
# --- ALERTING FUNCTION ---
# ==============================================================================

def generate_summary_email_body(events: list) -> str:
    """
    Generates a single HTML string to summarize a list of significant events.
    """
    header = """
    <html>
    <head>
    <style>
        body { font-family: sans-serif; margin: 20px; color: #333; }
        .container { border: 1px solid #ddd; border-radius: 8px; padding: 20px; max-width: 800px; margin: auto; }
        h2 { color: #2c3e50; }
        .event { border-top: 1px solid #eee; padding-top: 15px; margin-top: 15px; }
        .event p { margin: 5px 0; }
        .event-header { font-size: 1.1em; font-weight: bold; }
        .positive { color: #27ae60; }
        .negative { color: #c0392b; }
        a { color: #3498db; text-decoration: none; }
        a:hover { text-decoration: underline; }
        small { color: #7f8c8d; }
    </style>
    </head>
    <body>
    <div class="container">
    """
    
    footer = """
    <hr>
    <p><small>This is an automated summary from the Financial Causal Inference Engine.</small></p>
    </div>
    </body>
    </html>
    """

    event_html_parts = []
    for event in events:
        score_color_class = "positive" if event['score'] > 0 else "negative"
        event_html_parts.append(f"""
        <div class="event">
            <p class="event-header">
                <span class="{score_color_class}">{event['ticker']}</span>: {event['headline']}
            </p>
            <p>
                <strong>Sentiment Score:</strong> 
                <strong class="{score_color_class}">{event['score']:.2f}</strong>
            </p>
            <p><a href="{event['link']}">Read Full Article</a></p>
        </div>
        """)

    body = f"<h2>{len(events)} Significant Events Detected</h2><p>The following events met the sentiment threshold during this cycle.</p>"
    
    return header + body + "".join(event_html_parts) + footer

def send_email_alert(config: Dict[str, Any], subject: str, body_html: str):
    """Sends a formatted HTML email alert using the STARTTLS method."""
    # We check if recipient_emails is empty OR if critical keys are missing
    if not config.get('recipient_emails'):
        logger.warning("📧 No recipients configured in environment variables. Skipping alert.")
        return

    required_keys = ['email_sender', 'email_password', 'smtp_server', 'smtp_port']
    if not all(config.get(k) for k in required_keys):
        logger.warning("📧 Email configuration missing (Sender, Password, or Server). Skipping alert.")
        return

    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = config['email_sender']
        msg['To'] = ", ".join(config['recipient_emails'])
        msg.attach(MIMEText(body_html, 'html'))
        
        logger.info(f"Connecting to SMTP server {config['smtp_server']}:{config['smtp_port']}...")
        
        with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
            server.starttls()  
            server.login(config['email_sender'], config['email_password'])
            server.send_message(msg)
            
        logger.info("✅ Email alert sent successfully.")
    except Exception as e:
        logger.error(f"Could not send email alert. Reason: {e}")

def get_ai_analysis(headline, tickers):
    """
    Uses Gemini to determine:
    1. Sentiment (-1 to 1)
    2. Relevance (0 to 1) - Detecting False Positives (Metacognition)
    """
    if not gemini_client: 
        return 0.0, 0.0
    
    try:
        # --- IMPLEMENTING LOGIC FROM SCREENSHOT ---
        prompt = f"""
        Analyze this financial news headline for the specific tickers: {tickers}.
        Headline: "{headline}"

        Task:
        1. Calculate Sentiment Score (-1.0 Negative to 1.0 Positive).
        2. Calculate Relevance Score (0.0 to 1.0). 
           - CRITICAL: Ask yourself, is this news REALLY about the company {tickers}?
           - 0.0 (False Positive): Ticker appears as a common word (e.g. 'COP' the police vs 'COP' the oil company, 'NOW' vs ServiceNow, 'L' vs Loews).
           - 1.0 (Relevant): The news explicitly mentions the company or its products.

        Return ONLY a JSON object with keys 'score' and 'relevance'.
        """
        
        # UPDATED: Call using the new 'config' parameter for strictly structured JSON
        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        # Parse the response (guaranteed to be JSON by the model config)
        data = json.loads(response.text)
        
        # Return both values
        return float(data.get("score", 0.0)), float(data.get("relevance", 0.0))
        
    except Exception as e:
        logger.error(f"Gemini Analysis Failed: {e}")
        return 0.0, 0.0

# ==============================================================================
# --- DATA PIPELINE FUNCTIONS ---
# ==============================================================================

def check_live_news_for_events(db_manager: DatabaseManager, config: Dict[str, Any]):
    """
    Fetches live news, collects significant events with PRECISE TIMESTAMPS, and saves them.
    """
    logger.info("📰 Checking live news for significant events...")
    
    # Load feed URLs
    try:
        with open('news_urls.json', 'r', encoding='utf-8') as f:
            feed_urls = [s['url'] for s in json.load(f).get("sources", [])]
    except Exception as e:
        logger.error(f"Could not load news_urls.json: {e}")
        feed_urls = ["http://feeds.marketwatch.com/marketwatch/topstories/"] 

    articles = []
    for url in feed_urls:
        try:
            feed = feedparser.parse(url, agent='Mozilla/5.0')
            if feed.entries: articles.extend(feed.entries)
        except Exception as e:
            logger.warning(f"Could not fetch or parse feed from {url}. Reason: {e}")

    if not articles:
        logger.info("-> No new articles found in any feeds.")
        return

    try:
        graph = db_manager.get_graph_from_db()
        all_tickers = list(graph.nodes())
        
        if not all_tickers:
             all_tickers = config.get("target_tickers", [])

        sentiment_threshold = config.get("news_sentiment_threshold", 0.7)
        
        # Get recent headlines to skip duplicates
        recent_events = db_manager.get_recent_events(limit=500)
        processed_headlines = set(event['headline'] for event in recent_events)
        
        ticker_stopwords = {'A', 'ON', 'IT', 'HAS', 'SO', 'D', 'BE', 'ARE', 'SEE', 'CAN', 'OR'} 

        significant_events_found = []
        
        for article in articles:
            title = article.get('title', 'No Title')
            if title in processed_headlines: continue

            link = article.get('link', '#')
            
            # --- TRUTH LAYER UPDATE: Capture Exact Timestamp ---
            # We need the exact publication time to align with the price chart later.
            pub_struct = article.get('published_parsed', article.get('updated_parsed'))
            if pub_struct:
                event_dt = datetime.fromtimestamp(time.mktime(pub_struct))
            else:
                event_dt = datetime.now()

            for ticker in all_tickers:
                if ticker in ticker_stopwords: continue

                # Strict Regex Match to find ticker in headline
                if re.search(r'\b' + re.escape(ticker) + r'\b', title, re.IGNORECASE):
                    
                    # 1. Decide which model to use
                    if gemini_client:
                        # USE GEMINI (Metacognition Enabled)
                        score, relevance = get_ai_analysis(title, [ticker])
                        
                        # --- IMPLEMENTING ACTION FROM SCREENSHOT ---
                        # "Action: If Relevance < 50%, discard the event entirely."
                        if relevance < 0.5:
                            logger.info(f"🗑️ REJECTED (Noise): {title[:30]}... for {ticker} (Rel: {relevance:.2f})")
                            continue # Skip this loop iteration, do not save to DB
                    else:
                        # Fallback for FinBERT (No relevance check available)
                        score = get_financial_sentiment(title, finbert_tokenizer, finbert_model)
                        relevance = 1.0 
                    
                    # 2. Check Sentiment Threshold (Existing logic)
                    if abs(score) >= sentiment_threshold:
                        logger.info(f"🚨 Significant event: {ticker} | Score: {score:.2f} | Rel: {relevance:.2f} | {title}")
                        
                        processed_headlines.add(title)
                        
                        significant_events_found.append({
                            'ticker': ticker, 
                            'headline': title,
                            'score': score, 
                            'relevance': relevance, # <--- Added Relevance
                            'link': link,
                            'timestamp': event_dt
                        })
                        break 
            # --- MODIFIED LOGIC END ---

        if significant_events_found:
            logger.info(f"✍️ DB-WRITE: Writing batch of {len(significant_events_found)} new events...")
            try:
                # 1. WRITE TO GRAPH (Neo4j)
                db_manager.add_events_batch(significant_events_found)
                
                # 2. WRITE TO UI LIST (SQLite)
                count_sqlite = 0
                for event in significant_events_found:
                    # NOTE: Ensure your insert_event method accepts 'timestamp'
                    # If not, update your database_manager.py to handle it.
                    db_manager.insert_event(
                        ticker=event['ticker'], 
                        headline=event['headline'], 
                        score=event['score'], 
                        link=event['link'],
                        timestamp=event['timestamp'] # Passing the timestamp
                    )
                    count_sqlite += 1
                
                logger.info(f"✅ DB-WRITE: Wrote {len(significant_events_found)} to Neo4j and {count_sqlite} to SQLite.")

                subject = f"Financial KG Summary: {len(significant_events_found)} Significant Events Detected"
                body_html = generate_summary_email_body(significant_events_found)
                send_email_alert(config, subject, body_html)
                
            except Exception as e:
                logger.critical(f"FATAL DB WRITE ERROR: Could not write event batch. Reason: {e}", exc_info=True)
        else:
            logger.info("-> No new significant events found to write.")
            
    except Exception as e:
        logger.error(f"Error during news check execution: {e}")
        
    logger.info("✅ Live news check complete.")

def update_nodes_from_api(tickers: List[str], client: RESTClient, db_manager: DatabaseManager):
    """Fetches company data from the API and upserts it to the database in a batch."""
    if not tickers:
        logger.warning("No tickers provided to update_nodes_from_api. Skipping.")
        return

    logger.info("🔄 Updating company nodes from Polygon API...")
    nodes_to_update = []
    for ticker in tickers:
        try:
            resp = client.get_ticker_details(ticker)
            nodes_to_update.append({
                "ticker": ticker,
                "name": getattr(resp, 'name', 'N/A'),
                "sector": getattr(resp, 'sic_description', 'N/A'),
                "market_cap": getattr(resp, 'market_cap', 0)
            })
            logger.info(f"  -> Fetched data for {ticker}")
        except Exception as e:
            logger.error(f"  -> Could not fetch data for {ticker}. Reason: {e}")
        time.sleep(13) # Rate limit compliant
    
    if nodes_to_update:
        db_manager.upsert_company_nodes_batch(nodes_to_update)
    logger.info("✅ Company nodes update complete.")


def add_manual_relationships(db_manager: DatabaseManager):
    """Adds a predefined set of high-conviction relationships to the database."""
    logger.info("🔗 Adding/updating manual relationships in the database...")
    edges = [
        ("KO", "KR", {"type": "dependency", "weight": 0.9, "data_source": "manual"}),
        ("OXY", "KO", {"type": "dependency", "weight": 0.6, "data_source": "manual"}),
        ("INTC", "AMD", {"type": "competitor", "weight": 0.8, "data_source": "manual"}),
        ("AMD", "INTC", {"type": "competitor", "weight": 0.8, "data_source": "manual"}),
        ("OXY", "INTC", {"type": "dependency", "weight": 0.7, "data_source": "manual"})
    ]
    for u, v, attrs in edges:
        db_manager.upsert_relationship(u, v, attrs.get('type', 'RELATED'), attrs)
    logger.info(f"  -> ✅ Wrote {len(edges)} manual relationships to the database.")

# ==============================================================================
# --- SEC SECTION ---
# ==============================================================================
def discover_relations_from_filings(tickers: List[str], db_manager: DatabaseManager, dl: Downloader):
    """Downloads and parses SEC filings to discover new relationships."""
    logger.info("🧠 Discovering relationships from SEC filings...")
    
    logger.info("  -> Downloading recent 10-K filings...")
    for ticker in tickers:
        try:
            # Note: 'download_details' deprecated in newer versions, check your library version
            dl.get("10-K", ticker, limit=1, download_details=False)
        except Exception as e:
            logger.error(f"    -> Failed to download 10-K for {ticker}: {e}")

    # Load mapping
    try:
        with open('sp500_map.json', 'r', encoding='utf-8') as f:
            company_map = {item['ticker']: item['name'] for item in json.load(f) if 'name' in item}
    except FileNotFoundError:
        logger.error("  -> 'sp500_map.json' not found. Skipping SEC entity matching.")
        return

    # --- BATCHING LOGIC START ---
    all_new_relationships = []
    all_new_nodes_map = {} 
    collected_pairs_global = set() 
    # --- BATCHING LOGIC END ---

    for ticker in tickers:
        search_path = os.path.join('sec-edgar-filings', ticker, '10-K', '*', 'full-submission.txt')
        filing_paths = glob.glob(search_path)
        if not filing_paths: continue

        logger.info(f"--- Processing filing for {ticker} ---")
        try:
            with open(filing_paths[0], 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'lxml')
                full_text = soup.get_text(separator=' ', strip=True)
            
            if not full_text or len(full_text) < 1000: continue
            
            doc = nlp_spacy(full_text[:nlp_spacy.max_length])

            for sent in doc.sents:
                org_entities = [ent.text.lower() for ent in sent.ents if ent.label_ == 'ORG']
                if not org_entities: continue

                for related_ticker, name in company_map.items():
                    if ticker == related_ticker: continue
                    
                    pair = tuple(sorted((ticker, related_ticker)))
                    if pair in collected_pairs_global: continue

                    name_lower = name.lower()
                    if any(org in name_lower or name_lower in org for org in org_entities):
                        sentiment = get_financial_sentiment(sent.text, finbert_tokenizer, finbert_model)
                        weight = min(1.0, 0.5 + (0.2 * sentiment))
                        attrs = {"type": "sec_discovered", "weight": round(weight, 2), "sentiment": round(sentiment, 2)}
                        
                        if related_ticker not in all_new_nodes_map:
                            all_new_nodes_map[related_ticker] = {
                                "ticker": related_ticker, "name": name, "sector": "Discovered", "market_cap": 0
                            }

                        all_new_relationships.append((ticker, related_ticker, "SEC_DISCOVERED", attrs))
                        logger.info(f"  -> ✅ DISCOVERED: Relationship {ticker} -> {related_ticker} (Weight: {weight:.2f})")
                        collected_pairs_global.add(pair)
                        
        except Exception as e:
            logger.error(f"  -> Failed to process filing for {ticker}. Reason: {e}")

    # --- BATCHING LOGIC (WRITE) ---
    if all_new_nodes_map:
        nodes_list = list(all_new_nodes_map.values())
        logger.info(f"  -> ✍️ DB-WRITE: Upserting batch of {len(nodes_list)} newly discovered company nodes...")
        db_manager.upsert_company_nodes_batch(nodes_list)
    else:
        logger.info("  -> No new company nodes were discovered.")

    if all_new_relationships:
        logger.info(f"  -> ✍️ DB-WRITE: Upserting {len(all_new_relationships)} newly discovered unique relationships...")
        for u, v, rel_type, attrs in all_new_relationships:
            db_manager.upsert_relationship(u, v, rel_type, attrs)
        logger.info(f"  -> ✅ DB-WRITE: Wrote {len(all_new_relationships)} unique new relationships.")
    else:
        logger.info("  -> No new relationships were discovered.")

    logger.info("✅ SEC relationship discovery complete.")

# ==============================================================================
# --- SECTOR ENRICHMENT (AUTO-REPAIR) ---
# ==============================================================================
def clean_ticker_for_yahoo(ticker):
    """Converts DB format to Yahoo format (e.g. BRK.B -> BRK-B)."""
    if ":" in ticker:
        ticker = ticker.split(":")[-1]
    ticker = ticker.replace('.', '-')
    if ticker == 'BRK': return 'BRK-B'
    return ticker

def enrich_sectors_automatically(db_manager: DatabaseManager):
    """
    Finds nodes with 'Discovered' or 'Unknown' sectors and fixes them using yfinance.
    Run this at the end of the pipeline to ensure the graph is clean.
    """
    logger.info("🧹 SECTOR CLEANUP: Checking for companies with missing sector data...")
    
    # 1. Find the problem nodes
    query = """
    MATCH (c:Company) 
    WHERE c.sector IN ['Discovered', 'Unknown', 'null'] OR c.sector IS NULL
    RETURN c.ticker as ticker
    """
    results = db_manager.execute_read(query)
    tickers = [r['ticker'] for r in results]
    
    if not tickers:
        logger.info("   -> ✅ No missing sectors found. Database is clean.")
        return

    logger.info(f"   -> 📉 Found {len(tickers)} companies to fix. Fetching data from Yahoo Finance...")
    
    updated_count = 0
    batch_size = 20
    
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        yahoo_map = {clean_ticker_for_yahoo(t): t for t in batch}
        yahoo_list = list(yahoo_map.keys())
        
        try:
            # Fetch batch data
            tickers_obj = yf.Tickers(" ".join(yahoo_list))
            
            for y_sym in yahoo_list:
                try:
                    info = tickers_obj.tickers[y_sym].info
                    sector = info.get('sector')
                    industry = info.get('industry')
                    
                    if sector:
                        original_ticker = yahoo_map[y_sym]
                        
                        # Update Neo4j
                        update_query = """
                        MATCH (c:Company {ticker: $ticker})
                        SET c.sector = $sector,
                            c.industry = $industry
                        """
                        db_manager.execute_write(update_query, ticker=original_ticker, sector=sector, industry=industry)
                        updated_count += 1
                        logger.info(f"      -> Fixed {original_ticker}: {sector}")
                except Exception:
                    continue
                    
        except Exception as e:
            logger.error(f"      -> Batch failed: {e}")
        
        time.sleep(1) # Polite delay
        
    logger.info(f"✅ SECTOR CLEANUP: Repaired {updated_count} companies.")



# ==============================================================================
# --- MAIN WORKER ORCHESTRATION ---
# ==============================================================================

def run_full_data_pipeline():
    """The main function for the worker, orchestrating all data tasks."""
    # Use the robust config loader
    config = get_worker_config()
    
    db_manager = DatabaseManager(config)
    if not db_manager.is_connected():
        logger.critical("Cannot run pipeline without a valid database connection.")
        return

    # Check for API key before starting
    api_key = config.get("polygon_api_key")
    if not api_key:
        logger.warning("⚠️ No POLYGON_API_KEY found. Skipping API updates.")
        client = None
    else:
        client = RESTClient(api_key)
    
    dl = Downloader(config.get("downloader_company"), config.get("downloader_email"))
    tickers = config.get("target_tickers", [])
    
    logger.info("\n" + "="*60 + f"\n🚀 STARTING DATA REFRESH PIPELINE\n" + "="*60)
    try:
        # 1. Update Core Node Data (Polygon)
        if client:
            update_nodes_from_api(tickers, client, db_manager)
        
        # 2. Add Hardcoded Relationships
        add_manual_relationships(db_manager)
        
        # 3. Discovery (Creates 'Discovered' nodes)
        discover_relations_from_filings(tickers, db_manager, dl)
        
        # 4. *** NEW *** Auto-Repair Sectors (Fixes 'Discovered' nodes)
        enrich_sectors_automatically(db_manager)

        # 5. News & Sentiment
        check_live_news_for_events(db_manager, config)

        # 6. Calculate Company Risk
        db_manager.update_company_risk_scores()

        # 7. Garbage Collection (Prune old news)
        db_manager.prune_old_events(days=90)
        
    except Exception as e:
        logger.critical(f"A critical error occurred during the pipeline execution: {e}", exc_info=True)
    finally:
        db_manager.close() # Close database connection
    
    logger.info("\n" + "="*60 + f"\n✅ PIPELINE COMPLETE\n" + "="*60 + "\n")

if __name__ == "__main__":
    # This script is designed to be run once by a scheduler (like cron).
    run_full_data_pipeline()