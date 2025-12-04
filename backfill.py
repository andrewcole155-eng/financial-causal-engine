# ==============================================================================
# --- IMPORTS ---
# ==============================================================================
import json
import time
import re
import os
import sys
import logging
import smtplib
from datetime import datetime, timedelta
from typing import Dict, Any, List
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Third-party Imports
import feedparser
import spacy
import torch
import yfinance as yf
from polygon import RESTClient
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Local Imports
from database_manager import DatabaseManager

# ==============================================================================
# --- SETUP & CONFIGURATION (Shared with Worker) ---
# ==============================================================================

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("BackfillMaster")

def get_backfill_config() -> Dict[str, Any]:
    """
    Loads config prioritizing Env Vars > config.json > Defaults.
    """
    config = {
        "news_sentiment_threshold": 0.5, # Lower threshold for backfill to get more data
        "target_tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"],
        "backfill_limit": 50, # Articles per ticker
        "recipient_emails": [],
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "email_sender": "",
        "email_password": ""
    }

    # Load local JSON if exists
    try:
        if os.path.exists("config.json"):
            with open("config.json", "r", encoding='utf-8') as f:
                config.update(json.load(f))
    except Exception as e:
        logger.warning(f"⚠️ Error loading config.json: {e}")

    # Override with Env Vars
    env_map = {
        "NEO4J_URI": "neo4j_uri", "NEO4J_USER": "neo4j_user", "NEO4J_PASSWORD": "neo4j_password",
        "POLYGON_API_KEY": "polygon_api_key", "EMAIL_SERVER": "smtp_server", 
        "EMAIL_SENDER": "email_sender", "EMAIL_PASSWORD": "email_password"
    }
    for env_key, conf_key in env_map.items():
        if os.environ.get(env_key):
            config[conf_key] = os.environ.get(env_key)

    if os.environ.get("EMAIL_PORT"):
        try: config["smtp_port"] = int(os.environ.get("EMAIL_PORT"))
        except: pass

    if os.environ.get("RECIPIENT_EMAILS"):
        config["recipient_emails"] = [e.strip() for e in os.environ.get("RECIPIENT_EMAILS").split(",") if e.strip()]

    return config

# ==============================================================================
# --- NLP MODEL LOADING ---
# ==============================================================================

@torch.no_grad()
def get_financial_sentiment(text: str, tokenizer: Any, model: Any) -> float:
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # 0=Positive, 1=Negative, 2=Neutral
        return (scores[0][0] - scores[0][1]).item()
    except Exception:
        return 0.0

try:
    logger.info("🧠 Loading NLP models...")
    finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    nlp_spacy = spacy.load("en_core_web_sm")
    logger.info("✅ Models loaded.")
except Exception as e:
    logger.critical(f"FATAL: NLP Load Error: {e}")
    sys.exit(1)

# ==============================================================================
# --- DATA FETCHING (HYBRID: API + RSS) ---
# ==============================================================================

def fetch_polygon_history(ticker: str, limit: int, api_key: str) -> List[Dict]:
    """Fetches historical news via Polygon.io."""
    logger.info(f" 📡 Fetching history for {ticker} via Polygon API...")
    client = RESTClient(api_key)
    articles = []
    try:
        # Fetch news list
        response = client.list_ticker_news(ticker, limit=limit, order="desc", sort="published_utc")
        for item in response:
            articles.append({
                'title': item.title,
                'published': item.published_utc, # String ISO format
                'link': item.article_url,
                'summary': item.description
            })
    except Exception as e:
        logger.error(f"Polygon API failed for {ticker}: {e}")
    return articles

def fetch_rss_history(ticker: str) -> List[Dict]:
    """Fetches recent news via RSS (Fallback)."""
    logger.info(f" 📡 Fetching history for {ticker} via RSS Feeds...")
    # Load feed URLs or default
    feeds = ["http://feeds.marketwatch.com/marketwatch/topstories/"]
    try:
        if os.path.exists('news_urls.json'):
            with open('news_urls.json', 'r') as f:
                feeds = [s['url'] for s in json.load(f).get("sources", [])]
    except: pass

    articles = []
    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                # Basic filter to ensure relevance to ticker
                if re.search(r'\b' + re.escape(ticker) + r'\b', entry.title, re.IGNORECASE):
                    articles.append({
                        'title': entry.title,
                        'published': entry.get('published_parsed'), # Struct_time
                        'link': entry.link,
                        'summary': entry.get('summary', '')
                    })
        except: continue
    return articles

# ==============================================================================
# --- SECTOR REPAIR (From Worker) ---
# ==============================================================================
def clean_ticker_for_yahoo(ticker):
    if ":" in ticker: ticker = ticker.split(":")[-1]
    ticker = ticker.replace('.', '-')
    return 'BRK-B' if ticker == 'BRK' else ticker

def enrich_sectors_automatically(db_manager: DatabaseManager):
    """Fills in missing sector data using yfinance."""
    logger.info("🧹 checking for missing sector data...")
    try:
        query = "MATCH (c:Company) WHERE c.sector IN ['Discovered', 'Unknown', 'null', NULL] RETURN c.ticker as ticker"
        results = db_manager.execute_read(query)
        if not results: return

        tickers = [r['ticker'] for r in results]
        logger.info(f" -> Found {len(tickers)} nodes to enrich.")
        
        batch_size = 20
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            yahoo_map = {clean_ticker_for_yahoo(t): t for t in batch}
            try:
                tickers_obj = yf.Tickers(" ".join(yahoo_map.keys()))
                for y_sym, original_ticker in yahoo_map.items():
                    info = tickers_obj.tickers[y_sym].info
                    sector = info.get('sector')
                    industry = info.get('industry')
                    if sector:
                        db_manager.execute_write(
                            "MATCH (c:Company {ticker: $t}) SET c.sector = $s, c.industry = $i",
                            t=original_ticker, s=sector, i=industry
                        )
                        logger.info(f"    -> Fixed {original_ticker}: {sector}")
            except Exception: continue
            time.sleep(1)
    except Exception as e:
        logger.error(f"Sector enrichment failed: {e}")

# ==============================================================================
# --- MAIN BACKFILL LOGIC ---
# ==============================================================================

def run_backfill_process():
    config = get_backfill_config()
    db_manager = DatabaseManager(config)
    
    if not db_manager.is_connected():
        logger.critical("❌ DB Connection failed. Exiting.")
        return

    target_tickers = config.get("target_tickers", [])
    limit = config.get("backfill_limit", 50)
    api_key = config.get("polygon_api_key")

    logger.info("🚀 STARTING BACKFILL PROCESS")
    logger.info(f"🎯 Targets: {target_tickers}")
    
    all_processed_events = []
    
    for ticker in target_tickers:
        raw_articles = []
        
        # 1. Fetch Data (Priority: API > RSS)
        if api_key:
            raw_articles = fetch_polygon_history(ticker, limit, api_key)
        
        if not raw_articles:
            if api_key: logger.warning(f"Polygon found no data for {ticker}, trying RSS fallback.")
            raw_articles = fetch_rss_history(ticker)

        logger.info(f" -> Found {len(raw_articles)} raw items for {ticker}.")

        # 2. Process & Analyze
        batch_events = []
        now = datetime.now()

        for i, item in enumerate(raw_articles):
            title = item['title']
            
            # --- Timestamp Logic (Robust) ---
            raw_date = item.get('published')
            ts_dt = now # Default
            
            try:
                if isinstance(raw_date, str):
                    # Try ISO format (Polygon)
                    ts_dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                elif isinstance(raw_date, time.struct_time):
                    # Try Struct Time (RSS)
                    ts_dt = datetime.fromtimestamp(time.mktime(raw_date))
                elif raw_date is None:
                    # Fallback: Stagger time to preserve order
                    minutes_ago = (len(raw_articles) - i) * 10
                    ts_dt = now - timedelta(minutes=minutes_ago)
            except:
                ts_dt = now

            # Sentiment
            score = get_financial_sentiment(title, finbert_tokenizer, finbert_model)

            # Filter
            if abs(score) >= config["news_sentiment_threshold"]:
                event_obj = {
                    'ticker': ticker,
                    'headline': title,
                    'score': score,
                    'link': item['link'],
                    'timestamp': ts_dt
                }
                batch_events.append(event_obj)
                print(f"    ✅ [{score:.2f}] {title[:50]}...")

        # 3. Write Batch to DB
        if batch_events:
            # Write to Neo4j
            db_manager.add_events_batch(batch_events)
            
            # Write to SQLite
            for evt in batch_events:
                db_manager.insert_event(
                    evt['ticker'], evt['headline'], evt['score'], evt['link'], evt['timestamp']
                )
            
            all_processed_events.extend(batch_events)
            logger.info(f" -> 💾 Saved {len(batch_events)} events for {ticker}.")
        
        logger.info("⏳ Sleeping 15s to respect Polygon API rate limits...")
        time.sleep(15) # Be polite to APIs

    # 4. Post-Processing: Fix Sectors
    enrich_sectors_automatically(db_manager)

    # 5. Alerting
    if all_processed_events and config.get("recipient_emails"):
        logger.info("📧 Sending completion email...")
        try:
            # Reusing the simple HTML structure, simplified here
            msg_body = f"<h2>Backfill Complete</h2><p>Processed {len(all_processed_events)} historical events.</p>"
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"Backfill Report: {len(all_processed_events)} Events Added"
            msg['From'] = config['email_sender']
            msg['To'] = ", ".join(config['recipient_emails'])
            msg.attach(MIMEText(msg_body, 'html'))

            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                server.starttls()
                server.login(config['email_sender'], config['email_password'])
                server.send_message(msg)
            logger.info("✅ Email sent.")
        except Exception as e:
            logger.error(f"Email failed: {e}")

    db_manager.close()
    logger.info("✅ BACKFILL COMPLETE.")

if __name__ == "__main__":
    run_backfill_process()