# ==============================================================================
# --- SENTIMENT PULSE (Polygon.io Edition) ---
# ==============================================================================
# Polls Polygon.io News (Safe for Free Tier Limits), updates risk scores via 
# Gemini AI, and logs Events to the Graph.
#
# NOTE: This script runs 24/7 to catch breaking news outside market hours.
# During Market Hours (9:30-4:00 ET), the Alpaca Trading Bot should also be running.
# ==============================================================================

import json
import logging
import os
import time
import asyncio
from datetime import datetime, timedelta
import google.generativeai as genai
from polygon import RESTClient
from database_manager import DatabaseManager

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Pulse")

# --- CONFIGURATION ---
def load_config():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        with open(config_path, 'r') as f: return json.load(f)
    except: return {}

config = load_config()
GEMINI_KEY = config.get("GOOGLE_API_KEY")
POLYGON_KEY = config.get("polygon_api_key") # Ensure this is in your config.json

# --- AI SETUP ---
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash')
else:
    logger.warning("⚠️ No Gemini API Key found. Sentiment will be 0.0")
    model = None

# --- AI SCORING (Sync) ---
# --- UPDATED AI SCORING (With Relevance/Confidence) ---
def get_ai_analysis(headline, tickers):
    """
    Returns a tuple: (sentiment_score, relevance_score)
    """
    if not model: return 0.0, 0.0
    try:
        # We pass the tickers so the AI knows who the news SHOULD be about
        prompt = f"""
        Analyze this financial news headline for the specific tickers: {tickers}.
        Headline: "{headline}"

        Task:
        1. Calculate Sentiment Score (-1.0 Negative to 1.0 Positive).
        2. Calculate Relevance Score (0.0 to 1.0). 
           - Ask yourself: Is this news REALLY about the company {tickers}?
           - Example of LOW relevance: "Top cop retired" (Police) tagged for ticker "COP" (ConocoPhillips).
           - Example of LOW relevance: "ServiceNow brand voice" tagged for ticker "NOW".

        Return ONLY a JSON object with keys 'score' and 'relevance'.
        Example: {{"score": -0.5, "relevance": 0.1}}
        """
        
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text)
        
        return float(data.get("score", 0.0)), float(data.get("relevance", 0.0))
        
    except Exception as e:
        logger.error(f"AI Scoring Failed: {e}")
        return 0.0, 0.0

# --- MAIN LOOP ---
def main():
    if not POLYGON_KEY:
        logger.error("❌ Polygon API Key not found in config.json")
        return

    # Initialize Managers
    db = DatabaseManager(config)
    client = RESTClient(api_key=POLYGON_KEY)
    
    # Track seen articles to prevent duplicates
    seen_news_ids = set()
    
    logger.info("✅ Pulse Active (Polygon Mode). Polling every 25s.")
    logger.info("ℹ️  Confidence Filter Active: Events with < 50% relevance will be discarded.")

    while True:
        try:
            # 1. Fetch the generator (Limit 10 to catch bursts)
            news_generator = client.list_ticker_news(limit=10)
            
            # 2. Manually take only the first 10 items to avoid infinite pagination
            latest_batch = []
            try:
                for _ in range(10):
                    latest_batch.append(next(news_generator))
            except StopIteration:
                pass # Less than 10 items available
            
            # 3. Reverse to process oldest -> newest
            articles = reversed(latest_batch)

            for item in articles:
                # Deduplication check
                if item.id in seen_news_ids:
                    continue
                
                seen_news_ids.add(item.id)
                if len(seen_news_ids) > 1000: # Keep memory clean
                    seen_news_ids.pop()

                headline = item.title
                # Polygon gives tickers as a list ["AAPL", "MSFT"]
                tickers = item.tickers if item.tickers else []
                url = item.article_url

                if not tickers or not headline: 
                    continue

                # 4. Get Score AND Relevance (Metacognition)
                score, relevance = get_ai_analysis(headline, tickers)

                # 5. The Confidence Filter
                # If the AI thinks this is irrelevant noise (< 50% confidence), dump it.
                if relevance < 0.5:
                    logger.info(f"🗑️ REJECTED (Noise): {headline[:30]}... (Rel: {relevance:.2f})")
                    continue

                # 6. Filter Noise (Only log significant sentiment)
                if abs(score) >= 0.15:
                    logger.info(f"⚡ NEWS (Sc: {score} | Rel: {relevance}): {headline[:50]}... {tickers}")
                    
                    for ticker in tickers:
                        # Insert into SQLite (for UI)
                        db.insert_event(
                            ticker=ticker,
                            headline=headline,
                            score=score,
                            link=url
                        )

                        # Update Neo4j Graph
                        query = """
                        MATCH (c:Company {ticker: $ticker})
                        SET c.sentiment_score = $score,
                            c.last_news_update = datetime()
                        
                        CREATE (e:Event {
                            headline: $headline,
                            score: $score,
                            relevance: $relevance,
                            source: 'Polygon_Gemini',
                            timestamp: datetime(),
                            link: $url
                        })
                        MERGE (c)-[:HAD_EVENT]->(e)
                        """
                        db.execute_write(query, ticker=ticker, score=score, relevance=relevance, headline=headline, url=url)
            
            # 7. Rate Limit Wait (CRITICAL)
            time.sleep(25)

        except Exception as e:
            logger.error(f"Polling Error: {e}")
            time.sleep(25) # Wait before retrying

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Pulse stopped by user.")