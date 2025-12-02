# ==============================================================================
# --- SENTIMENT PULSE (Real-Time News Stream) ---
# ==============================================================================
# Listens to Alpaca's News Feed, updates risk scores, and logs Events to the Graph.
# ==============================================================================

import json
import logging
import os
import asyncio
from alpaca.data.live import NewsDataStream
from alpaca.common.exceptions import APIError
from database_manager import DatabaseManager

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Pulse")

def load_config():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "config.json")
        with open(config_path, 'r') as f: return json.load(f)
    except: return {}

# --- DUMMY AI SCORER ---
# In production, this would call Gemini or FinBERT.
# Uses simple keyword matching for speed/demo purposes.
def calculate_sentiment(headline):
    headline = headline.lower()
    if any(w in headline for w in ["crash", "plunge", "lawsuit", "ban", "misses", "hacked", "investigation", "fraud"]):
        return -0.8  # High Risk
    if any(w in headline for w in ["soars", "record", "beat", "approved", "partnership", "merger", "upgrade"]):
        return 0.8   # Low Risk / Bullish
    return 0.0

async def main():
    config = load_config()
    if not config: return

    api_key = config.get("ALPACA_API_KEY")
    secret_key = config.get("ALPACA_SECRET_KEY")
    
    if not api_key or not secret_key:
        logger.error("❌ Alpaca Keys not found in config.json")
        return

    # Initialize Database Connection
    db = DatabaseManager(config)

    # Initialize Alpaca News Stream
    logger.info("🔌 Connecting to Alpaca News Stream...")
    try:
        news_stream = NewsDataStream(api_key, secret_key)
    except Exception as e:
        logger.error(f"Failed to connect to Alpaca: {e}")
        return

    # --- THE HANDLER ---
    # This function runs every time a new headline hits the tape
    async def on_news(data):
        try:
            # Alpaca returns an object, we convert to dict
            news_item = data.__dict__
            
            headline = news_item.get('headline', '')
            symbols = news_item.get('symbols', [])
            url = news_item.get('url', 'Real-Time Stream') # Alpaca often provides a URL
            
            if not symbols: return

            # 1. Calculate Score
            score = calculate_sentiment(headline)
            
            # Only process significant news
            if score != 0:
                logger.info(f"⚡ NEWS ({score}): {headline} {symbols}")
                
                # 2. Update Graph (Create Event + Update Risk)
                for ticker in symbols:
                    query = """
                    MATCH (c:Company {ticker: $ticker})
                    
                    // 1. Update the Company Risk Profile
                    SET c.sentiment_score = $score,
                        c.last_news_update = datetime()
                        
                    // 2. Create a Visible Event Node (So it shows in Dashboard)
                    CREATE (e:Event {
                        headline: $headline,
                        score: $score,
                        source: 'Alpaca_Pulse',
                        timestamp: datetime(),
                        link: $url
                    })
                    
                    // 3. Link them
                    MERGE (c)-[:HAD_EVENT]->(e)
                    """
                    db.execute_write(query, ticker=ticker, score=score, headline=headline, url=url)
                    
        except Exception as e:
            logger.error(f"Error processing news: {e}")

    # Subscribe to all news ("*")
    news_stream.subscribe_news(on_news, "*")

    logger.info("✅ Pulse is Active. Waiting for news...")
    
    # Keep running forever
    await news_stream._run_forever()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Pulse stopped by user.")