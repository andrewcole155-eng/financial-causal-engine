# ==============================================================================
# --- SENTIMENT PULSE (Real-Time News Stream) ---
# ==============================================================================
# Listens to Alpaca's News Feed, updates risk scores via Gemini AI, 
# and logs Events to the Graph.
# ==============================================================================

import json
import logging
import os
import asyncio
import google.generativeai as genai
from alpaca.data.live import NewsDataStream
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

# --- AI CONFIGURATION ---
config = load_config()
GEMINI_KEY = config.get("GEMINI_API_KEY")

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    # Using 'flash' for speed/cost efficiency on high-volume streams
    model = genai.GenerativeModel('gemini-2.0-flash')
else:
    logger.warning("⚠️ No Gemini API Key found. Sentiment will be 0.0")
    model = None

# --- REAL SCORING SYSTEM (Gemini AI) ---
def get_ai_score_sync(headline):
    """
    Blocking function to query Gemini. 
    Returns a float between -1.0 (Bearish) and 1.0 (Bullish).
    """
    if not model: 
        return 0.0
    
    try:
        prompt = f"""
        Analyze the sentiment of this financial news headline.
        Headline: "{headline}"
        
        Return ONLY a JSON object with a single key 'score'.
        The score must be a float between -1.0 (Catastrophic/Negative) and 1.0 (Excellent/Positive).
        0.0 is neutral.
        Example output: {{"score": -0.65}}
        """
        
        response = model.generate_content(prompt)
        
        # Clean up response to ensure it's valid JSON
        text_response = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(text_response)
        return float(data.get("score", 0.0))

    except Exception as e:
        logger.error(f"AI Scoring Failed: {e}")
        return 0.0

async def calculate_sentiment(headline):
    """
    Wraps the blocking AI call in an executor to prevent 
    blocking the Alpaca Websocket heartbeat.
    """
    loop = asyncio.get_running_loop()
    # Run the synchronous API call in a separate thread
    score = await loop.run_in_executor(None, get_ai_score_sync, headline)
    return score

async def main():
    if not config: return

    alpaca_key = config.get("ALPACA_API_KEY")
    alpaca_secret = config.get("ALPACA_SECRET_KEY")
    
    if not alpaca_key or not alpaca_secret:
        logger.error("❌ Alpaca Keys not found in config.json")
        return

    # Initialize Database Connection
    db = DatabaseManager(config)

    # Initialize Alpaca News Stream
    logger.info("🔌 Connecting to Alpaca News Stream...")
    try:
        news_stream = NewsDataStream(alpaca_key, alpaca_secret)
    except Exception as e:
        logger.error(f"Failed to connect to Alpaca: {e}")
        return

    # --- THE HANDLER ---
    async def on_news(data):
        try:
            news_item = data.__dict__
            
            headline = news_item.get('headline', '')
            symbols = news_item.get('symbols', [])
            url = news_item.get('url', 'Real-Time Stream')
            
            if not symbols or not headline: return

            # 1. Calculate Score (Now Async with Real AI)
            # We ignore generic updates to save API costs/latency
            if "market update" in headline.lower(): 
                return

            score = await calculate_sentiment(headline)
            
            # Filter: Only log significant sentiment (Noise Filter)
            if abs(score) >= 0.2:
                logger.info(f"⚡ NEWS ({score}): {headline} {symbols}")
                
                # 2. Update Graph
                for ticker in symbols:
                    query = """
                    MATCH (c:Company {ticker: $ticker})
                    SET c.sentiment_score = $score,
                        c.last_news_update = datetime()
                        
                    CREATE (e:Event {
                        headline: $headline,
                        score: $score,
                        source: 'Alpaca_Gemini',
                        timestamp: datetime(),
                        link: $url
                    })
                    
                    MERGE (c)-[:HAD_EVENT]->(e)
                    """
                    # Note: db.execute_write is usually synchronous in Neo4j driver. 
                    # If high volume, wrap this in run_in_executor too.
                    db.execute_write(query, ticker=ticker, score=score, headline=headline, url=url)
            else:
                logger.info(f"Skipping neutral news ({score}): {headline[:30]}...")
                    
        except Exception as e:
            logger.error(f"Error processing news: {e}")

    # Subscribe to all news ("*")
    news_stream.subscribe_news(on_news, "*")

    logger.info("✅ Pulse is Active (Powered by Gemini). Waiting for news...")
    
    await news_stream._run_forever()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Pulse stopped by user.")