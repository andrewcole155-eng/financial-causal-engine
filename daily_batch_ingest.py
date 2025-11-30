# ==============================================================================
# --- DAILY BATCH INGESTION SCRIPT (Tuned for Higher Sensitivity) ---
# ==============================================================================
# Triggers Gemini 2.0 Flash (Batch) to process daily news into the Knowledge Graph.
# ==============================================================================

import os
import json
import time
import logging
from typing import List, Dict, Any
from google import genai
from google.genai import types

# --- IMPORT YOUR EXISTING DATABASE MANAGER ---
from database_manager import DatabaseManager

# --- CONFIGURATION SETUP ---
def load_app_config(config_path="config.json"):
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

# 1. Load the Config File
app_config = load_app_config()

# 2. Setup API Key 
GEMINI_API_KEY = app_config.get("GOOGLE_API_KEY") or app_config.get("gemini_api_key") or os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("❌ FATAL: API Key not found.")

# 3. Initialize Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.0-flash"

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("BatchIngest")


# ==============================================================================
# --- STEP 1: DEFINE THE "SCHEMA GUARDRAILS" ---
# ==============================================================================

VALID_RELATIONSHIPS = [
    "SUPPLIES_TO", "COMPETES_WITH", "INCREASES_RISK_FOR", 
    "POSITIVELY_CORRELATED_WITH", "NEGATIVELY_CORRELATED_WITH",
    "IS_SUBSIDIARY_OF", "REGULATES", "AFFECTS_DEMAND_FOR",
    "MOVES_IN_SYMPATHY_WITH", "HAS_SHARED_RISK_WITH"
]

# JSON Schema for Gemini Output
response_schema = {
    "type": "OBJECT",
    "properties": {
        "relationships": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "source_ticker": {"type": "STRING"},
                    "target_ticker": {"type": "STRING"},
                    "relationship_type": {"type": "STRING", "enum": VALID_RELATIONSHIPS},
                    "confidence": {"type": "NUMBER"},
                    "mechanism": {"type": "STRING", "description": "Short explanation of the link"}
                },
                "required": ["source_ticker", "target_ticker", "relationship_type", "confidence"]
            }
        }
    }
}

# ==============================================================================
# --- STEP 2: BATCH WORKFLOW FUNCTIONS ---
# ==============================================================================

def prepare_batch_file(news_items: List[Dict]) -> str:
    """
    Converts a list of news items into a JSONL file for Google's Native Batch API.
    """
    batch_requests = []
    
    for i, item in enumerate(news_items):
        request_id = f"req_{item.get('ticker', 'UNKNOWN')}_{i}"
        
        headline = item.get('headline', '')
        ticker = item.get('ticker', 'Unknown Ticker')
        
        # --- TUNED PROMPT: MORE AGGRESSIVE DISCOVERY ---
        prompt_text = f"""
        Analyze this financial news for ticker {ticker}: 
        "{headline}"
        
        Task: Identify ALL potential ripple effects, contagion risks, or sympathy moves.
        
        Thinking Process:
        1. Does this affect a Supplier? (e.g. Apple cuts production -> Foxconn hurts)
        2. Does this affect a Competitor? (e.g. AMD releases good chip -> NVDA hurts)
        3. Does this affect the whole Sector? (e.g. Oil price up -> Airlines hurt)
        
        Output:
        - Extract relationships even if they are implicit or speculative (just lower the confidence score).
        - Use 'MOVES_IN_SYMPATHY_WITH' for general correlation.
        - Output strictly valid JSON.
        """

        request_entry = {
            "custom_id": request_id, 
            "request": {
                "contents": [{"parts": [{"text": prompt_text}]}],
                "generationConfig": {
                    "response_schema": response_schema,
                    "response_mime_type": "application/json"
                }
            }
        }
        batch_requests.append(request_entry)

    filename = "daily_batch_input.jsonl"
    with open(filename, "w") as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")
            
    logger.info(f"✅ Prepared {len(batch_requests)} requests in {filename}")
    return filename

def run_batch_job(input_file: str):
    """Uploads file and waits for the Batch Job to complete."""
    
    # 1. Upload
    logger.info("🚀 Uploading batch file to Google...")
    batch_file = client.files.upload(
        file=input_file,
        config={'mime_type': 'application/json'}
    )
    
    # 2. Start Job
    job = client.batches.create(
        model=MODEL_NAME,
        src=batch_file.name,
        config={'display_name': f'ingest_{int(time.time())}'}
    )
    logger.info(f"⏳ Job {job.name} started. Waiting for completion (this may take time)...")

    # 3. Poll for Completion
    while True:
        job = client.batches.get(name=job.name)
        if job.state == "SUCCEEDED" or str(job.state) == "JobState.JOB_STATE_SUCCEEDED":
            logger.info("✅ Job Completed Successfully!")
            break
        elif "FAILED" in str(job.state):
            raise Exception(f"Batch Job Failed: {job.error.message}")
        else:
            time.sleep(30) # Check every 30 seconds
            
    return job

def process_and_save_results(job_name, db: DatabaseManager):
    """Downloads results and inserts them into Neo4j using the Staging Logic."""
    
    job = client.batches.get(name=job_name)
    
    # Robust file name extraction
    try:
        output_file_name = job.dest.file_name
    except AttributeError:
        output_file_name = "files/" + job.name.replace("batches/", "batch-")

    content_bytes = client.files.download(file=output_file_name)
    content = content_bytes.decode("utf-8")
    
    count = 0
    for line in content.splitlines():
        try:
            result = json.loads(line)
            
            # Skip items that failed individually
            if "response" not in result: continue 
            
            # Extract JSON from Gemini response
            candidates = result["response"].get("candidates", [])
            if not candidates or "content" not in candidates[0]: continue
            
            model_json_str = candidates[0]["content"]["parts"][0]["text"]
            model_data = json.loads(model_json_str) 
            
            # Write to Database
            if "relationships" in model_data and model_data["relationships"]:
                for rel in model_data["relationships"]:
                    
                    if rel['source_ticker'] == rel['target_ticker']: continue

                    rel_props = {
                        "weight": float(rel['confidence']),
                        "mechanism": rel['mechanism'],
                        "verification_status": "AI_PROPOSED", 
                        "source": "Gemini_Batch",
                        "last_updated": time.strftime("%Y-%m-%d")
                    }
                    
                    db.upsert_relationship(
                        source_ticker=rel['source_ticker'],
                        target_ticker=rel['target_ticker'],
                        rel_type=rel['relationship_type'],
                        properties=rel_props
                    )
                    count += 1

        except Exception as e:
            logger.error(f"Error parsing line: {e}")

    logger.info(f"🎉 Successfully ingested {count} new causal relationships into the Graph.")

# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================

if __name__ == "__main__":
    if not app_config:
        logger.error("❌ Config is empty. Cannot connect to Neo4j.")
        exit(1)

    try:
        db = DatabaseManager(app_config) 
        
        # 1. Fetch News 
        logger.info("🗞️ Fetching recent significant events for processing...")
        recent_events = db.get_recent_events(limit=50)
        
        if recent_events:
            # 2. Run the Pipeline
            logger.info(f" -> Found {len(recent_events)} events. Starting batch job...")
            input_file = prepare_batch_file(recent_events)
            job = run_batch_job(input_file)
            
            # 3. Process
            process_and_save_results(job.name, db)
            
            # 4. Clean up
            if os.path.exists(input_file):
                os.remove(input_file)
        else:
            logger.info("No new events found to process.")
            
    except Exception as e:
        logger.error(f"❌ Script failed: {e}")
    finally:
        if 'db' in locals():
            db.close()