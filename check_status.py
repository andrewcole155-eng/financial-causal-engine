# check_status.py (Corrected Parser)
import os
import time
import json
import logging
import sys
from google import genai
from database_manager import DatabaseManager

# --- CONFIGURATION ---
def load_app_config(config_path="config.json"):
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

app_config = load_app_config()
GEMINI_API_KEY = app_config.get("GOOGLE_API_KEY") or app_config.get("gemini_api_key") or os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    print("❌ Error: API Key not found.")
    sys.exit(1)

client = genai.Client(api_key=GEMINI_API_KEY)
JOB_NAME = "batches/bnh3m1cziimgmi2kfkxn90ja9i04t0p60d8v"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("StatusCheck")

def check_and_ingest():
    db = DatabaseManager(app_config)
    
    try:
        logger.info(f"🕵️ Checking status of job: {JOB_NAME}...")
        job = client.batches.get(name=JOB_NAME)
        state_str = str(job.state)

        if "SUCCEEDED" in state_str:
            logger.info("✅ Job is DONE! Downloading results...")
            
            # Extract file name
            try:
                output_file_name = job.dest.file_name
            except AttributeError:
                # Fallback if attribute structure varies
                output_file_name = "files/" + JOB_NAME.replace("batches/", "batch-")
            
            # Download content
            content_bytes = client.files.download(file=output_file_name)
            content = content_bytes.decode("utf-8")
            
            count = 0
            skipped_empty = 0
            
            for line in content.splitlines():
                try:
                    result = json.loads(line)
                    
                    if "response" not in result: continue 
                    
                    # --- THE FIX: REMOVED ["body"] ---
                    candidates = result["response"].get("candidates", [])
                    
                    if not candidates or "content" not in candidates[0]: 
                        continue
                    
                    model_json_str = candidates[0]["content"]["parts"][0]["text"]
                    model_data = json.loads(model_json_str)
                    
                    # 3. Insert relationships
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
                    else:
                        # This counts items where AI said "No relationships found"
                        skipped_empty += 1

                except Exception as e:
                    logger.error(f"Parse error: {e}")

            logger.info(f"🎉 Inserted {count} new AI-inferred relationships.")
            logger.info(f"ℹ️  Skipped {skipped_empty} news items where AI found no causal link.")
            
        else:
            logger.info(f"⏳ Job Status: {state_str}.")

    finally:
        db.close()

if __name__ == "__main__":
    check_and_ingest()