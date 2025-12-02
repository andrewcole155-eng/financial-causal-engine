# test_injection.py
from database_manager import DatabaseManager

# Initialize
config = {"neo4j": {"uri": "", "user": "", "password": ""}} # Dummy config for SQLite mode
db = DatabaseManager(config)

print("Injecting a fake event into SQLite...")
db.insert_event(
    ticker="TEST", 
    headline="✅ If you see this, the database connection is FIXED.", 
    score=0.99, 
    link="http://localhost"
)
print("Done. Now click 'Refresh Events' in your Streamlit app.")