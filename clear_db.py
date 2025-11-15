# clear_db.py
from database_manager import DatabaseManager
from worker import load_config

print("--- Running Database Cleanup Script ---")
config = load_config()
if config:
    db_manager = DatabaseManager(config)
    if db_manager.is_connected():
        print("WARNING: This will delete all data from your Neo4j database.")
        confirm = input("Are you sure you want to continue? (yes/no): ")
        if confirm.lower() == 'yes':
            db_manager.clear_neo4j_database()
            print("--- Cleanup complete. ---")
        else:
            print("Cleanup aborted.")
    else:
        print("Could not connect to database.")