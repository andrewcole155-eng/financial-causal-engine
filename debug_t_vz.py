import json
from database_manager import DatabaseManager

# Load Config
try:
    with open("config.json", 'r') as f:
        config = json.load(f)
except:
    print("❌ Config not found.")
    exit()

db = DatabaseManager(config)

print("\n------------------------------------------------")
print("🕵️‍♀️ DEBUGGING T -> VZ CONNECTION")
print("------------------------------------------------")

# Fetch the specific neighborhood used in the app
G = db.get_neighborhood_graph("T")

if G.has_edge("T", "VZ"):
    data = G.get_edge_data("T", "VZ")
    print(f"✅ Edge Found!")
    print(f"   - Status:    {data.get('verification_status')}")
    print(f"   - Mechanism: {data.get('mechanism')}")
    
    if data.get('verification_status') == 'AI_PROPOSED':
        print("\n🎉 SUCCESS: Python sees the AI data. If Streamlit doesn't, it is 100% a Cache issue.")
    else:
        print("\n❌ FAILURE: Python still sees the old Verified data.")
else:
    print("❌ No edge found between T and VZ.")

db.close()