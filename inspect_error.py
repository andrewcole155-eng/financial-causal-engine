import json

try:
    print("🔍 Reading first line of debug_raw_results.jsonl ...")
    with open("debug_raw_results.jsonl", "r") as f:
        line = f.readline()
        if not line:
            print("❌ File is empty.")
            exit()
            
        data = json.loads(line)
        
        # Print the full JSON structure nicely so we can see the error
        print(json.dumps(data, indent=2))
        
except FileNotFoundError:
    print("❌ Could not find 'debug_raw_results.jsonl'. Did you run the previous script?")