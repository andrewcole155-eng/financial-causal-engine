import pandas as pd
import requests
import datetime
from io import StringIO
import os

def patch_csv_sectors():
    csv_file = "live_risk_scores.csv"
    
    if not os.path.exists(csv_file):
        print(f"❌ Error: Could not find {csv_file}")
        return

    print(f"📂 Loading {csv_file}...")
    df_csv = pd.read_csv(csv_file)
    
    # 1. Fetch S&P 500 Data (The "Truth" Source)
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    print(f"📡 Fetching Sector data from Wikipedia...")
    
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        tables = pd.read_html(StringIO(response.text))
        
        df_wiki = None
        for t in tables:
            if "Symbol" in t.columns and "GICS Sector" in t.columns:
                df_wiki = t
                break
        
        if df_wiki is None:
            print("❌ Could not download Sector data. Check internet connection.")
            return

        # 2. Create Mapping Dictionary
        # Map Ticker -> Sector
        sector_map = dict(zip(df_wiki['Symbol'], df_wiki['GICS Sector']))
        
        # Handle dot/dash variance (BRK.B vs BRK-B)
        sector_map.update(dict(zip(df_wiki['Symbol'].str.replace('.', '-'), df_wiki['GICS Sector'])))
        
        # Add Manual Overrides for non-S&P stocks in your list
        manual_sectors = {
            "IONQ": "Technology",
            "SNAP": "Communication Services",
            "SIRI": "Communication Services",
            "COR":  "Health Care",
            "CPAY": "Financials",
            "DOC":  "Real Estate",
            "KKR":  "Financials",
            "GE":   "Industrials",
            "UBER": "Industrials",
            "SHOP": "Technology",
            "SQ":   "Financials"
        }
        sector_map.update(manual_sectors)

        # 3. Apply the Patch
        print("🛠️  Patching 'Discovered' sectors...")
        
        def get_real_sector(row):
            ticker = row['Ticker']
            current_sector = row['Sector']
            
            # If we have a better name in our map, use it
            if ticker in sector_map:
                return sector_map[ticker]
            
            # If it's still 'Discovered', try to guess or leave it
            return current_sector

        df_csv['Sector'] = df_csv.apply(get_real_sector, axis=1)

        # 4. Fix Timestamps
        # If Last_Updated is missing/NaN, fill with current time
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df_csv['Last_Updated'] = df_csv['Last_Updated'].fillna(now_str)

        # 5. Save
        df_csv.to_csv(csv_file, index=False)
        print(f"✅ Success! Patched {len(df_csv)} records.")
        print("-" * 50)
        print(df_csv[['Ticker', 'Sector', 'Risk_Score']].head())

    except Exception as e:
        print(f"❌ Error patching CSV: {e}")

if __name__ == "__main__":
    patch_csv_sectors()