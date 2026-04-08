import sqlite3
import datetime
import json
import base64
import requests
import os
import urllib.parse
from dotenv import load_dotenv
from apify_client import ApifyClient
from openai import OpenAI

# ==========================================
# CONFIGURATION - SECURED VIA .ENV
# ==========================================
# Load environment variables from the .env file
load_dotenv()

APIFY_API_TOKEN = os.getenv('APIFY_API_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not APIFY_API_TOKEN or not OPENAI_API_KEY:
    raise ValueError("Missing API keys! Check your .env file.")

# Search Parameters
SEARCH_TERMS = ["18650", "surron", "broken", "cnc", "3d printer", "makita battery"]
ALLOWED_CITIES = ['kamloops', 'chase', 'merritt', 'darfield', 'kelowna', 'vernon']
SAFETY_MARGIN = 1.5

# Initialize Clients
apify_client = ApifyClient(APIFY_API_TOKEN)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================================
# PHASE 1: DATABASE & SCRAPING ENGINE
# ==========================================
def setup_database():
    conn = sqlite3.connect('arbitrage_memory.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS seen_listings (item_id TEXT PRIMARY KEY, title TEXT, date_added TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS volume_history (date TEXT, keyword TEXT, new_count INTEGER, PRIMARY KEY (date, keyword))''')
    conn.commit()
    return conn

def get_dynamic_limit(keyword, db_conn):
    cursor = db_conn.cursor()
    cursor.execute('''SELECT AVG(new_count) FROM volume_history WHERE keyword = ? AND date >= date('now', '-7 days')''', (keyword,))
    result = cursor.fetchone()[0]
    
    if result is None:
        return 500 # Day 1 sweep
    return max(10, int(result * SAFETY_MARGIN))

def log_daily_volume(keyword, count, db_conn):
    cursor = db_conn.cursor()
    today = datetime.date.today().isoformat()
    cursor.execute('''INSERT OR REPLACE INTO volume_history (date, keyword, new_count) VALUES (?, ?, ?)''', (today, keyword, count))
    db_conn.commit()

def run_scraper(db_conn):
    fresh_deals = []
    cursor = db_conn.cursor()

    start_urls = []
    for keyword in SEARCH_TERMS:
        encoded_keyword = urllib.parse.quote(keyword)
        url = f"https://www.facebook.com/marketplace/114995818516099/search/?query={encoded_keyword}&sortBy=creation_time_descend"
        start_urls.append({"url": url})

    run_input = {
        "startUrls": start_urls,
        "resultsLimit": 150, 
        "includeListingDetails": False,
        "proxyConfiguration": {
            "useApifyProxy": True,
            "apifyProxyGroups": ["RESIDENTIAL"],
            "apifyProxyCountry": "CA"
        }
    }

    print(f"Spinning up Apify Actor...")
    run = apify_client.actor("crawlerbros/facebook-marketplace-scraper").call(run_input=run_input)
    raw_items = list(apify_client.dataset(run["defaultDatasetId"]).iterate_items())
    
    print(f"Batch scrape complete. Analyzing {len(raw_items)} items...")
    new_items_count = 0

    for item in raw_items:
        item_id = str(item.get('url', '')).rstrip('/').split('/')[-1]
        
        # --- FIXED LOCATION PARSING ---
        loc_raw = item.get('location', '')
        if isinstance(loc_raw, dict):
            location = str(loc_raw.get('name', '')).lower()
        else:
            location = str(loc_raw).lower()
        
        title = item.get('title', '')
        
        if not any(city in location for city in ALLOWED_CITIES):
            continue
            
        cursor.execute("SELECT item_id FROM seen_listings WHERE item_id = ?", (item_id,))
        if not cursor.fetchone():
            new_items_count += 1
            
            matched_keyword = "Unknown"
            for kw in SEARCH_TERMS:
                if kw.lower() in title.lower():
                    matched_keyword = kw
                    break
                    
            fresh_deals.append({
                "id": item_id,
                "title": title,
                "price": str(item.get('price', '0')).replace('$', '').replace(',', ''),
                "url": item.get('url', ''),
                "image_url": item.get('primary_image_url', ''),
                "keyword": matched_keyword
            })
            cursor.execute("INSERT INTO seen_listings (item_id, title) VALUES (?, ?)", (item_id, title))

    db_conn.commit()
    print(f"Found {new_items_count} net-new local listings.")
    
    # --- SAFETY VALVE ---
    # If this is the first run, we probably have HUNDREDS of items.
    # Let's only send the 5 most recent ones to the AI to test the logic.
    if new_items_count > 20:
        print(f"⚠️ Warning: {new_items_count} items found. Limiting AI evaluation to the 10 newest to save costs.")
        return fresh_deals[:10]
        
    return fresh_deals

# ==========================================
# PHASE 2: AI EVALUATION ENGINE (OPENAI)
# ==========================================
def evaluate_deal(deal):
    """Passes the listing data and image to GPT-4o-mini for appraisal."""
    print(f"\nEvaluating: {deal['title']} (${deal['price']})")
    
    # 1. Download and base64 encode the image
    base64_image = None
    if deal['image_url']:
        try:
            # We use a user-agent to prevent Facebook from blocking our script's download request
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(deal['image_url'], headers=headers, timeout=5)
            if response.status_code == 200:
                base64_image = base64.b64encode(response.content).decode('utf-8')
        except Exception as e:
            print(f"Failed to load image for {deal['title']}: {e}")

    # 2. Construct the strict prompt
    prompt = f"""
    You are an expert flipper and pawn shop appraiser. Evaluate this Facebook Marketplace listing.
    
    Item Title: {deal['title']}
    Asking Price: ${deal['price']} CAD
    Search Keyword Matched: {deal['keyword']}
    
    Task:
    1. Look at the title, price, and the provided image.
    2. Determine exactly what the item is (e.g., is it a full Sur-Ron bike, or just a replacement footpeg?).
    3. Assess if it is broken, used, or new based on context.
    4. Estimate a conservative fair market resale value in CAD.
    5. Determine if this is a highly profitable arbitrage opportunity (where resale value is significantly higher than asking price + repair effort).
    """

    # 3. Build the payload content list
    content_list = [{"type": "text", "text": prompt}]
    if base64_image:
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
        })

    # 4. Call OpenAI API enforcing a strict JSON schema
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": content_list}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "arbitrage_evaluation",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "identified_item": {"type": "string"},
                            "is_accessory_or_part": {"type": "boolean"},
                            "estimated_resale_value": {"type": "number"},
                            "is_deal": {"type": "boolean"},
                            "reasoning": {"type": "string"}
                        },
                        "required": ["identified_item", "is_accessory_or_part", "estimated_resale_value", "is_deal", "reasoning"],
                        "additionalProperties": False
                    },
                    "strict": True
                }
            }
        )
        
        # Parse the guaranteed JSON response
        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        print(f"AI Evaluation failed: {e}")
        return None

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    db = setup_database()
    
    print("--- STARTING PHASE 1: SCRAPE ---")
    new_deals = run_scraper(db)
    
    print(f"\n--- SCRAPE COMPLETE. Found {len(new_deals)} net-new items. ---")
    print("--- STARTING PHASE 2: AI EVALUATION ---")
    
    winning_deals = []

    for deal in new_deals:
        if not deal['price'] or deal['price'] == '0':
            print(f"Skipping {deal['title']} (No price listed)")
            continue
            
        ai_analysis = evaluate_deal(deal)
        
        if ai_analysis:
            print(f"AI Thinks: {ai_analysis['identified_item']}")
            print(f"Resale Value: ${ai_analysis['estimated_resale_value']}")
            print(f"Is Deal?: {ai_analysis['is_deal']}")
            print(f"Reason: {ai_analysis['reasoning']}")
            
            if ai_analysis['is_deal']:
                winning_deals.append({
                    "deal_info": deal,
                    "analysis": ai_analysis
                })
                print("🚨 ADDED TO WINNERS LIST 🚨")

    print(f"\n--- RUN COMPLETE. Found {len(winning_deals)} profitable flips today. ---")
    
    db.close()
