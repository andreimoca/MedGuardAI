import os
import time
import requests
import json
from bs4 import BeautifulSoup
from typing import List, Dict

# The openFDA API provides access to public DailyMed structured product labels.
API_BASE_URL = "https://api.fda.gov/drug/label.json"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'dailymed')

# OpenFDA hard limits: the `limit` query parameter is capped at 1000, and the
# total addressable window via `skip` is capped at 26,000.
PER_REQUEST_MAX = 1000
TOTAL_MAX = 26000

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_drug_labels(limit: int = PER_REQUEST_MAX) -> List[Dict]:
    """
    Fetch up to `limit` drug labels with boxed warnings, contraindications, or
    dosage instructions, paginated in 1000-record pages.

    Args:
        limit: total records to fetch across all pages (capped at TOTAL_MAX).
    """
    target = min(max(limit, 1), TOTAL_MAX)
    print(f"Fetching up to {target} drug labels from openFDA "
          f"(paginating in pages of {PER_REQUEST_MAX})...")

    search = (
        "_exists_:boxed_warning OR _exists_:contraindications "
        "OR _exists_:dosage_and_administration"
    )
    results: List[Dict] = []
    skip = 0
    while len(results) < target:
        page_size = min(PER_REQUEST_MAX, target - len(results))
        params = {"search": search, "limit": page_size, "skip": skip}
        try:
            response = requests.get(API_BASE_URL, params=params, timeout=30)
        except requests.RequestException as exc:
            print(f"  page skip={skip} failed: {exc}; stopping pagination")
            break
        if response.status_code == 404:
            # OpenFDA returns 404 once you exhaust the result window.
            print(f"  page skip={skip} returned 404 (end of results)")
            break
        response.raise_for_status()
        page = response.json().get("results", []) or []
        if not page:
            print(f"  page skip={skip} empty; stopping pagination")
            break
        results.extend(page)
        print(f"  page skip={skip} -> +{len(page)} records (total {len(results)})")
        skip += len(page)
        # Be nice to the API even though we're well under the rate limit.
        time.sleep(0.2)

    return results

def filter_and_save_labels(records: List[Dict]):
    """
    Extract relevant fields and save to individual JSON files in data/raw.
    """
    ensure_output_dir()
    
    saved_count = 0
    for record in records:
        try:
            # We want the brand name or generic name as the file identifier
            openfda_data = record.get('openfda', {})
            names = openfda_data.get('brand_name', openfda_data.get('generic_name', []))
            
            if not names:
                continue
                
            drug_name = names[0].lower().replace(" ", "_").replace("/", "_")
            product_id = record.get('id', 'unknown_id')
            
            filename = f"{drug_name}_{product_id}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            # Extract only the critical functional fields for MedGuardAI
            extracted_data = {
                "drug_name": names[0],
                "generic_name": openfda_data.get('generic_name', []),
                "active_ingredients": record.get("active_ingredient", []),
                "indications_and_usage": record.get("indications_and_usage", []),
                "contraindications": record.get("contraindications", []),
                "boxed_warning": record.get("boxed_warning", []),
                "warnings_and_cautions": record.get("warnings_and_cautions", []),
                "adverse_reactions": record.get("adverse_reactions", []),
                "drug_interactions": record.get("drug_interactions", []),
                "dosage_and_administration": record.get("dosage_and_administration", []),
                "overdosage": record.get("overdosage", [])
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, indent=4, ensure_ascii=False)
                
            saved_count += 1
            print(f"Saved: {filename}")
            
        except Exception as e:
            print(f"Error processing record {record.get('id')}: {e}")
            
    print(f"\nSuccessfully saved {saved_count} drug label files to {OUTPUT_DIR}")

if __name__ == "__main__":
    import sys
    # Allow override via CLI: `python fetch_openfda.py 5000`
    requested = int(sys.argv[1]) if len(sys.argv) > 1 else TOTAL_MAX
    try:
        records = fetch_drug_labels(limit=requested)
        filter_and_save_labels(records)
    except Exception as e:
        print(f"Failed to execute ingestion pipeline: {e}")
