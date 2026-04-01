import os
import time
import requests
import json
from bs4 import BeautifulSoup
from typing import List, Dict

# The openFDA API provides access to public DailyMed structured product labels.
API_BASE_URL = "https://api.fda.gov/drug/label.json"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'dailymed')

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_drug_labels(limit: int = 20) -> List[Dict]:
    """
    Fetches raw labeling data from the openFDA API representing DailyMed info.
    We are particularly interested in warnings, contraindications, and dosage.
    """
    print(f"Fetching {limit} drug labels from openFDA...")
    
    # We query records that have boxed warnings, contraindications, or dosage instructions
    params = {
        'search': '_exists_:boxed_warning OR _exists_:contraindications OR _exists_:dosage_and_administration',
        'limit': limit
    }
    
    response = requests.get(API_BASE_URL, params=params)
    response.raise_for_status()
    
    data = response.json()
    return data.get('results', [])

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
    try:
        # Fetching a small batch initially for demonstration and schema validation
        records = fetch_drug_labels(limit=50)
        filter_and_save_labels(records)
    except Exception as e:
        print(f"Failed to execute ingestion pipeline: {e}")
