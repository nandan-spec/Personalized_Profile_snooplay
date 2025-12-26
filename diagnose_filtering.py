import pickle
import numpy as np
import pandas as pd
import json
import re

# Load the data
print("Loading data...")
with open("product_search_index.pkl", "rb") as f:
    payload = pickle.load(f)
DATA = payload["data"]
EMB = payload["embeddings"].astype("float32")
print(f"✅ Loaded PKL: {len(DATA)} rows, embeddings {EMB.shape}")

# Load the maps
print("Loading maps...")
with open("generated_maps_fixed_v2.json", "r") as f:
    maps = json.load(f)

AGE_MAP = maps.get("AGE_MAP", {})
INTEREST_TAGS = maps.get("INTEREST_TAGS", {})
AGE_RX = maps.get("AGE_RX", {})
INTEREST_RX = maps.get("INTEREST_RX", {})

print(f"✅ Loaded maps:")
print(f"   - AGE_MAP: {len(AGE_MAP)} age bands")
print(f"   - INTEREST_TAGS: {len(INTEREST_TAGS)} interests")
print(f"   - AGE_RX: {len(AGE_RX)} age regex patterns")
print(f"   - INTEREST_RX: {len(INTEREST_RX)} interest regex patterns")

# Test the specific case: 0-2 years, Cars interest
print("\n" + "="*80)
print("DIAGNOSING FILTER ISSUES FOR 0-2 YEARS + CARS")
print("="*80)

# Check age bands for 0-2 years
print(f"\n1. AGE BANDS FOR 0-2 YEARS:")
age_bands = ["0-2 yr"]
print(f"   Looking for age bands: {age_bands}")

for band in age_bands:
    if band in AGE_MAP:
        print(f"   ✅ Found age band '{band}' with {len(AGE_MAP[band])} tags")
        print(f"      Sample tags: {AGE_MAP[band][:5]}")
    else:
        print(f"   ❌ Age band '{band}' not found in AGE_MAP")

# Check Cars interest
print(f"\n2. CARS INTEREST:")
if "Cars" in INTEREST_TAGS:
    print(f"   ✅ Found 'Cars' interest with {len(INTEREST_TAGS['Cars'])} tags")
    print(f"      Sample tags: {INTEREST_TAGS['Cars'][:10]}")
else:
    print(f"   ❌ 'Cars' interest not found in INTEREST_TAGS")

# Check regex patterns
print(f"\n3. REGEX PATTERNS:")
if "0-2 yr" in AGE_RX:
    print(f"   ✅ Found age regex for '0-2 yr': {AGE_RX['0-2 yr']}")
else:
    print(f"   ❌ No age regex for '0-2 yr'")

if "Cars" in INTEREST_RX:
    print(f"   ✅ Found interest regex for 'Cars': {INTEREST_RX['Cars']}")
else:
    print(f"   ❌ No interest regex for 'Cars'")

# Search for products that might match
print(f"\n4. SEARCHING FOR POTENTIAL MATCHES:")
print(f"   Searching through {len(DATA)} products...")

potential_matches = []
age_matches = 0
car_matches = 0
both_matches = 0

for idx, row in DATA.iterrows():
    title = str(row.get('title', '')).lower()
    tags = str(row.get('tags', '')).lower()
    product_type = str(row.get('product_type', '')).lower()
    text = f"{title} {tags} {product_type}"
    
    # Check age match
    age_match = False
    for band in age_bands:
        if band in AGE_RX:
            if AGE_RX[band].search(text):
                age_match = True
                break
    
    # Check car match
    car_match = False
    if "Cars" in INTEREST_RX:
        if INTEREST_RX["Cars"].search(text):
            car_match = True
    
    if age_match:
        age_matches += 1
    if car_match:
        car_matches += 1
    if age_match and car_match:
        both_matches += 1
        potential_matches.append({
            'id': row.get('id'),
            'title': row.get('title'),
            'tags': row.get('tags', ''),
            'product_type': row.get('product_type', ''),
            'vendor': row.get('vendor', ''),
            'price': row.get('variant_price', 0)
        })

print(f"   Results:")
print(f"   - Products matching age (0-2): {age_matches}")
print(f"   - Products matching cars: {car_matches}")
print(f"   - Products matching both: {both_matches}")

if potential_matches:
    print(f"\n   Sample products matching both criteria:")
    for i, product in enumerate(potential_matches[:5], 1):
        tags = product['tags'][:100] + "..." if len(str(product['tags'])) > 100 else product['tags']
        print(f"   {i}. ID: {product['id']}")
        print(f"      Title: {product['title']}")
        print(f"      Tags: {tags}")
        print(f"      Type: {product['product_type']} | Vendor: {product['vendor']}")
        print(f"      Price: {product['price']}")
        print()
else:
    print(f"\n   ❌ No products found matching both age and car criteria")

# Check if there are any car products for 0-2 years in the data
print(f"\n5. MANUAL SEARCH FOR CAR PRODUCTS (0-2 YEARS):")
manual_matches = []

for idx, row in DATA.iterrows():
    title = str(row.get('title', '')).lower()
    tags = str(row.get('tags', '')).lower()
    product_type = str(row.get('product_type', '')).lower()
    text = f"{title} {tags} {product_type}"
    
    # Look for car-related terms
    car_terms = ['car', 'vehicle', 'train', 'truck', 'jeep', 'police car', 'racing car', 'toy car']
    has_car = any(term in text for term in car_terms)
    
    # Look for 0-2 age terms
    age_terms = ['0-2', '0-11', '1-2', '1-3', '0-12 month', '0-24 month']
    has_age = any(term in text for term in age_terms)
    
    if has_car and has_age:
        # Exclude books and activity kits
        is_book_or_kit = any(term in text for term in ['book', 'activity', 'kit', 'workbook', 'worksheet'])
        if not is_book_or_kit:
            manual_matches.append({
                'id': row.get('id'),
                'title': row.get('title'),
                'tags': row.get('tags', ''),
                'product_type': row.get('product_type', ''),
                'vendor': row.get('vendor', ''),
                'price': row.get('variant_price', 0)
            })

print(f"   Found {len(manual_matches)} car products for 0-2 years:")
for i, product in enumerate(manual_matches[:10], 1):
    tags = product['tags'][:100] + "..." if len(str(product['tags'])) > 100 else product['tags']
    print(f"   {i}. ID: {product['id']}")
    print(f"      Title: {product['title']}")
    print(f"      Tags: {tags}")
    print(f"      Type: {product['product_type']} | Vendor: {product['vendor']}")
    print(f"      Price: {product['price']}")
    print()
