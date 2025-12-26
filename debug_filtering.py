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
GOAL_TAGS = maps.get("GOAL_TAGS", {})

print(f"✅ Loaded maps:")
print(f"   - AGE_MAP: {len(AGE_MAP)} age bands")
print(f"   - INTEREST_TAGS: {len(INTEREST_TAGS)} interests")
print(f"   - GOAL_TAGS: {len(GOAL_TAGS)} goals")

# Generate regex patterns at runtime
def _kws_to_compiled(kws):
    kws = [k for k in (kws or []) if k]
    if not kws:
        return None
    alts = []
    for k in kws:
        k = re.escape(k.strip().lower()).replace(r"\ ", r"[ -]?")
        alts.append(f"{k}s?")
    return re.compile(r"(?:^|[^a-z0-9])(?:" + "|".join(alts) + r")(?:[^a-z0-9]|$)")

AGE_RX = {b: _kws_to_compiled(AGE_MAP.get(b, [])) for b in AGE_MAP}
INTEREST_RX = {}
for k, syns in INTEREST_TAGS.items():
    INTEREST_RX[k] = _kws_to_compiled(syns or [k])

print(f"✅ Generated regex patterns:")
print(f"   - AGE_RX: {len(AGE_RX)} patterns")
print(f"   - INTEREST_RX: {len(INTEREST_RX)} patterns")

# Test filtering logic
print("\n" + "="*80)
print("TESTING FILTERING LOGIC")
print("="*80)

# Test case: age 3-6, Cars interest
age_min, age_max = 3, 6
interests = ["Cars"]

# Get age bands
def overlapping_age_bands(p):
    amin = p.get("age_min")
    amax = p.get("age_max")
    if amin is None and amax is None:
        return []
    try:
        amin = int(0 if amin is None else amin)
        amax = int(amax if amax is not None else amin)
    except Exception:
        return []
    if amax < amin:
        amin, amax = amax, amin
    bands = {"0-2 yr": (0,2), "3-5 yr": (3,5), "6-8 yr": (6,8), "9-11 yr": (9,11)}
    out = []
    for k,(lo,hi) in bands.items():
        if not (amax < lo or amin > hi):
            out.append(k)
    return out

profile = {"age_min": age_min, "age_max": age_max}
age_bands = overlapping_age_bands(profile)
print(f"Age bands for {age_min}-{age_max}: {age_bands}")

# Test filtering on a sample of products
print(f"\nTesting filtering on first 1000 products...")
passed_filters = 0
age_passed = 0
interest_passed = 0
both_passed = 0

for idx, row in DATA.head(1000).iterrows():
    title = str(row.get('title', '')).lower()
    tags = str(row.get('tags', '')).lower()
    product_type = str(row.get('product_type', '')).lower()
    text = f"{title} {tags} {product_type}"
    
    # Test age filter
    age_match = False
    for band in age_bands:
        rx = AGE_RX.get(band)
        if rx and rx.search(text):
            age_match = True
            break
    
    # Test interest filter
    interest_match = False
    for interest in interests:
        rx = INTEREST_RX.get(interest)
        if rx and rx.search(text):
            interest_match = True
            break
    
    if age_match:
        age_passed += 1
    if interest_match:
        interest_passed += 1
    if age_match and interest_match:
        both_passed += 1
        passed_filters += 1
        print(f"✅ Found match: {row.get('title')}")
        print(f"   Tags: {row.get('tags', '')[:100]}...")
        print(f"   Type: {row.get('product_type')}")
        print()

print(f"\nResults:")
print(f"   - Products passing age filter: {age_passed}")
print(f"   - Products passing interest filter: {interest_passed}")
print(f"   - Products passing both filters: {both_passed}")

# Test with broader search
print(f"\n" + "="*80)
print("BROADER SEARCH TEST")
print("="*80)

# Search for any car-related products
car_products = []
for idx, row in DATA.iterrows():
    title = str(row.get('title', '')).lower()
    tags = str(row.get('tags', '')).lower()
    product_type = str(row.get('product_type', '')).lower()
    text = f"{title} {tags} {product_type}"
    
    # Look for car-related terms
    car_terms = ['car', 'vehicle', 'train', 'truck', 'jeep', 'police car', 'racing car', 'toy car']
    has_car = any(term in text for term in car_terms)
    
    if has_car:
        car_products.append({
            'id': row.get('id'),
            'title': row.get('title'),
            'tags': row.get('tags', ''),
            'product_type': row.get('product_type', ''),
            'price': row.get('variant_price', 0)
        })

print(f"Found {len(car_products)} car-related products total")

# Check how many are for age 3-6
age_3_6_products = []
for product in car_products[:50]:  # Check first 50
    tags = str(product['tags']).lower()
    age_terms = ['3-6', '3-4', '4-6', '3-5', '4-5']
    has_age = any(term in tags for term in age_terms)
    if has_age:
        age_3_6_products.append(product)

print(f"Found {len(age_3_6_products)} car products for age 3-6:")
for i, product in enumerate(age_3_6_products[:10], 1):
    print(f"{i}. {product['title']}")
    print(f"   Tags: {product['tags'][:100]}...")
    print(f"   Type: {product['product_type']}")
    print()
