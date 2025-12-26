import pickle
import pandas as pd
import re
import json

# Load the data and check what's actually matching
with open("product_search_index.pkl", "rb") as f:
    pkl_data = pickle.load(f)
data = pkl_data["data"]

# Load the generated maps
with open("generated_maps.json", "r") as f:
    maps = json.load(f)

cars_tags = maps.get("INTEREST_TAGS", {}).get("Cars", [])
print(f"Original Cars tags: {len(cars_tags)}")

# Apply the cars_syns_guard function
def _cars_syns_guard(syns):
    bad = {"vehicle","vehicles","transport","transportation"}
    return [s for s in syns if s and s.strip().lower() not in bad]

filtered_cars_tags = _cars_syns_guard(cars_tags)
print(f"After cars_syns_guard: {len(filtered_cars_tags)}")

# Check what was removed
removed = set(cars_tags) - set(filtered_cars_tags)
print(f"Removed tags: {removed}")

# Test the regex with filtered tags
def _kws_to_compiled(kws):
    kws = [k for k in (kws or []) if k]
    if not kws:
        return None
    alts = []
    for k in kws:
        k = re.escape(k.strip().lower()).replace(r"\ ", r"[ -]?")
        alts.append(f"{k}s?")
    pattern = r"(?:^|[^a-z0-9])(?:" + "|".join(alts) + r")(?:[^a-z0-9]|$)"
    return re.compile(pattern)

cars_rx = _kws_to_compiled(filtered_cars_tags[:20])  # Use first 20 tags

# Check some of the products that are matching
with open("precomputed_masks.pkl", "rb") as f:
    masks_data = pickle.load(f)

cars_mask = masks_data['INTEREST_MASKS']['Cars']
car_indices = cars_mask.nonzero()[0]

print(f"\nChecking first 5 car products:")
for i in car_indices[:5]:
    row = data.iloc[i]
    text = f"{row.get('tags', '')} {row.get('title', '')} {row.get('description_plain', '')} {row.get('product_type', '')}".lower()
    print(f"\nProduct {i}: {row.get('title', 'No title')}")
    print(f"Text: {text[:200]}...")
    
    # Check which car tag is matching
    for tag in filtered_cars_tags[:10]:
        if tag.lower() in text:
            print(f"  Matches tag: '{tag}'")
            break
