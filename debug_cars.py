import re
import pandas as pd
from typing import List, Optional

# Load the CSV data
DATA = pd.read_csv("shopify_products_export.csv", low_memory=False)

# Create TEXT field like in the main app
if "description_plain" not in DATA.columns:
    DATA["description_plain"] = ""
DATA["TEXT"] = (
    (DATA.get("tags", "").fillna("").astype(str)) + " " +
    (DATA.get("title", "").fillna("").astype(str)) + " " +
    (DATA.get("description_plain", "").fillna("").astype(str)) + " " +
    (DATA.get("product_type", "").fillna("").astype(str))
).str.lower()

# Define the Cars interest keywords with more specific terms
INTEREST_TAGS = {
    "Cars": ["toy car","toy cars","pullback car","pullback cars","remote control car","remote control cars","ride-on","jeep","police car","ambulance car","fire truck","truck toy","car toy","cars toy"]
}

def _cars_syns_guard(syns: List[str]) -> List[str]:
    bad = {"vehicle","vehicles","transport","transportation"}
    return [s for s in syns if s and s.strip().lower() not in bad]

def _kws_to_compiled(kws: List[str]) -> Optional[re.Pattern]:
    kws = [k for k in (kws or []) if k]
    if not kws:
        return None
    alts = []
    for k in kws:
        k = re.escape(k.strip().lower()).replace(r"\ ", r"[ -]?")
        alts.append(f"{k}s?")
    return re.compile(r"(?:^|[^a-z0-9])(?:" + "|".join(alts) + r")(?:[^a-z0-9]|$)")

# Create the Cars regex
syns = _cars_syns_guard(INTEREST_TAGS["Cars"])
cars_rx = _kws_to_compiled(syns)
print(f"Cars regex pattern: {cars_rx.pattern}")

# Test the regex on some car-related products
test_texts = [
    "toy car",
    "pullback car",
    "remote control car", 
    "car toy",
    "cars collection",
    "ride-on car",
    "jeep toy",
    "police car toy",
    "ambulance car",
    "fire truck",
    "toy vehicle",
    "transport toy",
    "card games",  # This should NOT match
    "playing cards",  # This should NOT match
    "toy cars",
    "pullback cars",
    "remote control cars",
    "cars toy"
]

print("\nTesting regex on sample texts:")
for text in test_texts:
    match = cars_rx.search(text.lower())
    print(f"'{text}' -> {bool(match)}")

# Now test on actual products and show what's matching
print(f"\nTesting on actual products...")
car_products = []
for i, row in DATA.iterrows():
    text = row["TEXT"]
    match = cars_rx.search(text)
    if match:
        # Find all matches in the text
        all_matches = cars_rx.findall(text)
        car_products.append({
            "id": row.get("id"),
            "title": row.get("title"),
            "text": text[:200] + "..." if len(text) > 200 else text,
            "matches": all_matches
        })

print(f"Found {len(car_products)} products matching Cars interest")
for i, product in enumerate(car_products[:15]):  # Show first 15
    print(f"{i+1}. ID: {product['id']}, Title: {product['title']}")
    print(f"   Matches: {product['matches']}")
    print(f"   Text: {product['text'][:150]}...")
    print()

# Now let's look for actual car-related products
print("\nLooking for products with 'car' in title...")
car_title_products = []
for i, row in DATA.iterrows():
    title = str(row.get("title", "")).lower()
    if "car" in title:
        car_title_products.append({
            "id": row.get("id"),
            "title": row.get("title"),
            "text": row["TEXT"][:100] + "..." if len(row["TEXT"]) > 100 else row["TEXT"]
        })

print(f"Found {len(car_title_products)} products with 'car' in title")
for i, product in enumerate(car_title_products[:10]):
    print(f"{i+1}. ID: {product['id']}, Title: {product['title']}")
    print(f"   Text: {product['text']}")
    print()
