import re
import json
import pickle

# Load the fixed maps
with open("generated_maps_fixed.json", "r") as f:
    maps = json.load(f)

cars_tags = maps.get("INTEREST_TAGS", {}).get("Cars", [])
print(f"Cars tags: {len(cars_tags)}")

# Test the regex pattern
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

cars_rx = _kws_to_compiled(cars_tags[:20])  # Use first 20 tags

# Load some car products and test
with open("product_search_index.pkl", "rb") as f:
    pkl_data = pickle.load(f)
data = pkl_data["data"]

# Test on some known car products
test_products = [
    "Fortune Interceptor SUV Miniature Pullback Police Car",
    "KRT 1.6 Pullback Toy Car",
    "G-Power Pullback SUV Toy Car",
    "Remote Controlled Excavator Crane Toy"
]

print("\nTesting regex on car products:")
for title in test_products:
    match = cars_rx.search(title.lower())
    print(f"'{title}' -> {bool(match)}")
    if match:
        print(f"  Matched: {match.group()}")
