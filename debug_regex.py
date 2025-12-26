import re
import json

# Load the fixed maps
with open("generated_maps_fixed.json", "r") as f:
    maps = json.load(f)

cars_tags = maps.get("INTEREST_TAGS", {}).get("Cars", [])
print(f"Cars tags: {len(cars_tags)}")
print("First 10 tags:", cars_tags[:10])

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

# Test with just a few specific tags
test_tags = ["pullback car", "toy car", "police car"]
cars_rx = _kws_to_compiled(test_tags)
print(f"\nRegex pattern: {cars_rx.pattern}")

# Test on car products
test_products = [
    "Fortune Interceptor SUV Miniature Pullback Police Car",
    "KRT 1.6 Pullback Toy Car", 
    "G-Power Pullback SUV Toy Car"
]

print("\nTesting regex on car products:")
for title in test_products:
    title_lower = title.lower()
    print(f"\n'{title}'")
    print(f"Lowercase: '{title_lower}'")
    match = cars_rx.search(title_lower)
    print(f"Match: {bool(match)}")
    if match:
        print(f"Matched: '{match.group()}'")
    
    # Test individual tags
    for tag in test_tags:
        if tag in title_lower:
            print(f"  Contains '{tag}': True")
