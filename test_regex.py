import re
import json

# Load the generated maps to see the Cars regex pattern
with open("generated_maps.json", "r") as f:
    maps = json.load(f)

cars_tags = maps.get("INTEREST_TAGS", {}).get("Cars", [])
print(f"Cars tags from generated_maps.json: {len(cars_tags)} tags")
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

cars_rx = _kws_to_compiled(cars_tags[:10])  # Use first 10 tags for testing
print(f"Cars regex pattern: {cars_rx.pattern}")

# Test some problematic texts
test_texts = [
    "card games",
    "playing cards", 
    "toy car",
    "remote control car",
    "board games",
    "car toy"
]

print("\nTesting regex on sample texts:")
for text in test_texts:
    match = cars_rx.search(text.lower())
    print(f"'{text}' -> {bool(match)}")
