import json

# Load the generated maps and fix the Cars tags
with open("generated_maps.json", "r") as f:
    maps = json.load(f)

cars_tags = maps.get("INTEREST_TAGS", {}).get("Cars", [])
print(f"Original Cars tags: {len(cars_tags)}")

# Remove problematic simple tags that cause false matches
problematic_tags = {"car", "cars", "vehicle", "vehicles"}
filtered_cars_tags = [tag for tag in cars_tags if tag.lower() not in problematic_tags]
print(f"After removing problematic tags: {len(filtered_cars_tags)}")

# Add back some specific car terms that are safe
safe_car_terms = [
    "toy car", "toy cars", "pullback car", "pullback cars", 
    "remote control car", "remote control cars", "ride-on car",
    "police car", "ambulance car", "fire truck", "car toy", "cars toy"
]

# Update the maps
maps["INTEREST_TAGS"]["Cars"] = filtered_cars_tags + safe_car_terms

# Save the updated maps
with open("generated_maps_fixed.json", "w") as f:
    json.dump(maps, f, indent=2)

print(f"✅ Saved fixed maps to generated_maps_fixed.json")
print(f"Final Cars tags: {len(maps['INTEREST_TAGS']['Cars'])}")
print("First 10 tags:", maps['INTEREST_TAGS']['Cars'][:10])
