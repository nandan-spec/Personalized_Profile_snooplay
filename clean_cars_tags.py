import json

# Load the fixed maps and check for problematic tags
with open("generated_maps_fixed.json", "r") as f:
    maps = json.load(f)

cars_tags = maps.get("INTEREST_TAGS", {}).get("Cars", [])
print(f"Current Cars tags: {len(cars_tags)}")

# Find problematic tags that are too generic
problematic_patterns = [
    "rg_theme_vehicle",  # Too generic
    "vehicle",  # Too generic
    "vehicles",  # Too generic
    "transport",  # Too generic
    "transportation"  # Too generic
]

problematic_tags = []
for tag in cars_tags:
    for pattern in problematic_patterns:
        if pattern in tag.lower():
            problematic_tags.append(tag)
            break

print(f"Found {len(problematic_tags)} problematic tags:")
for tag in problematic_tags:
    print(f"  - {tag}")

# Remove these problematic tags
filtered_cars_tags = [tag for tag in cars_tags if tag not in problematic_tags]
print(f"After removing problematic tags: {len(filtered_cars_tags)}")

# Add back specific car terms that are safe
safe_car_terms = [
    "toy car", "toy cars", "pullback car", "pullback cars", 
    "remote control car", "remote control cars", "ride-on car",
    "police car", "ambulance car", "fire truck", "car toy", "cars toy",
    "diecast car", "model car", "racing car", "sports car"
]

# Update the maps
maps["INTEREST_TAGS"]["Cars"] = filtered_cars_tags + safe_car_terms

# Save the updated maps
with open("generated_maps_fixed_v2.json", "w") as f:
    json.dump(maps, f, indent=2)

print(f"✅ Saved updated maps to generated_maps_fixed_v2.json")
print(f"Final Cars tags: {len(maps['INTEREST_TAGS']['Cars'])}")
print("First 10 tags:", maps['INTEREST_TAGS']['Cars'][:10])
