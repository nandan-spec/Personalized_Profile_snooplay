import pickle
import pandas as pd
import json

# Find the "3-in-1 Activity Bundle" product and check what car tag is matching
with open("product_search_index.pkl", "rb") as f:
    pkl_data = pickle.load(f)
data = pkl_data["data"]

# Search for this specific product
target_title = "3-in-1 Activity Bundle"
found_product = None
found_index = None

for i, row in data.iterrows():
    if target_title in str(row.get('title', '')):
        found_product = row
        found_index = i
        break

if found_product is not None:
    print(f"Found product at index {found_index}")
    print(f"Title: {found_product.get('title')}")
    print(f"Tags: {found_product.get('tags', '')[:200]}...")
    
    # Check if it's in the Cars mask
    with open("precomputed_masks.pkl", "rb") as f:
        masks_data = pickle.load(f)
    
    cars_mask = masks_data['INTEREST_MASKS']['Cars']
    if cars_mask[found_index]:
        print(f"✅ This product IS in the Cars mask!")
        
        # Check what car tag is matching
        text = f"{found_product.get('tags', '')} {found_product.get('title', '')} {found_product.get('description_plain', '')} {found_product.get('product_type', '')}".lower()
        
        # Load the car tags and test
        with open("generated_maps_fixed.json", "r") as f:
            maps = json.load(f)
        
        cars_tags = maps.get("INTEREST_TAGS", {}).get("Cars", [])
        
        # Test each car tag to see which one matches
        for tag in cars_tags:
            if tag.lower() in text:
                print(f"  Matches car tag: '{tag}'")
                print(f"  Found in text: '{text[text.find(tag.lower())-20:text.find(tag.lower())+len(tag)+20]}'")
                break
    else:
        print(f"❌ This product is NOT in the Cars mask")
else:
    print(f"Product '{target_title}' not found")
