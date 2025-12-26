import pickle
import pandas as pd

# Test the updated masks to see if they're working
with open("precomputed_masks.pkl", "rb") as f:
    masks_data = pickle.load(f)

cars_mask = masks_data['INTEREST_MASKS']['Cars']
cars_count = cars_mask.sum()
print(f"Cars mask: {cars_count} products match")

# Load the data to see some examples
with open("product_search_index.pkl", "rb") as f:
    pkl_data = pickle.load(f)
data = pkl_data["data"]

# Find some car products
car_indices = cars_mask.nonzero()[0]
print(f"First 10 car product indices: {car_indices[:10]}")

# Check if these are actually car products
car_products_found = 0
for i in car_indices[:20]:
    row = data.iloc[i]
    title = row.get('title', '').lower()
    tags = row.get('tags', '').lower()
    text = f"{tags} {title}"
    
    # Check if it's actually car-related
    car_indicators = ['toy car', 'pullback', 'remote control', 'ride-on', 'jeep', 'police car', 'car toy']
    is_car = any(indicator in text for indicator in car_indicators)
    
    if is_car:
        car_products_found += 1
        print(f"\n✅ Car product {i}: {row.get('title', 'No title')}")
        print(f"   Tags: {row.get('tags', '')[:100]}...")
    else:
        print(f"\n❌ Non-car product {i}: {row.get('title', 'No title')}")
        print(f"   Tags: {row.get('tags', '')[:100]}...")

print(f"\nFound {car_products_found} actual car products out of 20 checked")
