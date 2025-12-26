import pickle
import pandas as pd

# Load the precomputed masks to check if Cars interest is working
try:
    with open("precomputed_masks.pkl", "rb") as f:
        masks_data = pickle.load(f)
    
    print("✅ Loaded precomputed masks")
    print(f"TEXT_ARRAY shape: {masks_data['TEXT_ARRAY'].shape}")
    print(f"INTEREST_MASKS keys: {list(masks_data['INTEREST_MASKS'].keys())}")
    
    if 'Cars' in masks_data['INTEREST_MASKS']:
        cars_mask = masks_data['INTEREST_MASKS']['Cars']
        cars_count = cars_mask.sum()
        print(f"Cars mask: {cars_count} products match")
        
        # Load the data to see some examples
        with open("product_search_index.pkl", "rb") as f:
            pkl_data = pickle.load(f)
        data = pkl_data["data"]
        
        # Find some car products
        car_indices = cars_mask.nonzero()[0]
        print(f"First 5 car product indices: {car_indices[:5]}")
        
        for i in car_indices[:3]:
            row = data.iloc[i]
            print(f"Car product {i}: {row.get('title', 'No title')}")
            print(f"  Tags: {row.get('tags', '')[:100]}...")
            print()
    else:
        print("❌ Cars not found in INTEREST_MASKS")
        
except Exception as e:
    print(f"Error: {e}")
