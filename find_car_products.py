import pandas as pd
import pickle

# Load the data and search for actual car products
with open("product_search_index.pkl", "rb") as f:
    pkl_data = pickle.load(f)
data = pkl_data["data"]

print(f"Total products: {len(data)}")

# Search for actual car-related products
car_keywords = ['toy car', 'pullback car', 'remote control car', 'ride-on', 'jeep', 'police car']
car_products = []

for i, row in data.iterrows():
    text = f"{row.get('tags', '')} {row.get('title', '')} {row.get('description_plain', '')}".lower()
    if any(keyword in text for keyword in car_keywords):
        car_products.append({
            'id': row.get('id'),
            'title': row.get('title'),
            'tags': row.get('tags', '')[:100],
            'index': i
        })

print(f"Found {len(car_products)} actual car products")

for i, product in enumerate(car_products[:10]):
    print(f"\n{i+1}. {product['title']}")
    print(f"   Tags: {product['tags']}")
    print(f"   Index: {product['index']}")
