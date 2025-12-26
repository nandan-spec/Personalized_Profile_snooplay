import pickle
import numpy as np
import pandas as pd
import faiss
import json

# Load the data
print("Loading data...")
with open("product_search_index.pkl", "rb") as f:
    payload = pickle.load(f)
DATA = payload["data"]
EMB = payload["embeddings"].astype("float32")
print(f"✅ Loaded PKL: {len(DATA)} rows, embeddings {EMB.shape}")

# Load FAISS index
print("Loading FAISS index...")
index = faiss.read_index("product_search_index.faiss")
index.nprobe = 12
print(f"✅ Loaded FAISS index: {index.ntotal} vectors")

def simple_text_search(query, k=50):
    """Simple text-based search without semantic embeddings"""
    results = []
    query_lower = query.lower()
    
    for i, row in DATA.iterrows():
        text = f"{row.get('title', '')} {row.get('tags', '')} {row.get('product_type', '')}".lower()
        if any(word in text for word in query_lower.split()):
            results.append({
                'rank': len(results) + 1,
                'score': 0.5,  # Default score for text search
                'id': row.get('id'),
                'title': row.get('title'),
                'tags': row.get('tags', ''),
                'product_type': row.get('product_type', ''),
                'vendor': row.get('vendor', ''),
                'price': row.get('variant_price', 0)
            })
            if len(results) >= k:
                break
    
    return results

def faiss_search_with_centroid(centroid, k=50):
    """Search using a precomputed centroid"""
    if centroid is None:
        return []
    
    # Normalize the centroid
    centroid_norm = np.linalg.norm(centroid, axis=1, keepdims=True) + 1e-8
    centroid_normalized = centroid / centroid_norm
    
    # Search
    D, I = index.search(centroid_normalized, k)
    
    results = []
    for i, (score, idx) in enumerate(zip(D[0], I[0])):
        if idx < 0 or idx >= len(DATA):
            continue
        row = DATA.iloc[int(idx)].to_dict()
        results.append({
            'rank': i + 1,
            'score': float(score),
            'id': row.get('id'),
            'title': row.get('title'),
            'tags': row.get('tags', ''),
            'product_type': row.get('product_type', ''),
            'vendor': row.get('vendor', ''),
            'price': row.get('variant_price', 0)
        })
    
    return results

# Load the maps to get centroids
print("Loading maps...")
with open("generated_maps_fixed_v2.json", "r") as f:
    maps = json.load(f)

INTEREST_CENTROIDS = maps.get("INTEREST_CENTROIDS", {})
print(f"✅ Loaded {len(INTEREST_CENTROIDS)} interest centroids")

# Test queries for car products suitable for 0-2 year olds
queries = [
    "toy cars for babies",
    "car toys for toddlers", 
    "baby car toys",
    "toddler car toys",
    "cars for 0-2 years",
    "baby vehicles",
    "toddler vehicles",
    "push cars for babies",
    "pull back cars for toddlers"
]

print("\n" + "="*80)
print("TEXT SEARCH RESULTS FOR CAR PRODUCTS (0-2 YEARS)")
print("="*80)

for query in queries:
    print(f"\n🔍 Query: '{query}'")
    print("-" * 60)
    
    results = simple_text_search(query, k=20)
    
    if not results:
        print("❌ No results found")
        continue
    
    for result in results[:10]:  # Show top 10
        tags = result['tags'][:100] + "..." if len(str(result['tags'])) > 100 else result['tags']
        print(f"{result['rank']:2d}. Score: {result['score']:.3f} | ID: {result['id']}")
        print(f"    Title: {result['title']}")
        print(f"    Tags: {tags}")
        print(f"    Type: {result['product_type']} | Vendor: {result['vendor']}")
        print(f"    Price: {result['price']}")
        print()

# Try using the Cars centroid
print("\n" + "="*80)
print("FAISS SEARCH USING CARS CENTROID")
print("="*80)

cars_centroid = INTEREST_CENTROIDS.get("Cars")
if cars_centroid is not None:
    cars_centroid = np.array(cars_centroid).reshape(1, -1)
    print(f"✅ Using Cars centroid with shape: {cars_centroid.shape}")
    
    results = faiss_search_with_centroid(cars_centroid, k=30)
    print(f"\n🔍 Cars centroid search results:")
    print("-" * 60)
    
    for result in results[:15]:
        tags = result['tags'][:100] + "..." if len(str(result['tags'])) > 100 else result['tags']
        print(f"{result['rank']:2d}. Score: {result['score']:.3f} | ID: {result['id']}")
        print(f"    Title: {result['title']}")
        print(f"    Tags: {tags}")
        print(f"    Type: {result['product_type']} | Vendor: {result['vendor']}")
        print(f"    Price: {result['price']}")
        print()
else:
    print("❌ Cars centroid not found")

# Also try a broader search for any car-related products
print("\n" + "="*80)
print("BROADER CAR SEARCH (ALL AGES)")
print("="*80)

broad_results = simple_text_search("toy cars vehicles", k=30)
print(f"\n🔍 Query: 'toy cars vehicles'")
print("-" * 60)

for result in broad_results[:15]:
    tags = result['tags'][:100] + "..." if len(str(result['tags'])) > 100 else result['tags']
    print(f"{result['rank']:2d}. Score: {result['score']:.3f} | ID: {result['id']}")
    print(f"    Title: {result['title']}")
    print(f"    Tags: {tags}")
    print(f"    Type: {result['product_type']} | Vendor: {result['vendor']}")
    print(f"    Price: {result['price']}")
    print()
