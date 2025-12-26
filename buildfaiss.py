#!/usr/bin/env python3
"""
Build Embeddings + FAISS Index from Shopify CSV

- Reads shopify_products_export.csv (already filtered for active + in-stock products)
- Builds embeddings (using SentenceTransformers)
- Saves both:
    1. product_search_index.pkl   → { data, embeddings }
    2. product_search_index.faiss → FAISS index
    3. product_search_index.meta.json → metadata (nprobe etc.)
"""

import pandas as pd
import numpy as np
import faiss
import pickle
import json
from sentence_transformers import SentenceTransformer

# ==== Paths ====
CSV_PATH = "shopify_products_export.csv"
PKL_PATH = "product_search_index.pkl"
FAISS_PATH = "product_search_index.faiss"
META_PATH = "product_search_index.meta.json"

# ==== Model ====
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

print("🚀 Loading data...")
df = pd.read_csv(CSV_PATH)
print(f"✅ Loaded {len(df)} rows")

# ==== Build combined_text column (match semantic_search.py) ====
print("📝 Preparing combined_text field...")
df["description_plain"] = df.get("description_plain", "").fillna("").astype(str)
df["title"] = df.get("title", "").fillna("").astype(str)
df["tags"] = df.get("tags", "").fillna("").astype(str)
df["product_type"] = df.get("product_type", "").fillna("").astype(str)
df["vendor"] = df.get("vendor", "").fillna("").astype(str)

df["combined_text"] = (
    df["title"] + " " +
    df["description_plain"] + " " +
    df["product_type"] + " " +
    df["vendor"] + " " +
    df["tags"]
).str.lower()

# ==== Generate embeddings ====
print(f"🔧 Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

print("📐 Encoding embeddings...")
embeddings = model.encode(
    df["combined_text"].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
).astype("float32")

print(f"✅ Generated embeddings: {embeddings.shape}")

# ==== Save PKL ====
print("💾 Saving PKL...")
payload = {"data": df, "embeddings": embeddings}
with open(PKL_PATH, "wb") as f:
    pickle.dump(payload, f)
print(f"✅ Saved {PKL_PATH}")

# ==== Build FAISS index ====
print("🔧 Building FAISS index...")
dim = embeddings.shape[1]
nlist = int(max(64, min(1024, round(np.sqrt(len(embeddings)) * 2))))  # cluster count
quantizer = faiss.IndexFlatIP(dim)
index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

index.train(embeddings)
index.add(embeddings)
index.nprobe = 12

print(f"✅ FAISS index built: {index.ntotal} vectors, dim={dim}, nlist={nlist}, nprobe={index.nprobe}")

# ==== Save FAISS index + metadata ====
faiss.write_index(index, FAISS_PATH)
with open(META_PATH, "w") as f:
    json.dump({"nprobe": index.nprobe, "nlist": nlist, "dim": dim}, f)

print(f"✅ Saved {FAISS_PATH} and {META_PATH}")
print("🎉 Build complete! You can now use PKL + FAISS directly in your app.")
