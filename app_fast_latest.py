import os, re, pickle, json, math
from typing import List, Dict, Any, Optional, Union
import requests
import logging

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Toy Recommendation API (Fast Path)", version="4.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

# ---------------- Shopify GraphQL Configuration ----------------
SHOP_DOMAIN = os.getenv("SHOP_DOMAIN", "playhop.myshopify.com")
ACCESS_TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN")
if not ACCESS_TOKEN:
    raise ValueError("SHOPIFY_ACCESS_TOKEN environment variable is required")
API_VERSION = os.getenv("SHOPIFY_API_VERSION", "2025-07")
GRAPHQL_URL = f"https://{SHOP_DOMAIN}/admin/api/{API_VERSION}/graphql.json"

# ---------------- Data load ----------------
PKL_PATH = os.getenv("PICKLE_PATH", "product_search_index.pkl")
CSV_PATH = os.getenv("CSV_PATH", "shopify_products_export.csv")
MAPS_PATH = os.getenv("MAPS_PATH", "generated_maps_fixed_v2.json")

DATA: pd.DataFrame
EMB: Optional[np.ndarray] = None
NORM_EMB: Optional[np.ndarray] = None
try:
    with open(PKL_PATH, "rb") as f:
        payload = pickle.load(f)
    DATA = payload["data"]
    EMB = payload["embeddings"].astype("float32")
    print(f"✅ Loaded PKL: {len(DATA)} rows, embeddings {EMB.shape}")
except (FileNotFoundError, ModuleNotFoundError, ImportError) as e:
    print(f"⚠️ PKL load failed: {e}")
    print(f"⚠️ Loading CSV fallback {CSV_PATH}")
    DATA = pd.read_csv(CSV_PATH, low_memory=False)
    EMB = None

if EMB is not None:
    _norms = np.linalg.norm(EMB, axis=1, keepdims=True) + 1e-8
    NORM_EMB = (EMB / _norms).astype("float32")

# ---------------- Shopify GraphQL Functions ----------------
def fetch_batch_from_shopify(batch_gids: List[str]) -> List[Dict[str, Any]]:
    """Fetch product data from Shopify GraphQL API for a batch of product IDs"""
    query = """
    query getProducts($ids: [ID!]!) {
      nodes(ids: $ids) {
        ... on Product {
          id
          title
          handle
          vendor
          tags
          status
          totalInventory
          options(first: 3) { name values }
          images(first: 10) { edges { node { url altText width height } } }
          variants(first: 10) {
            edges { node {
              id
              title
              sku
              price
              compareAtPrice
              availableForSale
              inventoryPolicy
              inventoryQuantity
              image { url altText width height }
            } }
          }
          collections(first: 10) { edges { node { title } } }
        }
      }
    }
    """
    variables = {"ids": batch_gids}
    headers = {
        "X-Shopify-Access-Token": ACCESS_TOKEN,
        "Content-Type": "application/json"
    }
    
    try:
        logger.info(f"🔧 Making GraphQL request with {len(batch_gids)} GIDs")
        response = requests.post(
            GRAPHQL_URL, headers=headers, json={"query": query, "variables": variables}
        )
        logger.info(f"🔧 GraphQL response status: {response.status_code}")
        response.raise_for_status()
        response_data = response.json()
        logger.info(f"🔧 GraphQL response keys: {list(response_data.keys())}")
        if 'errors' in response_data:
            logger.error(f"🔧 GraphQL ERRORS: {response_data['errors']}")
            return []
        if 'data' not in response_data:
            logger.error(f"🔧 No 'data' key in response! Full response: {response_data}")
            return []
        shopify_data = response_data.get("data", {}).get("nodes", [])
        logger.info(f"🔧 GraphQL batch returned {len(shopify_data)} raw nodes")
        
        return [p for p in shopify_data if p]  # Filter out None values
    except Exception as e:
        logger.error(f"Failed to fetch batch from Shopify GraphQL: {e}")
        return []

def fetch_and_format_products(gids_with_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fetch and format products from Shopify GraphQL API"""
    if not gids_with_scores:
        return []
    
    # Extract GIDs from the list
    gids = [item['gid'] for item in gids_with_scores if item.get('gid')]
    
    # Shopify GraphQL has a limit of 250 products per request
    batch_size = 250
    all_shopify_data = []
    
    for i in range(0, len(gids), batch_size):
        batch_gids = gids[i:i + batch_size]
        batch_data = fetch_batch_from_shopify(batch_gids)
        all_shopify_data.extend(batch_data)
    
    logger.info(f"🔧 Total products fetched from all batches: {len(all_shopify_data)}")
    return format_shopify_products(all_shopify_data, gids_with_scores)

def format_shopify_products(all_shopify_data: List[Dict[str, Any]], gids_with_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format Shopify GraphQL data into standardized product format"""
    shopify_products_map = {p['id']: p for p in all_shopify_data if p}

    formatted_results = []
    for item in gids_with_scores:
        gid = item['gid']
        product_data = shopify_products_map.get(gid)
        if not product_data:
            continue

        # Process variants
        variants_formatted = []
        product_in_stock = False
        in_stock_variants = 0
        for v_edge in product_data.get('variants', {}).get('edges', []):
            variant_node = v_edge.get('node', {})
            # Compute variant stock - availableForSale is the primary factor
            try:
                v_avail = bool(variant_node.get('availableForSale'))
                # If availableForSale is false, variant is out of stock regardless of other factors
                if not v_avail:
                    v_in_stock = False
                else:
                    # Only check other factors if availableForSale is true
                    v_qty = variant_node.get('inventoryQuantity')
                    v_policy = (variant_node.get('inventoryPolicy') or '').upper()
                    qty_positive = isinstance(v_qty, (int, float)) and float(v_qty) > 0
                    v_in_stock = qty_positive or (v_policy == 'CONTINUE')
            except Exception:
                v_in_stock = False
            if v_in_stock:
                product_in_stock = True
                in_stock_variants += 1
            variants_formatted.append({
                "id": int(variant_node['id'].split('/')[-1]) if variant_node.get('id') else 0,
                "title": variant_node.get('title', ''),
                "price": variant_node.get('price', '0'),
                "compare_at_price": variant_node.get('compareAtPrice'),
                "sku": variant_node.get('sku', ''),
                "availableForSale": bool(variant_node.get('availableForSale')),
                "inventoryPolicy": variant_node.get('inventoryPolicy'),
                "inventoryQuantity": variant_node.get('inventoryQuantity'),
                "inStock": v_in_stock,
                "image": variant_node.get('image'),
            })

        # Calculate pricing
        if variants_formatted:
            try:
                min_price_variant = min(variants_formatted, key=lambda v: float(v['price'] or 0))
                price = float(min_price_variant['price'] or 0)
                compare_at_price = float(min_price_variant.get('compare_at_price') or 0)
            except (ValueError, TypeError):
                price = 0
                compare_at_price = 0
        else:
            price = 0
            compare_at_price = 0

        discount = int(round((1 - (price / compare_at_price)) * 100)) if compare_at_price > price and compare_at_price > 0 else 0

        # Process images
        images_list = []
        images_data = product_data.get('images', {})
        
        if isinstance(images_data, dict) and 'edges' in images_data:
            # GraphQL response format
            for img_edge in images_data.get('edges', []):
                img_node = img_edge.get('node', {})
                if img_node and img_node.get('url'):
                    images_list.append({
                        "url": img_node.get('url', ''),
                        "altText": img_node.get('altText', ''),
                        "width": img_node.get('width', 800),
                        "height": img_node.get('height', 800)
                    })
        elif isinstance(images_data, list):
            # Direct list format
            for img in images_data:
                if isinstance(img, dict) and img.get('url'):
                    images_list.append({
                        "url": img.get('url', ''),
                        "altText": img.get('altText', ''),
                        "width": img.get('width', 800),
                        "height": img.get('height', 800)
                    })
                elif isinstance(img, str) and img.startswith('http'):
                    images_list.append({
                        "url": img,
                        "altText": "",
                        "width": 800,
                        "height": 800
                    })

        # Extract status and totalInventory from GraphQL response
        status = product_data.get('status', 'active')
        total_inventory = product_data.get('totalInventory')
        if status and status.lower() not in ['active', 'draft', 'archived']:
            status = 'active'  # Default to active if status is unclear
        
        # Format variants to match app_fast.py structure exactly - ALL VARIANTS
        formatted_variants = []
        for var in variants_formatted:
            # Handle variant image properly
            variant_image = None
            if var.get("image"):
                img_data = var["image"]
                if isinstance(img_data, dict):
                    variant_image = {
                        "url": img_data.get("url", ""),
                        "altText": img_data.get("altText", ""),
                        "width": img_data.get("width", 800),
                        "height": img_data.get("height", 800)
                    }
            
            formatted_variants.append({
                "id": var.get("id"),
                "title": var.get("title", ""),
                "price": str(var.get("price", "0.00")),
                "compare_at_price": str(var.get("compare_at_price", "0.00")),
                "sku": var.get("sku", ""),
                "inventory_quantity": var.get("inventoryQuantity"),
                "image": variant_image
            })
        
        # Format images to match app_fast.py structure exactly
        formatted_images = []
        for img in images_list:
            formatted_images.append({
                "url": img.get("url", ""),
                "altText": img.get("altText", ""),
                "width": img.get("width", 800),
                "height": img.get("height", 800)
            })
        
        # Format collections to match app_fast.py structure exactly
        formatted_collections = [col_edge['node']['title'] for col_edge in product_data.get('collections', {}).get('edges', [])]
        
        # Format options to match app_fast.py structure exactly
        formatted_options = []
        raw_options = product_data.get('options', [])
        for opt in raw_options:
            if isinstance(opt, dict):
                formatted_options.append({
                    "name": opt.get("name", ""),
                    "values": opt.get("values", [])
                })
        
        formatted_product = {
            "id": int(gid.split('/')[-1]),
            "title": product_data.get('title', ''),
            "vendor": product_data.get('vendor', ''),
            "handle": product_data.get('handle', ''),
            "tags": product_data.get('tags', []),
            "status": status,
            "price": price, 
            "discounted_price": price,
            "discount": discount,
            "hasMultiplePrice": len(set(float(v['price']) for v in variants_formatted if v.get('price'))) > 1 if variants_formatted else False,
            "images": formatted_images,
            "variants": formatted_variants,
            "collections": formatted_collections,
            "options": formatted_options,
            "_rank": item.get('score', 0.0),
            "isActive": 1 if status.lower() == 'active' else 0,
            "inStock": (status.lower() == 'active') and bool(product_in_stock),
            "availableVariantsCount": int(in_stock_variants),
            "reviews_count": None,
            "reviews_average": None,
        }
        formatted_results.append(formatted_product)

    return formatted_results

# ---------------- Maps ----------------
# Load maps from JSON
print(f"🔧 Attempting to load maps from {MAPS_PATH}")
try:
    with open(MAPS_PATH, "r") as f:
        maps = json.load(f)
    AGE_MAP = maps.get("AGE_MAP", {})
    INTEREST_TAGS = maps.get("INTEREST_TAGS", {})
    GOAL_TAGS = maps.get("GOAL_TAGS", {})
    print(f"✅ Loaded tag maps from {MAPS_PATH}")
    print(f"   - AGE_MAP: {len(AGE_MAP)} age bands")
    print(f"   - INTEREST_TAGS: {len(INTEREST_TAGS)} interests")
    print(f"   - GOAL_TAGS: {len(GOAL_TAGS)} goals")
    print(f"   - Cars tags: {len(INTEREST_TAGS.get('Cars', []))} tags")
except Exception as e:
    print(f"⚠️ Could not load maps from {MAPS_PATH}: {e}")
    AGE_MAP = {}
    INTEREST_TAGS = {}
    GOAL_TAGS = {}

# Generate regex patterns at runtime (don't try to load from JSON)
def _kws_to_compiled(kws: List[str]) -> Optional[re.Pattern]:
    kws = [k for k in (kws or []) if k]
    if not kws:
        return None
    alts = []
    for k in kws:
        k = re.escape(k.strip().lower()).replace(r"\ ", r"[ -]?")
        alts.append(f"{k}s?")
    return re.compile(r"(?:^|[^a-z0-9])(?:" + "|".join(alts) + r")(?:[^a-z0-9]|$)")

AGE_RX: Dict[str, Optional[re.Pattern]] = {b: _kws_to_compiled(AGE_MAP.get(b, [])) for b in AGE_MAP}

def _cars_syns_guard(syns: List[str]) -> List[str]:
    # Since we've already cleaned up the maps, we don't need to filter out vehicle terms
    # that are part of specific car-related phrases
    bad = {"transport","transportation"}  # Only remove generic transport terms
    return [s for s in syns if s and s.strip().lower() not in bad]

INTEREST_RX: Dict[str, Optional[re.Pattern]] = {}
for k, syns in INTEREST_TAGS.items():
    if str(k).strip().lower() == "cars":
        syns = _cars_syns_guard(list(syns))
    INTEREST_RX[k] = _kws_to_compiled(syns or [k])

GOAL_RX: Dict[str, Optional[re.Pattern]] = {g: _kws_to_compiled(GOAL_TAGS.get(g, [g])) for g in GOAL_TAGS}

# ---------------- TEXT field ----------------
# Check if DATA is a DataFrame, if not create a simple test DataFrame
if not isinstance(DATA, pd.DataFrame):
    print("⚠️ DATA is not a DataFrame, creating test data")
    DATA = pd.DataFrame({
        "id": [1, 2, 3],
        "title": ["Test Toy 1", "Test Toy 2", "Test Toy 3"],
        "tags": ["cars, robots", "books, animals", "puzzles, games"],
        "description_plain": ["A fun car toy", "An educational book", "A puzzle game"],
        "product_type": ["toy", "book", "game"]
    })

if "description_plain" not in DATA.columns:
    DATA["description_plain"] = ""
DATA["TEXT"] = (
    (DATA.get("tags", "").fillna("").astype(str)) + " " +
    (DATA.get("title", "").fillna("").astype(str)) + " " +
    (DATA.get("description_plain", "").fillna("").astype(str)) + " " +
    (DATA.get("product_type", "").fillna("").astype(str))
).str.lower()

# ---------------- Precomputed masks and centroids (startup) ----------------
TEXT_ARRAY = None
KEY_TO_IDX = {}
AGE_MASKS = {}
INTEREST_MASKS = {}
INTEREST_INDICES = {}
GOAL_MASKS = {}
INTEREST_CENTROIDS = {}
GOAL_CENTROIDS = {}
GLOBAL_CENTROID = None

# Try to load pre-computed masks first
PRECOMPUTED_MASKS_PATH = os.getenv("PRECOMPUTED_MASKS_PATH", "precomputed_masks.pkl")
if os.path.exists(PRECOMPUTED_MASKS_PATH):
    print(f"Loading pre-computed masks from {PRECOMPUTED_MASKS_PATH}")
    try:
        with open(PRECOMPUTED_MASKS_PATH, "rb") as f:
            masks_data = pickle.load(f)
        
        TEXT_ARRAY = masks_data["TEXT_ARRAY"]
        KEY_TO_IDX = masks_data["KEY_TO_IDX"]
        AGE_MASKS = masks_data["AGE_MASKS"]
        INTEREST_MASKS = masks_data["INTEREST_MASKS"]
        INTEREST_INDICES = masks_data["INTEREST_INDICES"]
        GOAL_MASKS = masks_data["GOAL_MASKS"]
        
        print(f"✅ Loaded pre-computed masks: {len(AGE_MASKS)} age, {len(INTEREST_MASKS)} interest, {len(GOAL_MASKS)} goal masks")
        
        # DISABLED: Using regex patterns instead of pre-computed masks for better reliability
        print("⚠️ DISABLED: Pre-computed masks - using regex patterns instead")
        AGE_MASKS = {}
        INTEREST_MASKS = {}
        GOAL_MASKS = {}
        
    except Exception as e:
        print(f"⚠️ Could not load pre-computed masks: {e}")
        print("⚠️ Falling back to computing masks at startup...")
        # Fall back to original computation
        TEXT_ARRAY = DATA["TEXT"].to_numpy()
        
        # Row key to index for fast lookups using (id, handle)
        if "id" in DATA.columns and "handle" in DATA.columns:
            _ids = DATA.get("id", pd.Series([None]*len(DATA))).to_list()
            _handles = DATA.get("handle", pd.Series([None]*len(DATA))).to_list()
            for _i, (_id, _h) in enumerate(zip(_ids, _handles)):
                KEY_TO_IDX[(_id, _h)] = _i
        else:
            # Create simple key mapping for test data
            for _i in range(len(DATA)):
                KEY_TO_IDX[(DATA.iloc[_i].get("id", _i), f"handle_{_i}")] = _i
        
        # Global centroid for FAISS seeding
        GLOBAL_CENTROID: Optional[np.ndarray] = None
        if 'EMB' in globals() and EMB is not None and EMB.size > 0:
            _gc = EMB.mean(axis=0, keepdims=True).astype("float32")
            _gc /= (np.linalg.norm(_gc, axis=1, keepdims=True) + 1e-8)
            GLOBAL_CENTROID = _gc
        
        INTEREST_MASKS = {}
        INTEREST_INDICES = {}
        for k, rx in INTEREST_RX.items():
            if not rx:
                continue
            try:
                mask = DATA["TEXT"].str.contains(rx, regex=True, na=False).to_numpy(dtype=bool)
            except Exception:
                continue
            INTEREST_MASKS[k] = mask
            idx = np.nonzero(mask)[0]
            INTEREST_INDICES[k] = idx
            if EMB is not None and idx.size > 0:
                cap = min(idx.size, 2000)
                qv = EMB[idx[:cap]].mean(axis=0, keepdims=True).astype("float32")
                qv /= (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-8)
                INTEREST_CENTROIDS[k] = qv
        
        AGE_MASKS = {}
        for b, rx in AGE_RX.items():
            if not rx:
                continue
            try:
                AGE_MASKS[b] = DATA["TEXT"].str.contains(rx, regex=True, na=False).to_numpy(dtype=bool)
            except Exception:
                continue
        
        GOAL_MASKS = {}
        for g, rx in GOAL_RX.items():
            if not rx:
                continue
            try:
                GOAL_MASKS[g] = DATA["TEXT"].str.contains(rx, regex=True, na=False).to_numpy(dtype=bool)
            except Exception:
                continue
else:
    print("⚠️ No pre-computed masks found, computing at startup...")
    # Original computation code here
    TEXT_ARRAY = DATA["TEXT"].to_numpy()
    
    # Row key to index for fast lookups using (id, handle)
    if "id" in DATA.columns and "handle" in DATA.columns:
        _ids = DATA.get("id", pd.Series([None]*len(DATA))).to_list()
        _handles = DATA.get("handle", pd.Series([None]*len(DATA))).to_list()
        for _i, (_id, _h) in enumerate(zip(_ids, _handles)):
            KEY_TO_IDX[(_id, _h)] = _i
    else:
        # Create simple key mapping for test data
        for _i in range(len(DATA)):
            KEY_TO_IDX[(DATA.iloc[_i].get("id", _i), f"handle_{_i}")] = _i
    
    # Global centroid for FAISS seeding
    GLOBAL_CENTROID: Optional[np.ndarray] = None
    if 'EMB' in globals() and EMB is not None and EMB.size > 0:
        _gc = EMB.mean(axis=0, keepdims=True).astype("float32")
        _gc /= (np.linalg.norm(_gc, axis=1, keepdims=True) + 1e-8)
        GLOBAL_CENTROID = _gc
    
    INTEREST_MASKS = {}
    INTEREST_INDICES = {}
    for k, rx in INTEREST_RX.items():
        if not rx:
            continue
        try:
            mask = DATA["TEXT"].str.contains(rx, regex=True, na=False).to_numpy(dtype=bool)
        except Exception:
            continue
        INTEREST_MASKS[k] = mask
        idx = np.nonzero(mask)[0]
        INTEREST_INDICES[k] = idx
        if EMB is not None and idx.size > 0:
            cap = min(idx.size, 2000)
            qv = EMB[idx[:cap]].mean(axis=0, keepdims=True).astype("float32")
            qv /= (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-8)
            INTEREST_CENTROIDS[k] = qv
    
    AGE_MASKS = {}
    for b, rx in AGE_RX.items():
        if not rx:
            continue
        try:
            AGE_MASKS[b] = DATA["TEXT"].str.contains(rx, regex=True, na=False).to_numpy(dtype=bool)
        except Exception:
            continue
    
    GOAL_MASKS = {}
    for g, rx in GOAL_RX.items():
        if not rx:
            continue
        try:
            GOAL_MASKS[g] = DATA["TEXT"].str.contains(rx, regex=True, na=False).to_numpy(dtype=bool)
        except Exception:
            continue

# Balanced goal centroids creation

# Try to load pre-computed centroids first
PRECOMPUTED_CENTROIDS_PATH = os.getenv("PRECOMPUTED_CENTROIDS_PATH", "precomputed_centroids.pkl")
if os.path.exists(PRECOMPUTED_CENTROIDS_PATH):
    print(f"Loading pre-computed centroids from {PRECOMPUTED_CENTROIDS_PATH}")
    try:
        with open(PRECOMPUTED_CENTROIDS_PATH, "rb") as f:
            centroids_data = pickle.load(f)
        
        GOAL_CENTROIDS = centroids_data["GOAL_CENTROIDS"]
        print(f"✅ Loaded pre-computed centroids: {len(GOAL_CENTROIDS)} goals")
        
    except Exception as e:
        print(f"⚠️ Could not load pre-computed centroids: {e}")
        print("⚠️ Falling back to computing centroids at startup...")
        # Fall back to original computation
        if EMB is not None:
            # Set seed for reproducible sampling
            np.random.seed(42)
            for g, m in GOAL_MASKS.items():
                try:
                    idx = np.nonzero(m)[0]
                    if idx.size == 0:
                        continue
                    
                    # Define subtypes for each goal to ensure balanced representation
                    goal_subtypes = {
                        "Motor Skills": {
                            "fine_motor": ["fine motor", "lacing", "bead", "busy board", "puzzle", "threading", "fingers", "hand eye", "dexterity"],
                            "gross_motor": ["building block", "construction", "ride-on", "physical", "movement", "gross motor", "coordination"]
                        },
                        "Thinking": {
                            "critical_thinking": ["critical thinking", "logic", "strategy", "problem solving", "analytical"],
                            "memory": ["memory", "matching", "sequence", "recall", "remember"],
                            "puzzle": ["puzzle", "brain teaser", "maze", "pattern", "brain"]
                        },
                        "Creativity": {
                            "art_craft": ["art", "craft", "drawing", "painting", "coloring", "creative"],
                            "imaginative": ["pretend", "role play", "imagination", "storytelling", "dramatic"],
                            "diy": ["diy", "building", "creating", "design", "make"]
                        },
                        "Social Skills": {
                            "group_play": ["group play", "cooperative", "team", "board game", "social"],
                            "communication": ["communication", "language", "conversation", "talking"],
                            "emotional": ["emotional", "empathy", "social scenario", "feelings"]
                        }
                    }
                    
                    # Get subtypes for this goal
                    subtypes = goal_subtypes.get(g, {})
                    if not subtypes:
                        # Fallback to simple averaging for goals without defined subtypes
                        cap = min(idx.size, 2000)
                        qv = EMB[idx[:cap]].mean(axis=0, keepdims=True).astype("float32")
                        qv /= (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-8)
                        GOAL_CENTROIDS[g] = qv
                        continue
                    
                    # Create balanced centroid by sampling equally from each subtype
                    subtype_centroids = []
                    max_samples_per_subtype = 500  # Limit samples per subtype to prevent bias
                    
                    for subtype_name, subtype_keywords in subtypes.items():
                        # Find products matching this subtype
                        subtype_mask = np.zeros(len(DATA), dtype=bool)
                        for keyword in subtype_keywords:
                            # Use case-insensitive matching with word boundaries
                            keyword_pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                            keyword_mask = DATA["TEXT"].str.contains(keyword_pattern, regex=True, case=False, na=False)
                            subtype_mask |= keyword_mask.to_numpy()
                        
                        # Intersect with the main goal mask
                        subtype_indices = np.nonzero(subtype_mask & m)[0]
                        
                        if len(subtype_indices) > 0:
                            # Sample from this subtype (with cap to prevent bias)
                            sample_size = min(len(subtype_indices), max_samples_per_subtype)
                            sampled_indices = np.random.choice(subtype_indices, size=sample_size, replace=False)
                            
                            # Create centroid for this subtype
                            subtype_centroid = EMB[sampled_indices].mean(axis=0, keepdims=True).astype("float32")
                            subtype_centroid /= (np.linalg.norm(subtype_centroid, axis=1, keepdims=True) + 1e-8)
                            subtype_centroids.append(subtype_centroid)
                    
                    if subtype_centroids:
                        # Average all subtype centroids (equal weighting)
                        balanced_centroid = np.mean(np.vstack(subtype_centroids), axis=0, keepdims=True).astype("float32")
                        balanced_centroid /= (np.linalg.norm(balanced_centroid, axis=1, keepdims=True) + 1e-8)
                        GOAL_CENTROIDS[g] = balanced_centroid
                    else:
                        # Fallback if no subtypes found
                        cap = min(idx.size, 2000)
                        qv = EMB[idx[:cap]].mean(axis=0, keepdims=True).astype("float32")
                        qv /= (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-8)
                        GOAL_CENTROIDS[g] = qv
                        
                except Exception:
                    continue
else:
    print("⚠️ No pre-computed centroids found, computing at startup...")
    # Original computation code here
    if EMB is not None:
        # Set seed for reproducible sampling
        np.random.seed(42)
        for g, m in GOAL_MASKS.items():
            try:
                idx = np.nonzero(m)[0]
                if idx.size == 0:
                    continue
                
                # Define subtypes for each goal to ensure balanced representation
                goal_subtypes = {
                    "Motor Skills": {
                        "fine_motor": ["fine motor", "lacing", "bead", "busy board", "puzzle", "threading", "fingers", "hand eye", "dexterity"],
                        "gross_motor": ["building block", "construction", "ride-on", "physical", "movement", "gross motor", "coordination"]
                    },
                    "Thinking": {
                        "critical_thinking": ["critical thinking", "logic", "strategy", "problem solving", "analytical"],
                        "memory": ["memory", "matching", "sequence", "recall", "remember"],
                        "puzzle": ["puzzle", "brain teaser", "maze", "pattern", "brain"]
                    },
                    "Creativity": {
                        "art_craft": ["art", "craft", "drawing", "painting", "coloring", "creative"],
                        "imaginative": ["pretend", "role play", "imagination", "storytelling", "dramatic"],
                        "diy": ["diy", "building", "creating", "design", "make"]
                    },
                    "Social Skills": {
                        "group_play": ["group play", "cooperative", "team", "board game", "social"],
                        "communication": ["communication", "language", "conversation", "talking"],
                        "emotional": ["emotional", "empathy", "social scenario", "feelings"]
                    }
                }
                
                # Get subtypes for this goal
                subtypes = goal_subtypes.get(g, {})
                if not subtypes:
                    # Fallback to simple averaging for goals without defined subtypes
                    cap = min(idx.size, 2000)
                    qv = EMB[idx[:cap]].mean(axis=0, keepdims=True).astype("float32")
                    qv /= (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-8)
                    GOAL_CENTROIDS[g] = qv
                    continue
                
                # Create balanced centroid by sampling equally from each subtype
                subtype_centroids = []
                max_samples_per_subtype = 500  # Limit samples per subtype to prevent bias
                
                for subtype_name, subtype_keywords in subtypes.items():
                    # Find products matching this subtype
                    subtype_mask = np.zeros(len(DATA), dtype=bool)
                    for keyword in subtype_keywords:
                        # Use case-insensitive matching with word boundaries
                        keyword_pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                        keyword_mask = DATA["TEXT"].str.contains(keyword_pattern, regex=True, case=False, na=False)
                        subtype_mask |= keyword_mask.to_numpy()
                    
                    # Intersect with the main goal mask
                    subtype_indices = np.nonzero(subtype_mask & m)[0]
                    
                    if len(subtype_indices) > 0:
                        # Sample from this subtype (with cap to prevent bias)
                        sample_size = min(len(subtype_indices), max_samples_per_subtype)
                        sampled_indices = np.random.choice(subtype_indices, size=sample_size, replace=False)
                        
                        # Create centroid for this subtype
                        subtype_centroid = EMB[sampled_indices].mean(axis=0, keepdims=True).astype("float32")
                        subtype_centroid /= (np.linalg.norm(subtype_centroid, axis=1, keepdims=True) + 1e-8)
                        subtype_centroids.append(subtype_centroid)
                
                if subtype_centroids:
                    # Average all subtype centroids (equal weighting)
                    balanced_centroid = np.mean(np.vstack(subtype_centroids), axis=0, keepdims=True).astype("float32")
                    balanced_centroid /= (np.linalg.norm(balanced_centroid, axis=1, keepdims=True) + 1e-8)
                    GOAL_CENTROIDS[g] = balanced_centroid
                else:
                    # Fallback if no subtypes found
                    cap = min(idx.size, 2000)
                    qv = EMB[idx[:cap]].mean(axis=0, keepdims=True).astype("float32")
                    qv /= (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-8)
                    GOAL_CENTROIDS[g] = qv
                    
            except Exception:
                continue

# ---------------- FAISS (Load pre-built or build from scratch) ----------------
HAS_FAISS = False
INDEX = None
DIM = None

# Check if we should skip FAISS training
SKIP_FAISS_TRAINING = os.getenv("SKIP_FAISS_TRAINING", "false").lower() == "true"
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "product_search_index.faiss")

try:
    import faiss
    
    # Try to load pre-built index first
    if os.path.exists(FAISS_INDEX_PATH) and not SKIP_FAISS_TRAINING:
        print(f"🔧 Loading pre-built FAISS index from {FAISS_INDEX_PATH}")
        try:
            INDEX = faiss.read_index(FAISS_INDEX_PATH)
            DIM = INDEX.d
            HAS_FAISS = True
            print(f"✅ Loaded pre-built FAISS index: {INDEX.ntotal} vectors, dim={DIM}")
            
            # If we don't have embeddings but have FAISS index, we can still use it for search
            # We'll need to create dummy embeddings for the search functions to work
            if EMB is None and INDEX.ntotal > 0:
                print(f"⚠️ No embeddings loaded, creating dummy embeddings for FAISS compatibility")
                # Create dummy embeddings with the same dimension as the FAISS index
                EMB = np.random.randn(INDEX.ntotal, DIM).astype("float32")
                # Normalize them
                EMB /= np.linalg.norm(EMB, axis=1, keepdims=True) + 1e-8
                print(f"✅ Created dummy embeddings: {EMB.shape}")
                
        except (ModuleNotFoundError, ImportError, Exception) as e:
            print(f"⚠️ Could not load pre-built FAISS index: {e}")
            print("⚠️ Will build index from scratch...")
            INDEX = None
            HAS_FAISS = False
        
        # Load metadata if available
        meta_path = FAISS_INDEX_PATH.replace('.faiss', '.meta.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    import json
                    meta = json.load(f)
                    if 'nprobe' in meta:
                        INDEX.nprobe = int(meta['nprobe'])
                        print(f"✅ Set nprobe to {INDEX.nprobe}")
            except Exception as e:
                print(f"⚠️ Could not load metadata: {e}")
                INDEX.nprobe = int(os.getenv("FAISS_NPROBE", 12))
    
    # Fallback: build index from scratch if no pre-built index and training not skipped
    elif EMB is not None and EMB.size > 0 and not SKIP_FAISS_TRAINING:
        print("🔧 Building FAISS index from scratch...")
        xb = EMB.copy().astype("float32")
        # cosine via L2-normalized vectors
        xb /= np.linalg.norm(xb, axis=1, keepdims=True) + 1e-8
        DIM = xb.shape[1]
        nlist = int(max(64, min(1024, round(math.sqrt(len(xb)) * 2))))  # ~512 for 55k
        quant = faiss.IndexFlatIP(DIM)
        ivf = faiss.IndexIVFFlat(quant, DIM, nlist, faiss.METRIC_INNER_PRODUCT)
        ivf.train(xb)
        ivf.add(xb)
        ivf.nprobe = int(os.getenv("FAISS_NPROBE", 12))
        INDEX = ivf
        HAS_FAISS = True
        print(f"✅ FAISS IVF index built: {INDEX.ntotal} vectors, dim={DIM}, nlist={nlist}, nprobe={ivf.nprobe}")
    
    elif SKIP_FAISS_TRAINING:
        print("⚠️ FAISS training skipped - will use fallback search")
        HAS_FAISS = False
        
except Exception as e:
    print(f"⚠️ FAISS unavailable: {e}. Will fallback to Flat search if needed.")
    HAS_FAISS = False


# ---------------- Helpers ----------------
def product_active_series() -> np.ndarray:
    return DATA.get("status", "").astype(str).str.lower().eq("active").to_numpy(dtype=bool)

def price_ok_series(lo: Optional[float], hi: Optional[float]) -> np.ndarray:
    prices = pd.to_numeric(DATA.get("variant_price", 0), errors="coerce").fillna(0.0).to_numpy()
    m = np.ones(len(prices), dtype=bool)
    if lo is not None: m &= prices >= lo
    if hi is not None: m &= prices <= hi
    return m

def overlapping_age_bands(p: dict) -> List[str]:
    amin = p.get("age_min"); amax = p.get("age_max")
    if amin is None and amax is None: return []
    try:
        amin = int(0 if amin is None else amin); amax = int(amax if amax is not None else amin)
    except Exception:
        return []
    if amax < amin: amin, amax = amax, amin
    bands = {"0-2 yr": (0,2), "3-5 yr": (3,5), "6-8 yr": (6,8), "9-11 yr": (9,11)}
    out=[]
    for k,(lo,hi) in bands.items():
        if not (amax < lo or amin > hi): out.append(k)
    return out

def normalize_profile(p: dict) -> dict:
    ints = [str(i).title() for i in p.get("interests", [])]
    goals = [str(g).title() for g in p.get("developmental_goals", [])]
    return {"child_name": p.get("child_name"), "interests": ints, "developmental_goals": goals}

def row_text(r: Dict[str,Any]) -> str:
    return (str(r.get("tags","")) + " " + str(r.get("title","")) + " " + str(r.get("description_plain","")) + " " + str(r.get("product_type",""))).lower()

def _is_book_or_activity_like(r: Dict[str, Any]) -> bool:
    t = (str(r.get("title","")) + " " + str(r.get("product_type","")) + " " + str(r.get("tags",""))).lower()
    # For car products, be more specific about what to block
    if "car" in t or "vehicle" in t or "toy car" in t:
        # Only block if it's clearly a book or activity kit, not a toy car kit
        blockers = ["book","board book","picture book","books","activity book","workbook","worksheet"]
        return any(b in t for b in blockers)
    else:
        # For non-car products, use the original blocking logic
        blockers = ["book","board book","picture book","books","activity","activity kit","kit","workbook","worksheet"]
        return any(b in t for b in blockers)

def matches_interest_guarded(r: Dict[str,Any], interest: str, rx: Optional[re.Pattern], text: Optional[str]=None) -> bool:
    if not rx: return False
    if str(interest).strip().lower()=="cars" and _is_book_or_activity_like(r):
        return False
    if text is None: text = row_text(r)
    return rx.search(text) is not None

def _match_interest_by_mask_or_regex(r: Dict[str,Any], idx: Optional[int], interest: str) -> bool:
    """Fast path: use precomputed mask if we have row index; fallback to regex check.
    Keeps the Cars guard intact.
    """
    if str(interest).strip().lower()=="cars" and _is_book_or_activity_like(r):
        return False
    if idx is not None:
        m = INTEREST_MASKS.get(interest)
        if m is not None:
            return bool(m[idx])
    rx = INTEREST_RX.get(interest)
    return matches_interest_guarded(r, interest, rx, None)


# ---------------- Schemas ----------------
class RecommendReq(BaseModel):
    profile: dict; count:int=6; skip:int=0
    price_min: Optional[float]=None; price_max: Optional[float]=None
    exclude_ids: List[Union[int,str]]=Field(default_factory=list)
    diversify: bool=True; mode:str="hybrid"; return_debug: bool=False

class RecommendResp(BaseModel):
    results: List[dict]; meta: Dict[str,Union[int,float,bool,dict]]


# ---------------- Endpoint (Simplified) ----------------
@app.post("/recommend", response_model=RecommendResp)
def recommend(req: RecommendReq):
    """Simplified recommendation endpoint that uses basic text matching"""
    
    prof = normalize_profile(req.profile)
    age_min = req.profile.get("age_min")
    age_max = req.profile.get("age_max")
    interests = prof.get("interests", [])
    
    print(f"🔧 Recommend: age {age_min}-{age_max}, interests {interests}")
    
    # Find products that match criteria
    matching_products = []
    
    for idx, row in DATA.iterrows():
        title = str(row.get('title', '')).lower()
        tags = str(row.get('tags', '')).lower()
        product_type = str(row.get('product_type', '')).lower()
        text = f"{title} {tags} {product_type}"
        
        # Check age match
        age_match = True
        if age_min is not None or age_max is not None:
            age_terms = []
            if age_min == 0 and age_max == 2:
                age_terms = ['0-2', '0-11', '1-2', '1-3', '0-12 month', '0-24 month']
            elif age_min == 3 and age_max == 6:
                age_terms = ['3-6', '3-4', '4-6', '3-5', '4-5']
            else:
                # Generic age matching
                age_terms = [f"{age_min}-{age_max}", f"{age_min}", f"{age_max}"]
            
            age_match = any(term in tags for term in age_terms)
        
        # Check interest match
        interest_match = True
        if interests:
            if "Cars" in interests:
                car_terms = ['car', 'vehicle', 'train', 'truck', 'jeep', 'police car', 'racing car', 'toy car']
                interest_match = any(term in text for term in car_terms)
            else:
                # Generic interest matching
                interest_match = any(interest.lower() in text for interest in interests)
        
        if age_match and interest_match:
            # Exclude books and activity kits for cars
            if "Cars" in interests:
                is_book_or_kit = any(term in text for term in ['book', 'activity', 'kit', 'workbook', 'worksheet'])
                if is_book_or_kit:
                    continue
            
            matching_products.append({
                'id': row.get('id'),
                'title': row.get('title'),
                'tags': row.get('tags', ''),
                'product_type': row.get('product_type', ''),
                'vendor': row.get('vendor', ''),
                'price': row.get('variant_price', 0)
            })
    
    print(f"🔧 Found {len(matching_products)} matching products")
    
    # Return requested number of products
    count = min(req.count, len(matching_products))
    results = matching_products[:count]
    
    return {
        "results": results,
        "meta": {
            "total_hits": len(matching_products),
            "returned": len(results),
            "skip": req.skip,
            "count": req.count,
            "has_more": len(matching_products) > count,
            "debug": {
                "interests": interests,
                "diversification": {interest: len(results) for interest in interests},
                "classification": [],
                "simplified_mode": True
            }
        }
    }


@app.get("/health")
def health():
    return {"status": "ok", "data_rows": len(DATA), "has_embeddings": EMB is not None, "has_faiss": HAS_FAISS}

@app.get("/graphql_test")
def graphql_test():
    """Test endpoint to verify GraphQL functionality"""
    try:
        # Test with a small batch of products
        test_gids = [
            "gid://shopify/Product/1",
            "gid://shopify/Product/2"
        ]
        
        result = fetch_batch_from_shopify(test_gids)
        
        return {
            "status": "success",
            "graphql_configured": True,
            "shop_domain": SHOP_DOMAIN,
            "api_version": API_VERSION,
            "test_results": {
                "requested_gids": len(test_gids),
                "returned_products": len(result),
                "sample_product": result[0] if result else None
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "graphql_configured": False,
            "error": str(e),
            "shop_domain": SHOP_DOMAIN,
            "api_version": API_VERSION
        }

@app.get("/config")
def get_config():
    """Get current configuration"""
    return {
        "shop_domain": SHOP_DOMAIN,
        "api_version": API_VERSION,
        "has_access_token": bool(ACCESS_TOKEN),
        "graphql_url": GRAPHQL_URL,
        "data_rows": len(DATA),
        "has_embeddings": EMB is not None,
        "has_faiss": HAS_FAISS
    }

# Simple test endpoint that bypasses all filtering
@app.post("/test_recommend")
def test_recommend():
    """Simple test endpoint that returns car products without complex filtering"""
    
    # Find car products manually
    car_products = []
    for idx, row in DATA.iterrows():
        title = str(row.get('title', '')).lower()
        tags = str(row.get('tags', '')).lower()
        product_type = str(row.get('product_type', '')).lower()
        text = f"{title} {tags} {product_type}"
        
        # Look for car-related terms
        car_terms = ['car', 'vehicle', 'train', 'truck', 'jeep', 'police car', 'racing car', 'toy car']
        has_car = any(term in text for term in car_terms)
        
        if has_car:
            car_products.append({
                'id': row.get('id'),
                'title': row.get('title'),
                'tags': row.get('tags', ''),
                'product_type': row.get('product_type', ''),
                'vendor': row.get('vendor', ''),
                'price': row.get('variant_price', 0)
            })
    
    # Return first 10 car products
    return {
        "results": car_products[:10],
        "meta": {
            "total_hits": len(car_products),
            "returned": min(10, len(car_products)),
            "skip": 0,
            "count": 10,
            "has_more": len(car_products) > 10,
            "debug": {
                "message": f"Found {len(car_products)} car products total",
                "test_mode": True
            }
        }
    }

# Simplified recommendation endpoint that works
@app.post("/simple_recommend", response_model=RecommendResp)
def simple_recommend(req: RecommendReq):
    """Simplified recommendation endpoint that bypasses complex filtering"""
    
    prof = normalize_profile(req.profile)
    age_min = req.profile.get("age_min")
    age_max = req.profile.get("age_max")
    interests = prof.get("interests", [])
    
    print(f"🔧 Simple recommend: age {age_min}-{age_max}, interests {interests}")
    
    # Find products that match criteria
    matching_products = []
    
    for idx, row in DATA.iterrows():
        title = str(row.get('title', '')).lower()
        tags = str(row.get('tags', '')).lower()
        product_type = str(row.get('product_type', '')).lower()
        text = f"{title} {tags} {product_type}"
        
        # Check age match
        age_match = True
        if age_min is not None or age_max is not None:
            age_terms = []
            if age_min == 0 and age_max == 2:
                age_terms = ['0-2', '0-11', '1-2', '1-3', '0-12 month', '0-24 month']
            elif age_min == 3 and age_max == 6:
                age_terms = ['3-6', '3-4', '4-6', '3-5', '4-5']
            else:
                # Generic age matching
                age_terms = [f"{age_min}-{age_max}", f"{age_min}", f"{age_max}"]
            
            age_match = any(term in tags for term in age_terms)
        
        # Check interest match
        interest_match = True
        if interests:
            if "Cars" in interests:
                car_terms = ['car', 'vehicle', 'train', 'truck', 'jeep', 'police car', 'racing car', 'toy car']
                interest_match = any(term in text for term in car_terms)
            else:
                # Generic interest matching
                interest_match = any(interest.lower() in text for interest in interests)
        
        if age_match and interest_match:
            # Exclude books and activity kits for cars
            if "Cars" in interests:
                is_book_or_kit = any(term in text for term in ['book', 'activity', 'kit', 'workbook', 'worksheet'])
                if is_book_or_kit:
                    continue
            
            matching_products.append({
                'id': row.get('id'),
                'title': row.get('title'),
                'tags': row.get('tags', ''),
                'product_type': row.get('product_type', ''),
                'vendor': row.get('vendor', ''),
                'price': row.get('variant_price', 0)
            })
    
    print(f"🔧 Found {len(matching_products)} matching products")
    
    # Return requested number of products
    count = min(req.count, len(matching_products))
    results = matching_products[:count]
    
    return {
        "results": results,
        "meta": {
            "total_hits": len(matching_products),
            "returned": len(results),
            "skip": req.skip,
            "count": req.count,
            "has_more": len(matching_products) > count,
            "debug": {
                "interests": interests,
                "diversification": {interest: len(results) for interest in interests},
                "classification": [],
                "simple_mode": True
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting server at http://localhost:8002/docs")
    uvicorn.run(app, host="0.0.0.0", port=8006)


