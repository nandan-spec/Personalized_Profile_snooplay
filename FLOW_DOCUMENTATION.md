# Toy Recommendation API - Flow Documentation

## Table of Contents
1. [Application Overview](#application-overview)
2. [Application Startup Flow](#application-startup-flow)
3. [Request Processing Flows](#request-processing-flows)
4. [Data Flow Architecture](#data-flow-architecture)
5. [Component Interactions](#component-interactions)
6. [Decision Points & Fallbacks](#decision-points--fallbacks)

---

## Application Overview

**Application Name:** Toy Recommendation API (Fast Path)  
**Version:** 4.0.0  
**Framework:** FastAPI  
**Port:** 8006  
**Purpose:** Personalized toy recommendation system that matches products based on child profiles (age, interests, developmental goals)

### Key Components
- **Product Catalogue:** Loaded from pickle/CSV files
- **Embeddings:** Pre-computed product embeddings for semantic search
- **FAISS Index:** Approximate nearest neighbor search index
- **Tag Maps:** Age bands, interests, and developmental goals mappings
- **Shopify Integration:** GraphQL API for real-time product data enrichment
- **Regex Patterns:** Runtime-compiled patterns for text matching

---

## Application Startup Flow

### Phase 1: Configuration & Initialization
```
1. Load Environment Variables
   ├─ SHOP_DOMAIN (default: "playhop.myshopify.com")
   ├─ SHOPIFY_ACCESS_TOKEN
   ├─ SHOPIFY_API_VERSION (default: "2025-07")
   ├─ PICKLE_PATH (default: "product_search_index.pkl")
   ├─ CSV_PATH (default: "shopify_products_export.csv")
   ├─ MAPS_PATH (default: "generated_maps_fixed_v2.json")
   ├─ PRECOMPUTED_MASKS_PATH (default: "precomputed_masks.pkl")
   ├─ PRECOMPUTED_CENTROIDS_PATH (default: "precomputed_centroids.pkl")
   └─ FAISS_INDEX_PATH (default: "product_search_index.faiss")

2. Initialize FastAPI Application
   ├─ Create FastAPI instance
   ├─ Add CORS middleware (allow all origins)
   └─ Configure logging
```

### Phase 2: Product Data Loading
```
3. Load Product Catalogue
   ├─ Try: Load from PKL_PATH (pickle file)
   │   ├─ Extract DATA (DataFrame)
   │   ├─ Extract EMB (embeddings array)
   │   └─ Success → Continue
   │
   └─ Fallback: Load from CSV_PATH
       ├─ Read CSV file
       ├─ DATA = DataFrame from CSV
       └─ EMB = None (no embeddings available)

4. Normalize Embeddings (if available)
   ├─ Calculate L2 norms
   └─ Create NORM_EMB (normalized embeddings)
```

### Phase 3: Tag Maps & Regex Patterns
```
5. Load Tag Maps from JSON
   ├─ Read MAPS_PATH (generated_maps_fixed_v2.json)
   ├─ Extract:
   │   ├─ AGE_MAP (age band → keywords)
   │   ├─ INTEREST_TAGS (interest → synonyms)
   │   └─ GOAL_TAGS (goal → keywords)
   └─ Fallback: Empty dictionaries if file missing

6. Compile Regex Patterns
   ├─ AGE_RX: Compile age band patterns
   ├─ INTEREST_RX: Compile interest patterns
   │   └─ Special handling for "Cars" (filter transport terms)
   └─ GOAL_RX: Compile goal patterns
```

### Phase 4: Text Field Preparation
```
7. Create TEXT Field
   ├─ Concatenate: tags + title + description_plain + product_type
   ├─ Convert to lowercase
   └─ Store in DATA["TEXT"]
```

### Phase 5: Masks & Indices Computation
```
8. Load/Compute Masks
   ├─ Try: Load precomputed masks from PRECOMPUTED_MASKS_PATH
   │   ├─ Load TEXT_ARRAY, KEY_TO_IDX
   │   ├─ Load AGE_MASKS, INTEREST_MASKS, GOAL_MASKS
   │   └─ Note: Currently DISABLED (using regex instead)
   │
   └─ Fallback: Compute masks at startup
       ├─ Create TEXT_ARRAY from DATA["TEXT"]
       ├─ Build KEY_TO_IDX mapping (id, handle) → row index
       ├─ Compute INTEREST_MASKS (regex match per interest)
       ├─ Compute INTEREST_INDICES (nonzero indices)
       ├─ Compute AGE_MASKS (regex match per age band)
       └─ Compute GOAL_MASKS (regex match per goal)

9. Compute Interest Centroids
   ├─ For each interest with matching products:
   │   ├─ Get indices of matching products (max 2000)
   │   ├─ Compute mean embedding
   │   ├─ Normalize to unit vector
   │   └─ Store in INTEREST_CENTROIDS
   └─ Compute GLOBAL_CENTROID (mean of all embeddings)

10. Load/Compute Goal Centroids
    ├─ Try: Load from PRECOMPUTED_CENTROIDS_PATH
    │   └─ Load GOAL_CENTROIDS
    │
    └─ Fallback: Compute balanced centroids
        ├─ For each goal:
        │   ├─ Identify subtypes (e.g., Motor Skills → fine_motor, gross_motor)
        │   ├─ Sample products per subtype (max 500 each)
        │   ├─ Compute subtype centroids
        │   ├─ Average subtype centroids (equal weighting)
        │   └─ Store in GOAL_CENTROIDS
        └─ Use simple averaging for goals without subtypes
```

### Phase 6: FAISS Index Setup
```
11. Initialize FAISS Index
    ├─ Try: Import faiss library
    │   ├─ Try: Load pre-built index from FAISS_INDEX_PATH
    │   │   ├─ Read index file
    │   │   ├─ Set DIM (embedding dimension)
    │   │   ├─ Load metadata (nprobe setting)
    │   │   └─ HAS_FAISS = True
    │   │
    │   ├─ Fallback: Build index from scratch
    │   │   ├─ Normalize embeddings
    │   │   ├─ Calculate nlist (clustering factor)
    │   │   ├─ Create IndexIVFFlat
    │   │   ├─ Train index
    │   │   ├─ Add vectors
    │   │   └─ Set nprobe
    │   │
    │   └─ Special: Create dummy embeddings if index exists but EMB missing
    │
    └─ Fallback: HAS_FAISS = False (use flat search)
```

### Phase 7: Server Ready
```
12. Application Ready
    └─ FastAPI server starts on port 8006
    └─ All endpoints available
```

---

## Request Processing Flows

### Flow 1: `/recommend` Endpoint

```
Request: POST /recommend
Body: {
  profile: { age_min, age_max, interests, developmental_goals, child_name },
  count: 6,
  skip: 0,
  price_min: optional,
  price_max: optional,
  exclude_ids: [],
  diversify: true,
  mode: "hybrid",
  return_debug: false
}

Processing Steps:
├─ 1. Normalize Profile
│   ├─ Extract interests → title case
│   ├─ Extract developmental_goals → title case
│   └─ Preserve child_name, age_min, age_max
│
├─ 2. Iterate Through Product Catalogue
│   ├─ For each row in DATA:
│   │   ├─ Extract: title, tags, product_type
│   │   ├─ Build search text: title + tags + product_type (lowercase)
│   │   │
│   │   ├─ 3. Age Matching
│   │   │   ├─ If age_min/age_max provided:
│   │   │   │   ├─ Map to age terms (e.g., 0-2 → ['0-2', '0-11', '1-2'])
│   │   │   │   └─ Check if any age term in tags
│   │   │   └─ If no age provided: age_match = True
│   │   │
│   │   ├─ 4. Interest Matching
│   │   │   ├─ If interests provided:
│   │   │   │   ├─ Special case: "Cars"
│   │   │   │   │   ├─ Check for car terms in text
│   │   │   │   │   └─ Exclude if book/activity-like
│   │   │   │   └─ Generic: Check if interest.lower() in text
│   │   │   └─ If no interests: interest_match = True
│   │   │
│   │   └─ 5. Apply Filters
│   │       ├─ If age_match AND interest_match:
│   │       │   ├─ Special: For "Cars" interest
│   │       │   │   └─ Exclude if book/kit/worksheet detected
│   │       │   └─ Add to matching_products list
│   │       └─ Continue to next product
│   │
│   └─ 6. Build Results
│       ├─ Slice matching_products[skip:skip+count]
│       ├─ Format product data:
│       │   ├─ id, title, tags, product_type
│       │   ├─ vendor, price
│       │   └─ Basic fields only (no Shopify enrichment)
│       │
│       └─ 7. Return Response
│           ├─ results: List of matching products
│           └─ meta:
│               ├─ total_hits: Total matches found
│               ├─ returned: Number returned
│               ├─ skip, count, has_more
│               └─ debug: interests, diversification stats
```

### Flow 2: `/simple_recommend` Endpoint

```
Request: POST /simple_recommend
Body: Same as /recommend

Processing Steps:
└─ Identical to /recommend endpoint
   └─ Note: Currently duplicates /recommend logic
```

### Flow 3: `/test_recommend` Endpoint

```
Request: POST /test_recommend
Body: None

Processing Steps:
├─ 1. Iterate Through All Products
│   └─ For each row in DATA:
│       ├─ Build search text (title + tags + product_type)
│       └─ Check for car-related terms
│
├─ 2. Filter Car Products
│   └─ Match terms: ['car', 'vehicle', 'train', 'truck', 'jeep', ...]
│
└─ 3. Return First 10 Car Products
    └─ No additional filtering applied
```

### Flow 4: `/health` Endpoint

```
Request: GET /health

Processing Steps:
└─ Return Status
   ├─ status: "ok"
   ├─ data_rows: len(DATA)
   ├─ has_embeddings: EMB is not None
   └─ has_faiss: HAS_FAISS
```

### Flow 5: `/graphql_test` Endpoint

```
Request: GET /graphql_test

Processing Steps:
├─ 1. Prepare Test GIDs
│   └─ ["gid://shopify/Product/1", "gid://shopify/Product/2"]
│
├─ 2. Call fetch_batch_from_shopify()
│   ├─ Build GraphQL query
│   ├─ Set headers (access token)
│   ├─ POST to GRAPHQL_URL
│   └─ Parse response
│
└─ 3. Return Test Results
    ├─ status: "success" or "error"
    ├─ graphql_configured: True/False
    ├─ shop_domain, api_version
    └─ test_results: requested vs returned counts
```

### Flow 6: `/config` Endpoint

```
Request: GET /config

Processing Steps:
└─ Return Configuration
   ├─ shop_domain, api_version
   ├─ has_access_token: bool(ACCESS_TOKEN)
   ├─ graphql_url
   ├─ data_rows: len(DATA)
   ├─ has_embeddings: EMB is not None
   └─ has_faiss: HAS_FAISS
```

---

## Data Flow Architecture

### Data Sources
```
┌─────────────────┐
│  Pickle File    │──┐
│  (PKL_PATH)     │  │
└─────────────────┘  │
                     ├─→ DATA (DataFrame)
┌─────────────────┐  │   EMB (numpy array)
│  CSV File       │──┘
│  (CSV_PATH)     │
└─────────────────┘

┌─────────────────┐
│  Maps JSON      │──→ AGE_MAP, INTEREST_TAGS, GOAL_TAGS
│  (MAPS_PATH)    │
└─────────────────┘

┌─────────────────┐
│  Precomputed    │──→ Masks & Centroids (optional)
│  Pickle Files   │
└─────────────────┘

┌─────────────────┐
│  FAISS Index    │──→ INDEX (if available)
│  (.faiss file)  │
└─────────────────┘
```

### Data Transformation Pipeline
```
Raw Data (CSV/Pickle)
    │
    ├─→ DataFrame (DATA)
    │   ├─ Add TEXT field (concatenated text)
    │   ├─ Create KEY_TO_IDX mapping
    │   └─ Extract embeddings (EMB)
    │
    ├─→ Embeddings (EMB)
    │   ├─ Normalize → NORM_EMB
    │   ├─ Compute centroids (INTEREST_CENTROIDS, GOAL_CENTROIDS)
    │   └─ Build FAISS index
    │
    └─→ Tag Maps (JSON)
        ├─ Compile regex patterns (AGE_RX, INTEREST_RX, GOAL_RX)
        └─ Compute boolean masks (AGE_MASKS, INTEREST_MASKS, GOAL_MASKS)
```

### Request-to-Response Flow
```
Client Request
    │
    ├─→ FastAPI Router
    │   ├─ Validate request (Pydantic schema)
    │   └─ Route to endpoint handler
    │
    ├─→ Profile Normalization
    │   └─ Standardize interests/goals (title case)
    │
    ├─→ Product Matching
    │   ├─ Iterate DATA DataFrame
    │   ├─ Apply age filters (regex/terms)
    │   ├─ Apply interest filters (regex/terms)
    │   └─ Apply special rules (Cars → exclude books)
    │
    ├─→ Result Formatting
    │   └─ Build product dictionaries
    │
    └─→ Response
        ├─ results: List[Product]
        └─ meta: Metadata dict
```

---

## Component Interactions

### Component Dependency Graph
```
FastAPI Application
    │
    ├─→ Product Catalogue (DATA)
    │   ├─ Used by: All endpoints
    │   └─ Source: Pickle/CSV
    │
    ├─→ Embeddings (EMB, NORM_EMB)
    │   ├─ Used by: Centroids computation, FAISS
    │   └─ Source: Pickle file
    │
    ├─→ Tag Maps (AGE_MAP, INTEREST_TAGS, GOAL_TAGS)
    │   ├─ Used by: Regex compilation, mask computation
    │   └─ Source: JSON file
    │
    ├─→ Regex Patterns (AGE_RX, INTEREST_RX, GOAL_RX)
    │   ├─ Used by: Mask computation, product matching
    │   └─ Generated from: Tag maps
    │
    ├─→ Masks (AGE_MASKS, INTEREST_MASKS, GOAL_MASKS)
    │   ├─ Used by: Fast product filtering (currently disabled)
    │   └─ Generated from: Regex patterns + DATA
    │
    ├─→ Centroids (INTEREST_CENTROIDS, GOAL_CENTROIDS)
    │   ├─ Used by: Semantic search (if implemented)
    │   └─ Generated from: EMB + masks
    │
    ├─→ FAISS Index (INDEX)
    │   ├─ Used by: Vector similarity search (if implemented)
    │   └─ Source: Pre-built file or computed from EMB
    │
    └─→ Shopify GraphQL Client
        ├─ Used by: fetch_batch_from_shopify()
        └─ Configuration: SHOP_DOMAIN, ACCESS_TOKEN, API_VERSION
```

### Function Call Hierarchy
```
recommend() / simple_recommend()
    │
    ├─→ normalize_profile()
    │   └─ Standardizes profile data
    │
    ├─→ Iterate DATA.iterrows()
    │   ├─→ row_text() [implicit via text building]
    │   ├─→ _is_book_or_activity_like() [for Cars interest]
    │   └─→ matches_interest_guarded() [if regex matching used]
    │
    └─→ Return formatted results

fetch_and_format_products()
    │
    ├─→ fetch_batch_from_shopify()
    │   └─→ requests.post() [GraphQL API call]
    │
    └─→ format_shopify_products()
        └─→ Process variants, images, pricing, inventory
```

---

## Decision Points & Fallbacks

### Startup Decision Tree

#### 1. Product Data Loading
```
IF pickle file exists AND valid:
    Load DATA + EMB from pickle
ELSE IF CSV file exists:
    Load DATA from CSV, EMB = None
ELSE:
    Create minimal test DataFrame
```

#### 2. Embeddings Normalization
```
IF EMB is not None:
    Compute NORM_EMB (L2 normalized)
ELSE:
    NORM_EMB = None
```

#### 3. Tag Maps Loading
```
IF MAPS_PATH file exists:
    Load AGE_MAP, INTEREST_TAGS, GOAL_TAGS
ELSE:
    Use empty dictionaries
```

#### 4. Masks Computation
```
IF PRECOMPUTED_MASKS_PATH exists:
    Try to load precomputed masks
    └─ Note: Currently disabled, recomputes anyway
ELSE:
    Compute masks at startup using regex
```

#### 5. Centroids Computation
```
IF PRECOMPUTED_CENTROIDS_PATH exists:
    Load GOAL_CENTROIDS
ELSE:
    Compute balanced centroids:
        ├─ For goals with subtypes: Sample per subtype, average
        └─ For goals without subtypes: Simple mean
```

#### 6. FAISS Index
```
IF faiss library available:
    IF FAISS_INDEX_PATH exists:
        Load pre-built index
        └─ IF EMB missing: Create dummy embeddings
    ELSE IF EMB available AND not SKIP_FAISS_TRAINING:
        Build index from scratch
    ELSE:
        HAS_FAISS = False
ELSE:
    HAS_FAISS = False (fallback to flat search)
```

### Runtime Decision Points

#### 1. Age Matching
```
IF age_min AND age_max provided:
    Map to age terms based on range
    Check if terms in product tags
ELSE:
    age_match = True (no filtering)
```

#### 2. Interest Matching
```
IF interests provided:
    IF "Cars" in interests:
        Check for car terms in text
        └─ Exclude if book/activity-like
    ELSE:
        Check if interest.lower() in text
ELSE:
    interest_match = True (no filtering)
```

#### 3. Product Filtering
```
IF age_match AND interest_match:
    IF "Cars" interest:
        IF is_book_or_activity_like:
            EXCLUDE product
        ELSE:
            INCLUDE product
    ELSE:
        INCLUDE product
ELSE:
    EXCLUDE product
```

### Error Handling & Fallbacks

#### GraphQL API Calls
```
TRY:
    POST to Shopify GraphQL API
    Parse response
    Return products
EXCEPT:
    Log error
    Return empty list
```

#### Data Access
```
TRY:
    Access DATA row by index
    Extract field values
EXCEPT:
    Use default values (empty strings, 0, None)
    Continue processing
```

#### Regex Matching
```
TRY:
    Compile regex pattern
    Match against text
EXCEPT:
    Pattern = None
    Match = False
```

---

## Key Design Patterns

### 1. **Lazy Loading with Fallbacks**
- Try to load optimized/precomputed data first
- Fall back to computation if files missing
- Graceful degradation (e.g., no embeddings → text-only matching)

### 2. **Regex-Based Filtering**
- Runtime-compiled patterns for flexibility
- Word-boundary matching for accuracy
- Special handling for edge cases (Cars interest)

### 3. **Batch Processing**
- Shopify GraphQL calls batched (250 products max)
- Efficient iteration over DataFrame

### 4. **Normalization at Boundaries**
- Profile normalization (title case)
- Text normalization (lowercase)
- Embedding normalization (L2 norm)

### 5. **Configuration via Environment**
- All paths/configurable values via env vars
- Sensible defaults for development

---

## Performance Considerations

### Startup Time
- **Fast Path:** Precomputed masks/centroids/FAISS index loaded
- **Slow Path:** Computation at startup (masks, centroids, FAISS training)

### Runtime Performance
- **Current Implementation:** Linear scan through DataFrame (O(n))
- **Potential Optimization:** Use precomputed masks for O(1) lookups
- **FAISS:** Available but not currently used in recommendation flow

### Memory Usage
- **DATA:** DataFrame in memory (all products)
- **EMB:** Full embedding matrix (can be large)
- **Masks:** Boolean arrays (one per tag/age/goal)
- **FAISS Index:** Additional memory for index structure

---

## Future Enhancements

1. **Use FAISS for Semantic Search:** Currently available but not integrated into recommendation flow
2. **Shopify Enrichment:** Fetch real-time product data for recommendations
3. **Deduplicate Endpoints:** Merge `/recommend` and `/simple_recommend`
4. **Enable Precomputed Masks:** Currently disabled, could improve performance
5. **Add Caching:** Cache frequent queries/results
6. **Add Logging:** More detailed request/response logging

---

## Appendix: File Dependencies

### Required Files (Startup)
- `product_search_index.pkl` OR `shopify_products_export.csv` (product data)
- `generated_maps_fixed_v2.json` (tag maps)

### Optional Files (Performance)
- `precomputed_masks.pkl` (precomputed boolean masks)
- `precomputed_centroids.pkl` (precomputed embedding centroids)
- `product_search_index.faiss` (pre-built FAISS index)
- `product_search_index.meta.json` (FAISS metadata)

### Generated Files (Runtime)
- None (application is read-only)

---

**Document Version:** 1.0  
**Last Updated:** Based on app_fast_latest.py v4.0.0  
**Maintained By:** Development Team

