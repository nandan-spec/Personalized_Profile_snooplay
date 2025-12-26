# Personalized Profile - Snooplay Toy Recommendation API

## Version Information

- **API Version:** 4.0.0
- **Main File:** `app_fast_latest.py`
- **Framework:** FastAPI
- **Python Version:** 3.7+

## Main File

**`app_fast_latest.py`** is the main application file that contains:
- FastAPI application server
- Toy recommendation engine
- Shopify GraphQL integration
- Semantic search using FAISS
- Product filtering and ranking logic

## Setup Instructions

**⚠️ IMPORTANT: Before starting the application, you must generate the required data files.**

### Step 1: Generate CSV File

First, run `generatecsv.py` to fetch products from Shopify and generate the CSV file:

```bash
# Set required environment variables
export SHOPIFY_ACCESS_TOKEN="your_token_here"
export SHOP_DOMAIN="playhop.myshopify.com"  # Optional, defaults to playhop.myshopify.com
export SHOPIFY_API_VERSION="2025-07"  # Optional, defaults to 2025-07

# Run the CSV generator
python generatecsv.py
```

This will generate:
- `shopify_products_export.csv` - Main product dataset (active products with in-stock variants only)
- `variant_missing.csv` - Products with no variants (for reference)

### Step 2: Generate Embeddings and FAISS Index

After generating the CSV file, run `buildfaiss.py` to create the embeddings and FAISS index:

```bash
# Run the FAISS builder
python buildfaiss.py
```

This will generate:
- `product_search_index.pkl` - Contains product data and embeddings
- `product_search_index.faiss` - FAISS index for fast similarity search
- `product_search_index.meta.json` - FAISS index metadata

**Note:** The `buildfaiss.py` script requires the `sentence-transformers` library. Install it if needed:
```bash
pip install sentence-transformers
```

### Step 3: Run the Application

Once both files are generated, you can start the application:

```bash
# Set required environment variables (if not already set)
export SHOPIFY_ACCESS_TOKEN="your_token_here"
export SHOP_DOMAIN="playhop.myshopify.com"  # Optional
export SHOPIFY_API_VERSION="2025-07"  # Optional

# Run the application
python app_fast_latest.py
```

### Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Set required environment variables
export SHOPIFY_ACCESS_TOKEN="your_token_here"
export SHOP_DOMAIN="playhop.myshopify.com"  # Optional, defaults to playhop.myshopify.com
export SHOPIFY_API_VERSION="2025-07"  # Optional, defaults to 2025-07

# Run the application
python app_fast_latest.py
```

The server will start on `http://localhost:8006`

API documentation will be available at: `http://localhost:8006/docs`

## Important Notes

### 🔐 Security

1. **Never commit secrets or API keys to the repository**
   - The `SHOPIFY_ACCESS_TOKEN` must be set as an environment variable
   - The code will raise an error if the token is not provided
   - Use environment variables or secure secret management systems in production

2. **Environment Variables Required:**
   - `SHOPIFY_ACCESS_TOKEN` (Required) - Shopify Admin API access token
   - `SHOP_DOMAIN` (Optional) - Defaults to "playhop.myshopify.com"
   - `SHOPIFY_API_VERSION` (Optional) - Defaults to "2025-07"

### 📁 Data Files

The application expects the following data files (configurable via environment variables):

- `PICKLE_PATH` (default: `product_search_index.pkl`) - Pre-computed embeddings and product data
- `CSV_PATH` (default: `shopify_products_export.csv`) - Fallback CSV data file
- `MAPS_PATH` (default: `generated_maps_fixed_v2.json`) - Tag mapping configuration
- `FAISS_INDEX_PATH` (default: `product_search_index.faiss`) - Pre-built FAISS index (optional)
- `PRECOMPUTED_MASKS_PATH` (default: `precomputed_masks.pkl`) - Pre-computed filtering masks (optional)
- `PRECOMPUTED_CENTROIDS_PATH` (default: `precomputed_centroids.pkl`) - Pre-computed centroids (optional)

### 🚀 Features

- **Semantic Search:** Uses FAISS for fast vector similarity search
- **Shopify Integration:** Fetches real-time product data via GraphQL API
- **Personalized Recommendations:** Based on child's age, interests, and developmental goals
- **Product Filtering:** Filters by price range, availability, and product status
- **Diversification:** Ensures diverse recommendations across interests

### 📡 API Endpoints

- `POST /recommend` - Get personalized toy recommendations
- `POST /simple_recommend` - Simplified recommendation endpoint
- `GET /health` - Health check endpoint
- `GET /config` - Get current configuration
- `GET /graphql_test` - Test Shopify GraphQL connectivity
- `POST /test_recommend` - Test endpoint for car products

### 🛠️ Utility Scripts

The repository includes several utility and test scripts:

- **`generatecsv.py`** - **REQUIRED FIRST STEP** - Fetches products from Shopify and generates `shopify_products_export.csv`
- **`buildfaiss.py`** - **REQUIRED SECOND STEP** - Generates embeddings and FAISS index from CSV file
- `precompute_centroids.py` - Pre-compute goal centroids
- `precompute_masks.py` - Pre-compute filtering masks
- `clean_cars_tags.py` - Clean car-related tags
- `fix_cars_tags.py` - Fix car tags in data
- Various test and debug scripts (prefixed with `test_` or `debug_`)

### 📦 Dependencies

See `requirements.txt` for the complete list of dependencies. Key dependencies include:

- FastAPI - Web framework
- Uvicorn - ASGI server
- NumPy & Pandas - Data processing
- FAISS - Vector similarity search
- Requests - HTTP client for Shopify API

### 🔧 Configuration

All configuration can be done via environment variables. The application supports:

- Custom data file paths
- Shopify API configuration
- FAISS index settings
- Pre-computed data paths

### 📝 Documentation

- `FLOW_DOCUMENTATION.md` - Application flow and logic documentation
- `TECH_STACK.md` - Technical stack information

### ⚠️ Notes

- **Setup Order is Critical:** Always run `generatecsv.py` first, then `buildfaiss.py`, before starting the application
- The application uses port **8006** by default
- Ensure all required data files are present before starting the server
- The application will fall back to CSV data if PKL file is not available
- FAISS index is optional but recommended for better performance
- The `buildfaiss.py` script requires `sentence-transformers` library (install with `pip install sentence-transformers`)

## Development

For development and testing, refer to the test scripts in the repository. The application includes comprehensive logging for debugging.

## License

[Add your license information here]

