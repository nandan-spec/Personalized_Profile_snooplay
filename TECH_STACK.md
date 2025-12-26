# Tech Stack Documentation

## Overview
This document outlines the complete technology stack used in the Toy Recommendation API application.

---

## Core Technology Stack

### **Programming Language**
- **Python 3.x**
  - Primary language for the entire application
  - Used for API development, data processing, and ML operations

### **Web Framework**
- **FastAPI** (v4.0.0)
  - Modern, high-performance web framework for building APIs
  - Automatic API documentation (OpenAPI/Swagger)
  - Built-in data validation with Pydantic
  - Async support (though current implementation is synchronous)

### **ASGI Server**
- **Uvicorn**
  - ASGI server implementation
  - Used to run the FastAPI application
  - Default port: 8006
  - Supports hot-reload in development

---

## Data Processing & Scientific Computing

### **Numerical Computing**
- **NumPy**
  - Multi-dimensional array operations
  - Used for:
    - Embedding vector operations
    - Normalization (L2 norm calculations)
    - Centroid computations
    - Boolean mask operations
    - Mathematical operations on embeddings

### **Data Manipulation**
- **Pandas**
  - DataFrame operations for product catalogue
  - Data loading from CSV files
  - Column operations and filtering
  - Text processing and concatenation
  - Series operations for boolean masks

### **Vector Similarity Search**
- **FAISS (Facebook AI Similarity Search)**
  - High-performance similarity search library
  - Used for approximate nearest neighbor search
  - Index types: IndexIVFFlat (Inverted File Index)
  - Supports cosine similarity via inner product on normalized vectors
  - Pre-built index loading and runtime index construction

---

## HTTP & API Integration

### **HTTP Client**
- **Requests**
  - HTTP library for making API calls
  - Used for Shopify GraphQL API integration
  - POST requests with JSON payloads
  - Error handling and response parsing

### **External API**
- **Shopify GraphQL API**
  - Version: 2025-07 (configurable via environment)
  - GraphQL endpoint for product data enrichment
  - Real-time product information retrieval
  - Batch processing (up to 250 products per request)

---

## Data Validation & Serialization

### **Data Validation**
- **Pydantic**
  - Request/response model validation
  - Type checking and data coercion
  - Schema definitions (BaseModel, Field)
  - Automatic validation of API inputs

### **Serialization Formats**
- **Pickle**
  - Python object serialization
  - Used for:
    - Storing product data + embeddings
    - Precomputed masks storage
    - Precomputed centroids storage
    - FAISS index metadata

- **JSON**
  - Configuration and tag maps storage
  - API request/response format
  - FAISS metadata storage

- **CSV**
  - Product catalogue fallback format
  - Import from Shopify exports

---

## Standard Library Modules

### **Core Utilities**
- `os` - Environment variables, file paths
- `re` - Regular expression pattern matching
- `pickle` - Object serialization
- `json` - JSON parsing and generation
- `math` - Mathematical operations (sqrt for FAISS nlist calculation)
- `logging` - Application logging
- `typing` - Type hints (List, Dict, Any, Optional, Union)

---

## Data Storage & File Formats

### **File-Based Storage**
- **Pickle Files (.pkl)**
  - `product_search_index.pkl` - Product data + embeddings
  - `precomputed_masks.pkl` - Boolean masks for filtering
  - `precomputed_centroids.pkl` - Embedding centroids

- **FAISS Index Files**
  - `.faiss` - Binary FAISS index format
  - `.meta.json` - Index metadata (nprobe settings)

- **JSON Files**
  - `generated_maps_fixed_v2.json` - Tag mappings (age, interests, goals)

- **CSV Files**
  - `shopify_products_export.csv` - Product catalogue fallback

### **Azure Storage (Optional)**
- **Azurite** (Azure Storage Emulator)
  - Based on `__azurite_db_table__.json` and `AzuriteConfig` files
  - Likely used for cloud storage emulation in development

---

## Development & Testing Tools

### **Testing Scripts** (Based on file structure)
- Custom test scripts for:
  - Regex pattern testing (`test_regex.py`, `test_car_regex.py`)
  - Car product filtering (`test_cars.py`, `test_cars_debug.py`)
  - Semantic search (`test_semantic_search.py`)
  - Mask validation (`test_masks.py`, `test_updated_masks.py`)

### **Debugging Tools**
- Custom debug scripts:
  - `debug_cars.py` - Car product debugging
  - `debug_filtering.py` - Filter logic debugging
  - `debug_regex.py` - Regex pattern debugging
  - `debug_activity_bundle.py` - Activity bundle debugging
  - `diagnose_filtering.py` - Filtering diagnostics

### **Data Processing Scripts**
- `buildfaiss.py` - FAISS index construction
- `precompute_centroids.py` - Centroid precomputation
- `precompute_masks.py` - Mask precomputation
- `clean_cars_tags.py` - Tag cleaning utilities
- `fix_cars_tags.py` - Tag fixing utilities
- `find_car_products.py` - Product discovery

---

## Architecture Patterns

### **API Architecture**
- **RESTful API** (via FastAPI)
  - POST endpoints for recommendations
  - GET endpoints for health checks and configuration
  - JSON request/response format

### **Data Processing Architecture**
- **Batch Processing** - Product catalogue loaded at startup
- **In-Memory Processing** - All data structures in memory
- **Lazy Loading** - Optional components loaded on demand

### **Search Architecture**
- **Hybrid Search Approach**
  - Text-based filtering (regex patterns)
  - Vector similarity search (FAISS - available but not fully integrated)
  - Rule-based filtering (age, interests, goals)

---

## Environment & Configuration

### **Configuration Management**
- Environment variables for configuration:
  - `SHOP_DOMAIN` - Shopify store domain
  - `SHOPIFY_ACCESS_TOKEN` - API authentication
  - `SHOPIFY_API_VERSION` - API version (default: 2025-07)
  - `PICKLE_PATH` - Product data file path
  - `CSV_PATH` - CSV fallback path
  - `MAPS_PATH` - Tag maps JSON path
  - `PRECOMPUTED_MASKS_PATH` - Masks file path
  - `PRECOMPUTED_CENTROIDS_PATH` - Centroids file path
  - `FAISS_INDEX_PATH` - FAISS index file path
  - `SKIP_FAISS_TRAINING` - Skip FAISS training flag
  - `FAISS_NPROBE` - FAISS search parameter

### **Deployment**
- **Standalone Application**
  - Runs as a single Python process
  - Uvicorn ASGI server
  - No database dependencies (file-based storage)

---

## Dependencies Summary

### **Core Dependencies** (Required)
```python
fastapi>=0.68.0          # Web framework
uvicorn[standard]       # ASGI server
pydantic>=1.8.0         # Data validation
numpy>=1.21.0           # Numerical computing
pandas>=1.3.0           # Data manipulation
requests>=2.26.0        # HTTP client
```

### **Optional Dependencies**
```python
faiss-cpu>=1.7.0        # Vector similarity search (CPU version)
# OR
faiss-gpu>=1.7.0        # Vector similarity search (GPU version)
```

### **Python Version**
- **Python 3.7+** (recommended: Python 3.8+)
  - Type hints support required
  - f-string support
  - Modern standard library features

---

## Technology Stack Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Client Applications                   │
│              (Web, Mobile, API Consumers)                │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/REST
                     ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Application                   │
│  ┌──────────────────────────────────────────────────┐   │
│  │              Uvicorn ASGI Server                 │   │
│  └──────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────┐   │
│  │         Pydantic Request/Response Models         │   │
│  └──────────────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌──────────────┐        ┌──────────────────┐
│  In-Memory   │        │  Shopify GraphQL  │
│   Data       │        │       API         │
│              │        │                   │
│ • Pandas     │        │  (External API)   │
│ • NumPy      │        │                   │
│ • FAISS      │        └───────────────────┘
│ • Masks      │
│ • Centroids  │
└──────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│         File-Based Storage              │
│                                         │
│ • Pickle files (.pkl)                  │
│ • FAISS index (.faiss)                 │
│ • JSON configs (.json)                  │
│ • CSV exports (.csv)                    │
└─────────────────────────────────────────┘
```

---

## Performance Characteristics

### **Computational Libraries**
- **NumPy**: Optimized C-based array operations
- **Pandas**: Efficient DataFrame operations
- **FAISS**: Highly optimized similarity search (C++ backend)

### **API Performance**
- **FastAPI**: High performance, comparable to Node.js and Go
- **Uvicorn**: Async-capable server (though current code is sync)
- **In-Memory Processing**: Fast lookups, but memory-intensive

### **Scalability Considerations**
- **Current**: Single-process, in-memory architecture
- **Limitations**: Memory-bound by product catalogue size
- **Potential**: Can be scaled horizontally with load balancer

---

## Development Workflow

### **Local Development**
1. Python virtual environment
2. Install dependencies
3. Load/prepare data files
4. Run with Uvicorn: `python app_fast_latest.py`
5. Access API docs at: `http://localhost:8006/docs`

### **Data Preparation**
1. Export products from Shopify (CSV)
2. Generate embeddings (external process)
3. Build FAISS index (`buildfaiss.py`)
4. Precompute masks (`precompute_masks.py`)
5. Precompute centroids (`precompute_centroids.py`)
6. Generate tag maps (JSON)

---

## Security Considerations

### **API Security**
- **CORS**: Currently open (`allow_origins=["*"]`)
- **Authentication**: None implemented (Shopify token in code - should be env var)
- **Input Validation**: Pydantic models provide validation

### **Data Security**
- **Sensitive Data**: Shopify access token should be in environment variables
- **File Access**: Local file system access required

---

## Version Information

- **Application Version**: 4.0.0
- **Shopify API Version**: 2025-07 (configurable)
- **Python**: 3.7+ (type hints suggest 3.7+)

---

## Summary

This application uses a **modern Python data science stack** optimized for:
- **Fast API development** (FastAPI)
- **Efficient data processing** (Pandas, NumPy)
- **Vector similarity search** (FAISS)
- **Real-time API integration** (Shopify GraphQL)
- **File-based persistence** (Pickle, JSON, CSV)

The stack is well-suited for:
- ✅ Recommendation systems
- ✅ Product search and filtering
- ✅ Semantic similarity matching
- ✅ Real-time product data enrichment
- ✅ High-performance API services

---

**Document Version:** 1.0  
**Last Updated:** Based on app_fast_latest.py analysis  
**Maintained By:** Development Team

