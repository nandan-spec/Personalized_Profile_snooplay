#!/usr/bin/env python3
"""
Shopify Product Exporter (Active + In-stock only, with missing variant log)

- Fetches ALL products from Shopify Admin API
- Exports ONLY active products
- Exports ONLY variants with real stock (inventory_quantity > 0)
- Adds retries + sleeps to handle Shopify API limits and network issues
- Saves to:
    shopify_products_export.csv → main dataset (only real in-stock variants)
    variant_missing.csv         → products with no variants
"""

import os
import requests
from bs4 import BeautifulSoup
import csv
import time

# === Replace with your actual details ===
SHOP_DOMAIN = os.getenv("SHOP_DOMAIN", "playhop.myshopify.com")
ACCESS_TOKEN = os.getenv("SHOPIFY_ACCESS_TOKEN")
if not ACCESS_TOKEN:
    raise ValueError("SHOPIFY_ACCESS_TOKEN environment variable is required")
API_VERSION = os.getenv("SHOPIFY_API_VERSION", "2025-07")

# === Request Setup ===
headers = {
    "X-Shopify-Access-Token": ACCESS_TOKEN,
    "Content-Type": "application/json"
}

# ---------------- Safe GET with retry ----------------
def safe_get(url, headers, params=None, max_retries=5, timeout=60):
    """Wrapper for requests.get with retry + exponential backoff"""
    for attempt in range(max_retries):
        try:
            response = requests.get(
                url, headers=headers, params=params,
                timeout=timeout
            )
            response.raise_for_status()
            return response
        except (requests.exceptions.ConnectTimeout,
                requests.exceptions.ReadTimeout):
            wait = 5 * (2 ** attempt)  # exponential backoff
            print(f"⚠️ Timeout (attempt {attempt+1}/{max_retries}) → retrying in {wait}s...")
            time.sleep(wait)
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            break
    return None

# ---------------- Fetch All Products ----------------
def fetch_all_products_simple():
    """Fetch all products using since_id pagination"""
    all_products = []
    since_id = 0

    while True:
        url = f"https://{SHOP_DOMAIN}/admin/api/{API_VERSION}/products.json"
        params = {"limit": 250, "since_id": since_id}

        print(f"Fetching products since ID: {since_id}")
        response = safe_get(url, headers, params, max_retries=5, timeout=60)
        if not response:
            print("❌ Giving up on this page due to repeated errors")
            break

        data = response.json()
        products = data.get("products", [])

        if not products:
            break

        all_products.extend(products)
        since_id = products[-1]['id']  # move to next page
        print(f"✅ Fetched {len(products)} products (Total: {len(all_products)})")

        # --- Sleep between page fetches (Shopify rate-limit safe) ---
        time.sleep(1.0)

    return all_products

# ---------------- Create CSVs ----------------
def create_products_csv(products, main_csv="shopify_products_export.csv", missing_csv="variant_missing.csv"):
    active_count, inactive_count, variant_included, variant_skipped, variant_missing = 0, 0, 0, 0, 0

    with open(main_csv, 'w', newline='', encoding='utf-8') as mainfile, \
         open(missing_csv, 'w', newline='', encoding='utf-8') as missfile:

        # Main export → full variant details
        fieldnames = [
            'id', 'title', 'handle', 'product_type', 'vendor', 'tags',
            'status', 'created_at', 'updated_at',
            'description_html', 'description_plain',
            'seo_title', 'seo_description',
            'variant_id', 'variant_title', 'variant_sku',
            'variant_price', 'variant_compare_price',
            'variant_inventory_quantity', 'variant_inventory_policy',
            'variant_weight', 'variant_requires_shipping',
            'image_url', 'image_alt'
        ]
        writer = csv.DictWriter(mainfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        # Missing log → only product-level details
        miss_fields = ['id', 'title', 'handle', 'status', 'vendor']
        miss_writer = csv.DictWriter(missfile, fieldnames=miss_fields, quoting=csv.QUOTE_ALL)
        miss_writer.writeheader()

        for product in products:
            status = str(product.get('status', '')).lower()

            # Skip inactive products
            if status != "active":
                inactive_count += 1
                continue
            active_count += 1

            desc_html = product.get('body_html', '') or ''
            desc_plain = BeautifulSoup(desc_html, "html.parser").get_text().strip() if desc_html else ''

            variants = product.get('variants', [])
            images = product.get('images', [])

            if variants:
                for variant in variants:
                    qty = variant.get('inventory_quantity') or 0

                    # ✅ NEW: Skip ALL out-of-stock variants (even oversell)
                    if qty <= 0:
                        variant_skipped += 1
                        continue
                    variant_included += 1

                    # Pick variant image
                    variant_image_url, variant_image_alt = "", ""
                    if images:
                        variant_image_url = images[0].get('src', '')
                        variant_image_alt = images[0].get('alt', '')
                        vimg_id = variant.get('image_id')
                        if vimg_id:
                            for img in images:
                                if img.get('id') == vimg_id:
                                    variant_image_url = img.get('src', '')
                                    variant_image_alt = img.get('alt', '')
                                    break

                    row = {
                        'id': product.get('id', ''),
                        'title': product.get('title', ''),
                        'handle': product.get('handle', ''),
                        'product_type': product.get('product_type', ''),
                        'vendor': product.get('vendor', ''),
                        'tags': product.get('tags', ''),
                        'status': product.get('status', ''),
                        'created_at': product.get('created_at', ''),
                        'updated_at': product.get('updated_at', ''),
                        'description_html': desc_html,
                        'description_plain': desc_plain,
                        'seo_title': product.get('seo_title', ''),
                        'seo_description': product.get('seo_description', ''),
                        'variant_id': variant.get('id', ''),
                        'variant_title': variant.get('title', ''),
                        'variant_sku': variant.get('sku', ''),
                        'variant_price': variant.get('price', ''),
                        'variant_compare_price': variant.get('compare_at_price', ''),
                        'variant_inventory_quantity': qty,
                        'variant_inventory_policy': variant.get('inventory_policy', ''),
                        'variant_weight': variant.get('weight', ''),
                        'variant_requires_shipping': variant.get('requires_shipping', ''),
                        'image_url': variant_image_url,
                        'image_alt': variant_image_alt
                    }
                    writer.writerow(row)

            else:
                # Log missing-variant products separately
                variant_missing += 1
                miss_writer.writerow({
                    'id': product.get('id', ''),
                    'title': product.get('title', ''),
                    'handle': product.get('handle', ''),
                    'status': product.get('status', ''),
                    'vendor': product.get('vendor', '')
                })

    # Summary
    print(f"📄 Main CSV created: {main_csv}")
    print(f"📄 Missing variants log: {missing_csv}")
    print("📊 Summary:")
    print(f"   - {len(products)} total products fetched")
    print(f"   - {active_count} active products exported")
    print(f"   - {inactive_count} inactive products skipped")
    print(f"   - {variant_included} variants exported")
    print(f"   - {variant_skipped} out-of-stock variants skipped (including oversell)")
    print(f"   - {variant_missing} products logged as missing variants")

# ---------------- Run Script ----------------
if __name__ == "__main__":
    print("🚀 Fetching ALL products from Shopify...")
    products = fetch_all_products_simple()
    create_products_csv(products)
