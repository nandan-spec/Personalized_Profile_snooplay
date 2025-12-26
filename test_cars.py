import requests
import json

# Test with the original request parameters
payload = {
    "profile": {
        "child_name": "Anand",
        "age_min": 0,
        "age_max": 2,
        "interests": ["Cars"],
        "developmental_goals": []
    },
    "count": 5,
    "skip": 0,
    "price_min": 0,
    "price_max": 500000,
    "exclude_ids": [],
    "diversify": True,
    "mode": "hybrid",
    "return_debug": True
}

try:
    response = requests.post('http://localhost:8006/recommend', json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        results = data.get('results', [])
        print(f"Results: {len(results)}")
        
        # Print meta information
        meta = data.get('meta', {})
        print(f"Total hits: {meta.get('total_hits', 0)}")
        print(f"Returned: {meta.get('returned', 0)}")
        
        for i, result in enumerate(results[:5]):
            print(f"{i+1}. {result.get('title', 'No title')}")
            print(f"   Status: {result.get('status', 'Unknown')}")
            print(f"   In Stock: {result.get('inStock', 'Unknown')}")
            print(f"   Tags: {result.get('tags', [])[:3]}...")
            print()
        
        # Check debug info
        if 'meta' in data and 'debug' in data['meta']:
            debug = data['meta']['debug']
            if 'diversification' in debug:
                print("Interest diversification:")
                for interest, count in debug['diversification'].items():
                    print(f"  {interest}: {count}")
    else:
        print(f"Error: {response.text}")
        
except Exception as e:
    print(f"Error: {e}")
