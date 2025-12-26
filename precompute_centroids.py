#!/usr/bin/env python3
"""
Pre-compute goal centroids and save them to files for fast startup
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import re
from typing import List, Dict, Optional

def precompute_goal_centroids():
    """Pre-compute goal centroids and save them to files"""
    
    print("Pre-computing goal centroids for fast startup...")
    
    # Load data
    PKL_PATH = os.getenv("PICKLE_PATH", "product_search_index.pkl")
    
    print(f"Loading data from {PKL_PATH}")
    with open(PKL_PATH, "rb") as f:
        payload = pickle.load(f)
    
    DATA = payload["data"]
    EMB = payload["embeddings"].astype("float32")
    print(f"Loaded: {len(DATA)} rows, embeddings shape: {EMB.shape}")
    
    # Load pre-computed masks
    MASKS_PATH = "precomputed_masks.pkl"
    if not os.path.exists(MASKS_PATH):
        print(f"❌ Pre-computed masks not found: {MASKS_PATH}")
        print("Please run precompute_masks.py first")
        return False
    
    print(f"Loading pre-computed masks from {MASKS_PATH}")
    with open(MASKS_PATH, "rb") as f:
        masks_data = pickle.load(f)
    
    GOAL_MASKS = masks_data["GOAL_MASKS"]
    print(f"Loaded {len(GOAL_MASKS)} goal masks")
    
    # Create TEXT field for regex operations
    print("Creating TEXT field for subtype matching...")
    if "description_plain" not in DATA.columns:
        DATA["description_plain"] = ""
    
    DATA["TEXT"] = (
        (DATA.get("tags", "").fillna("").astype(str)) + " " +
        (DATA.get("title", "").fillna("").astype(str)) + " " +
        (DATA.get("description_plain", "").fillna("").astype(str)) + " " +
        (DATA.get("product_type", "").fillna("").astype(str))
    ).str.lower()
    
    # Define subtypes for each goal
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
    
    # Pre-compute goal centroids
    print("Computing goal centroids...")
    GOAL_CENTROIDS = {}
    
    # Set seed for reproducible sampling
    np.random.seed(42)
    
    for g, m in GOAL_MASKS.items():
        try:
            idx = np.nonzero(m)[0]
            if idx.size == 0:
                continue
            
            print(f"  Processing {g}: {len(idx)} products")
            
            # Get subtypes for this goal
            subtypes = goal_subtypes.get(g, {})
            if not subtypes:
                # Fallback to simple averaging for goals without defined subtypes
                cap = min(idx.size, 2000)
                qv = EMB[idx[:cap]].mean(axis=0, keepdims=True).astype("float32")
                qv /= (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-8)
                GOAL_CENTROIDS[g] = qv
                print(f"    Simple centroid: {qv.shape}")
                continue
            
            # Create balanced centroid by sampling equally from each subtype
            subtype_centroids = []
            max_samples_per_subtype = 500  # Limit samples per subtype to prevent bias
            
            for subtype_name, subtype_keywords in subtypes.items():
                print(f"    Processing subtype: {subtype_name}")
                
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
                    print(f"      {subtype_name}: {len(sampled_indices)} samples")
            
            if subtype_centroids:
                # Average all subtype centroids (equal weighting)
                balanced_centroid = np.mean(np.vstack(subtype_centroids), axis=0, keepdims=True).astype("float32")
                balanced_centroid /= (np.linalg.norm(balanced_centroid, axis=1, keepdims=True) + 1e-8)
                GOAL_CENTROIDS[g] = balanced_centroid
                print(f"    Balanced centroid: {balanced_centroid.shape}")
            else:
                # Fallback if no subtypes found
                cap = min(idx.size, 2000)
                qv = EMB[idx[:cap]].mean(axis=0, keepdims=True).astype("float32")
                qv /= (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-8)
                GOAL_CENTROIDS[g] = qv
                print(f"    Fallback centroid: {qv.shape}")
                
        except Exception as e:
            print(f"  Error processing {g}: {e}")
            continue
    
    # Save goal centroids
    print("Saving goal centroids...")
    centroids_data = {
        "GOAL_CENTROIDS": GOAL_CENTROIDS,
        "total_goals": len(GOAL_CENTROIDS)
    }
    
    with open("precomputed_centroids.pkl", "wb") as f:
        pickle.dump(centroids_data, f)
    
    print(f"✅ Saved goal centroids to precomputed_centroids.pkl")
    print(f"✅ Total goals processed: {len(GOAL_CENTROIDS)}")
    
    return True

if __name__ == "__main__":
    success = precompute_goal_centroids()
    if success:
        print("\n🎉 Goal centroids ready!")
        print("📝 Now you can use ultra-fast startup:")
        print("   python app_fast_latest.py")
    else:
        print("\n❌ Failed to pre-compute goal centroids")
