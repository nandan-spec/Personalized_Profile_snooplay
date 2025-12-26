#!/usr/bin/env python3
"""
Pre-compute regex masks and save them to files for fast startup
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import re
from typing import List, Dict, Optional

def _kws_to_compiled(kws: List[str]) -> Optional[re.Pattern]:
    """Convert keywords to compiled regex pattern"""
    kws = [k for k in (kws or []) if k]
    if not kws:
        return None
    alts = []
    for k in kws:
        k = re.escape(k.strip().lower()).replace(r"\ ", r"[ -]?")
        alts.append(f"{k}s?")
    return re.compile(r"(?:^|[^a-z0-9])(?:" + "|".join(alts) + r")(?:[^a-z0-9]|$)")

def precompute_masks():
    """Pre-compute all regex masks and save them to files"""
    
    print("Pre-computing regex masks for fast startup...")
    
    # Load data
    PKL_PATH = os.getenv("PICKLE_PATH", "product_search_index.pkl")
    MAPS_PATH = os.getenv("MAPS_PATH", "generated_maps_fixed_v2.json")
    
    print(f"Loading data from {PKL_PATH}")
    with open(PKL_PATH, "rb") as f:
        payload = pickle.load(f)
    
    DATA = payload["data"]
    print(f"Loaded: {len(DATA)} rows")
    
    # Load tag maps
    print(f"Loading tag maps from {MAPS_PATH}")
    with open(MAPS_PATH, "r") as f:
        maps = json.load(f)
    
    AGE_MAP = maps.get("AGE_MAP", {})
    INTEREST_TAGS = maps.get("INTEREST_TAGS", {})
    GOAL_TAGS = maps.get("GOAL_TAGS", {})
    
    # Create TEXT field
    print("Creating TEXT field...")
    if "description_plain" not in DATA.columns:
        DATA["description_plain"] = ""
    
    DATA["TEXT"] = (
        (DATA.get("tags", "").fillna("").astype(str)) + " " +
        (DATA.get("title", "").fillna("").astype(str)) + " " +
        (DATA.get("description_plain", "").fillna("").astype(str)) + " " +
        (DATA.get("product_type", "").fillna("").astype(str))
    ).str.lower()
    
    TEXT_ARRAY = DATA["TEXT"].to_numpy()
    
    # Compile regex patterns
    print("Compiling regex patterns...")
    AGE_RX = {b: _kws_to_compiled(AGE_MAP.get(b, [])) for b in AGE_MAP}
    
    def _cars_syns_guard(syns: List[str]) -> List[str]:
        # Since we've already cleaned up the maps, we don't need to filter out vehicle terms
        # that are part of specific car-related phrases
        bad = {"transport","transportation"}  # Only remove generic transport terms
        return [s for s in syns if s and s.strip().lower() not in bad]
    
    INTEREST_RX = {}
    for k, syns in INTEREST_TAGS.items():
        if str(k).strip().lower() == "cars":
            syns = _cars_syns_guard(list(syns))
        INTEREST_RX[k] = _kws_to_compiled(syns or [k])
    
    GOAL_RX = {g: _kws_to_compiled(GOAL_TAGS.get(g, [g])) for g in GOAL_TAGS}
    
    # Pre-compute masks
    print("Computing AGE masks...")
    AGE_MASKS = {}
    for b, rx in AGE_RX.items():
        if not rx:
            continue
        try:
            mask = DATA["TEXT"].str.contains(rx, regex=True, na=False).to_numpy(dtype=bool)
            AGE_MASKS[b] = mask
            print(f"  {b}: {np.sum(mask)} matches")
        except Exception as e:
            print(f"  Error with {b}: {e}")
    
    print("Computing INTEREST masks...")
    INTEREST_MASKS = {}
    INTEREST_INDICES = {}
    for k, rx in INTEREST_RX.items():
        if not rx:
            continue
        try:
            mask = DATA["TEXT"].str.contains(rx, regex=True, na=False).to_numpy(dtype=bool)
            INTEREST_MASKS[k] = mask
            idx = np.nonzero(mask)[0]
            INTEREST_INDICES[k] = idx
            print(f"  {k}: {len(idx)} matches")
        except Exception as e:
            print(f"  Error with {k}: {e}")
    
    print("Computing GOAL masks...")
    GOAL_MASKS = {}
    for g, rx in GOAL_RX.items():
        if not rx:
            continue
        try:
            mask = DATA["TEXT"].str.contains(rx, regex=True, na=False).to_numpy(dtype=bool)
            GOAL_MASKS[g] = mask
            print(f"  {g}: {np.sum(mask)} matches")
        except Exception as e:
            print(f"  Error with {g}: {e}")
    
    # Create KEY_TO_IDX mapping
    print("Creating KEY_TO_IDX mapping...")
    KEY_TO_IDX = {}
    if "id" in DATA.columns and "handle" in DATA.columns:
        _ids = DATA.get("id", pd.Series([None]*len(DATA))).to_list()
        _handles = DATA.get("handle", pd.Series([None]*len(DATA))).to_list()
        for _i, (_id, _h) in enumerate(zip(_ids, _handles)):
            KEY_TO_IDX[(_id, _h)] = _i
    else:
        for _i in range(len(DATA)):
            KEY_TO_IDX[(DATA.iloc[_i].get("id", _i), f"handle_{_i}")] = _i
    
    # Save all pre-computed data
    print("Saving pre-computed masks...")
    
    masks_data = {
        "AGE_MASKS": AGE_MASKS,
        "INTEREST_MASKS": INTEREST_MASKS,
        "INTEREST_INDICES": INTEREST_INDICES,
        "GOAL_MASKS": GOAL_MASKS,
        "KEY_TO_IDX": KEY_TO_IDX,
        "TEXT_ARRAY": TEXT_ARRAY,
        "AGE_RX": {k: v.pattern if v else None for k, v in AGE_RX.items()},
        "INTEREST_RX": {k: v.pattern if v else None for k, v in INTEREST_RX.items()},
        "GOAL_RX": {k: v.pattern if v else None for k, v in GOAL_RX.items()},
        "total_rows": len(DATA)
    }
    
    with open("precomputed_masks.pkl", "wb") as f:
        pickle.dump(masks_data, f)
    
    print(f"✅ Saved pre-computed masks to precomputed_masks.pkl")
    print(f"✅ Total rows processed: {len(DATA)}")
    print(f"✅ AGE masks: {len(AGE_MASKS)}")
    print(f"✅ INTEREST masks: {len(INTEREST_MASKS)}")
    print(f"✅ GOAL masks: {len(GOAL_MASKS)}")
    
    return True

if __name__ == "__main__":
    success = precompute_masks()
    if success:
        print("\n🎉 Pre-computed masks ready!")
        print("📝 Now you can use fast startup with full regex filtering:")
        print("   python app_fast_latest.py")
    else:
        print("\n❌ Failed to pre-compute masks")
