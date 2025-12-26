import os, re, pickle, json, math
from typing import List, Dict, Any, Optional, Union

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(title="Toy Recommendation API (Fast Path)", version="4.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

# ---------------- Data load ----------------
PKL_PATH = os.getenv("PICKLE_PATH", "product_search_index.pkl")
CSV_PATH = os.getenv("CSV_PATH", "shopify_products_export.csv")
MAPS_PATH = os.getenv("MAPS_PATH", "generated_maps.json")

DATA: pd.DataFrame
EMB: Optional[np.ndarray] = None
NORM_EMB: Optional[np.ndarray] = None
try:
    with open(PKL_PATH, "rb") as f:
        payload = pickle.load(f)
    DATA = payload["data"]
    EMB = payload["embeddings"].astype("float32")
    print(f"✅ Loaded PKL: {len(DATA)} rows, embeddings {EMB.shape}")
except FileNotFoundError:
    print(f"⚠️ PKL not found at {PKL_PATH}, loading CSV fallback {CSV_PATH}")
    DATA = pd.read_csv(CSV_PATH, low_memory=False)
    EMB = None

if EMB is not None:
    _norms = np.linalg.norm(EMB, axis=1, keepdims=True) + 1e-8
    NORM_EMB = (EMB / _norms).astype("float32")

# ---------------- Maps ----------------
try:
    with open(MAPS_PATH, "r") as f:
        maps = json.load(f)
    AGE_MAP = maps.get("AGE_MAP", {})
    INTEREST_TAGS = maps.get("INTEREST_TAGS", {})
    GOAL_TAGS = maps.get("GOAL_TAGS", {})
    print(f"✅ Loaded tag maps from {MAPS_PATH}")
except FileNotFoundError:
    print(f"⚠️ No {MAPS_PATH} found, using conservative fallbacks")
    AGE_MAP = {
        "0-2 yr": ["0-2 years","0-24 months","0-11 months","1-2 years","1-3 years"],
        "3-5 yr": ["3-6 years","3-4 years","4-6 years"],
        "6-8 yr": ["6-8 years","7-12 years","6-12 years","7-8 years"],
        "9-11 yr": ["7-12 years","9-12 years","10-12 years"],
    }
    INTEREST_TAGS = {
        "Cars": ["toy car","toy cars","ride-on car","jeep toy","mini car","diecast car","pull back car","remote control car"],  # more specific car terms
        "Robots": ["robot","robotics","coding robot","stem robot","programmable"],
        "Books": ["book","story book","board book","picture book"],
        "Animals": ["animal","animals","zoo","wildlife","dog","cat","puppy"],
    }
    GOAL_TAGS = {
        "Motor Skills": ["motor","fine motor","gross motor","coordination"],
        "Social Skills": ["social","cooperative","turn taking","group play"],
        "Creativity": ["creative","imagination","art","craft"],
        "Thinking": ["thinking","logic","problem","puzzle","brain","strategy"],
    }

# ---------------- TEXT field ----------------
if "description_plain" not in DATA.columns:
    DATA["description_plain"] = ""
DATA["TEXT"] = (
    (DATA.get("tags", "").fillna("").astype(str)) + " " +
    (DATA.get("title", "").fillna("").astype(str)) + " " +
    (DATA.get("description_plain", "").fillna("").astype(str)) + " " +
    (DATA.get("product_type", "").fillna("").astype(str))
).str.lower()


# ---------------- Regex builders (compiled at startup) ----------------
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
    bad = {
        "vehicle", "vehicles", "transport", "transportation",
        "construction vehicle", "construction vehicles",
        "model_construction_vehicles", "model_vehicle_set_multi",
        "model_vehicles_single", "car & vehicles_set",
        "cars & vehicles", "cars / vehicles set",
        "fun play toys cars vehicles & more_0-11 months",
        "fun play toys cars vehicles & more_0-2",
        "fun play toys cars vehicles & more_1-3 years",
        "ride-ons walkers & more_0-11 months",
        "ride-ons walkers & more_0-2", 
        "ride-ons walkers & more_1-3 years"
    }
    return [s for s in syns if s and s.strip().lower() not in bad]

INTEREST_RX: Dict[str, Optional[re.Pattern]] = {}
for k, syns in INTEREST_TAGS.items():
    if str(k).strip().lower() == "cars":
        syns = _cars_syns_guard(list(syns))
    INTEREST_RX[k] = _kws_to_compiled(syns or [k])

GOAL_RX: Dict[str, Optional[re.Pattern]] = {g: _kws_to_compiled(GOAL_TAGS.get(g, [g])) for g in GOAL_TAGS}


# ---------------- Precomputed masks and centroids (startup) ----------------
TEXT_ARRAY = DATA["TEXT"].to_numpy()

# Row key to index for fast lookups using (id, handle)
KEY_TO_IDX: Dict[tuple, int] = {}
_ids = DATA.get("id", pd.Series([None]*len(DATA))).to_list()
_handles = DATA.get("handle", pd.Series([None]*len(DATA))).to_list()
for _i, (_id, _h) in enumerate(zip(_ids, _handles)):
    KEY_TO_IDX[( _id, _h )] = _i

# Global centroid for FAISS seeding
GLOBAL_CENTROID: Optional[np.ndarray] = None
if 'EMB' in globals() and EMB is not None and EMB.size > 0:
    _gc = EMB.mean(axis=0, keepdims=True).astype("float32")
    _gc /= (np.linalg.norm(_gc, axis=1, keepdims=True) + 1e-8)
    GLOBAL_CENTROID = _gc

INTEREST_MASKS: Dict[str, np.ndarray] = {}
INTEREST_INDICES: Dict[str, np.ndarray] = {}
INTEREST_CENTROIDS: Dict[str, np.ndarray] = {}
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

AGE_MASKS: Dict[str, np.ndarray] = {}
for b, rx in AGE_RX.items():
    if not rx:
        continue
    try:
        AGE_MASKS[b] = DATA["TEXT"].str.contains(rx, regex=True, na=False).to_numpy(dtype=bool)
    except Exception:
        continue

GOAL_MASKS: Dict[str, np.ndarray] = {}
for g, rx in GOAL_RX.items():
    if not rx:
        continue
    try:
        GOAL_MASKS[g] = DATA["TEXT"].str.contains(rx, regex=True, na=False).to_numpy(dtype=bool)
    except Exception:
        continue

# Balanced goal centroids creation
GOAL_CENTROIDS: Dict[str, np.ndarray] = {}
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

# ---------------- FAISS (IVF preferred, Flat fallback) ----------------
HAS_FAISS = False
INDEX = None
DIM = None
try:
    import faiss
    if EMB is not None and EMB.size > 0:
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
except Exception as e:
    print(f"⚠️ FAISS IVF unavailable: {e}. Will fallback to Flat search if needed.")


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
    blockers = ["book","board book","picture book","books","activity","activity kit","kit","workbook","worksheet","building block","building blocks","construction","mechanix","blocks"]
    return any(b in t for b in blockers)

def _is_actual_toy_car(r: Dict[str, Any]) -> bool:
    """Check if this is actually a toy car, not a building block with car theme."""
    t = (str(r.get("title","")) + " " + str(r.get("product_type","")) + " " + str(r.get("tags",""))).lower()
    
    # Negative indicators (building/construction items)
    construction_indicators = [
        "building block", "construction", "mechanix", "blocks", "lego", "duplo", "mega bloks",
        "building set", "construction set", "mechanix set", "building kit", "construction kit"
    ]
    
    # Check for construction indicators first (higher priority)
    has_construction_indicator = any(indicator in t for indicator in construction_indicators)
    if has_construction_indicator:
        return False
    
    # Check for any car indicator
    car_indicators = [
        "toy car", "diecast", "pull back", "remote control", "mini car", "scale model", 
        "hot wheels", "matchbox", "rc car", "racing car", "car toy", "car track",
        "car carrier", "car set", "car collection", "car pack", "car bundle"
    ]
    
    has_car_indicator = any(indicator in t for indicator in car_indicators)
    if has_car_indicator:
        return True
    
    # Check for generic "car" but only if not in construction context
    if "car" in t and not any(construction_word in t for construction_word in ["block", "construction", "mechanix", "building"]):
        # Additional check: if it's a generic "car" term, make sure it's not a building set
        product_type = str(r.get("product_type", "")).lower()
        if "building" in product_type or "construction" in product_type or "mechanix" in product_type:
            return False
        return True
    
    return False

def _get_car_priority(r: Dict[str, Any]) -> int:
    """Get priority for car ranking: 1=RC, 2=Diecast, 3=Others"""
    t = (str(r.get("title","")) + " " + str(r.get("product_type","")) + " " + str(r.get("tags",""))).lower()
    
    # RC cars (highest priority)
    rc_indicators = ["remote control", "rc car", "rc cars", "battery operated car", "electric car"]
    if any(indicator in t for indicator in rc_indicators):
        return 1
    
    # Diecast cars (second priority)
    diecast_indicators = ["diecast", "die cast", "scale model", "hot wheels", "matchbox", "metal car"]
    if any(indicator in t for indicator in diecast_indicators):
        return 2
    
    # Other cars (lowest priority)
    return 3

def matches_interest_guarded(r: Dict[str,Any], interest: str, rx: Optional[re.Pattern], text: Optional[str]=None) -> bool:
    if not rx: return False
    if str(interest).strip().lower()=="cars":
        return _is_actual_toy_car(r)
    if text is None: text = row_text(r)
    return rx.search(text) is not None

def _match_interest_by_mask_or_regex(r: Dict[str,Any], idx: Optional[int], interest: str) -> bool:
    """Fast path: use precomputed mask if we have row index; fallback to regex check.
    Keeps the Cars guard intact.
    """
    if str(interest).strip().lower()=="cars":
        # Use the more sophisticated car detection
        return _is_actual_toy_car(r)
    
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


# ---------------- Endpoint (FAISS-first) ----------------
@app.post("/recommend", response_model=RecommendResp)
def recommend(req: RecommendReq):
    prof = normalize_profile(req.profile)
    lo = req.price_min if (req.price_min is not None and req.price_min>0) else None
    hi = req.price_max if (req.price_max is not None and req.price_max>0) else None

    # Query vector: use precomputed interest/global centroids (no per-request scan)
    if EMB is not None and HAS_FAISS:
        centroids = []
        for i in prof["interests"]:
            q = INTEREST_CENTROIDS.get(i)
            if q is not None:
                centroids.append(q)
        if centroids:
            qv = np.mean(np.vstack(centroids), axis=0, keepdims=True).astype("float32")
            qv /= (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-8)
        elif GLOBAL_CENTROID is not None:
            qv = GLOBAL_CENTROID
        else:
            # last resort: mean of head without scanning contains
            head = min(len(DATA), 2000)
            qv = EMB[:head].mean(axis=0, keepdims=True).astype("float32")
            qv /= (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-8)
    else:
        qv = None

    def faiss_search(k: int) -> List[tuple]:
        out: List[tuple] = []
        if HAS_FAISS and qv is not None:
            D, I = INDEX.search(qv, min(k*3, len(DATA)))  # slight overfetch
            for sim, gi in zip(D[0], I[0]):
                if gi < 0 or gi >= len(DATA):
                    continue
                r = DATA.iloc[int(gi)].to_dict()
                out.append((r, float(sim)))
        else:
            # Fallback: simple scan (limited to k*3 rows)
            head = min(len(DATA), k*3)
            for i in range(head):
                r = DATA.iloc[i].to_dict()
                out.append((r, 0.5))
        return out

    def faiss_search_for_interest(interest: str, k: int) -> List[tuple]:
        """Search FAISS using a precomputed centroid for this interest (no per-request scan)."""
        out: List[tuple] = []
        if not HAS_FAISS or EMB is None:
            return out
        qv_i = INTEREST_CENTROIDS.get(interest)
        if qv_i is None:
            return out
        D, I = INDEX.search(qv_i, min(k*2, len(DATA)))
        for sim, gi in zip(D[0], I[0]):
            if gi < 0 or gi >= len(DATA):
                continue
            r = DATA.iloc[int(gi)].to_dict()
            out.append((r, float(sim)))
        return out

    def faiss_search_for_goals(goals: List[str], k: int) -> List[tuple]:
        """Enrich candidates using centroid averaged across requested goals."""
        out: List[tuple] = []
        if not HAS_FAISS or EMB is None or not goals:
            return out
        cents = []
        for g in goals:
            q = GOAL_CENTROIDS.get(g)
            if q is not None:
                cents.append(q)
        if not cents:
            return out
        qv_g = np.mean(np.vstack(cents), axis=0, keepdims=True).astype("float32")
        qv_g /= (np.linalg.norm(qv_g, axis=1, keepdims=True) + 1e-8)
        D, I = INDEX.search(qv_g, min(k*2, len(DATA)))
        for sim, gi in zip(D[0], I[0]):
            if gi < 0 or gi >= len(DATA):
                continue
            r = DATA.iloc[int(gi)].to_dict()
            out.append((r, float(sim)))
        return out

    def passes_filters(r: Dict[str,Any]) -> bool:
        try:
            price = float(r.get("variant_price") or 0.0)
        except Exception:
            price = 0.0
        if (lo is not None and price < lo) or (hi is not None and price > hi):
            return False
        # age bands OR
        bands = overlapping_age_bands(req.profile)
        if bands:
            # use precomputed age masks where available
            idx = KEY_TO_IDX.get((r.get("id"), r.get("handle")))
            matched_age = False
            if idx is not None:
                for b in bands:
                    m = AGE_MASKS.get(b)
                    if m is not None and bool(m[idx]):
                        matched_age = True
                        break
            else:
                t = row_text(r)
                for b in bands:
                    rx_b = AGE_RX.get(b)
                    if rx_b and rx_b.search(t):
                        matched_age = True
                        break
            if not matched_age:
                return False
        # interests OR
        if prof["interests"]:
            idx = KEY_TO_IDX.get((r.get("id"), r.get("handle")))
            if not any(_match_interest_by_mask_or_regex(r, idx, i) for i in prof["interests"]):
                return False
        # goals AND
        for g in prof["developmental_goals"]:
            idx = KEY_TO_IDX.get((r.get("id"), r.get("handle")))
            m = GOAL_MASKS.get(g)
            if m is not None and idx is not None:
                if not bool(m[idx]):
                    return False
            else:
                rx = GOAL_RX.get(g)
                if not rx: continue
                if not rx.search(row_text(r)):
                    return False
        # active flag
        status = str(r.get("status",""))
        if status:
            if status.lower() != "active":
                return False
        return True

    count, skip = int(req.count), int(req.skip)
    topK = max(count*3, skip + count*2)
    candidates = faiss_search(topK)
    # Enrich candidate set with per-interest searches to preserve multi-interest coverage
    if prof["interests"]:
        import math as _math
        min_quota = max(1, _math.ceil(count / max(1, len(prof["interests"]))))
        extra_k = max(60, min_quota * 20)
        seen_keys = set((r.get("id"), r.get("handle")) for r, _ in candidates)
        for intr in prof["interests"]:
            for r, s in faiss_search_for_interest(intr, extra_k):
                key = (r.get("id"), r.get("handle"))
                if key in seen_keys:
                    continue
                candidates.append((r, s))
                seen_keys.add(key)

    # Enrich by goals when provided (use averaged goal centroid)
    if req.profile.get("developmental_goals"):
        goals = [str(g).title() for g in req.profile.get("developmental_goals", [])]
        extra_k = max(60, count * 10)
        seen_keys = set((r.get("id"), r.get("handle")) for r, _ in candidates)
        for r, s in faiss_search_for_goals(goals, extra_k):
            key = (r.get("id"), r.get("handle"))
            if key in seen_keys:
                continue
            candidates.append((r, s))
            seen_keys.add(key)

    # Filter only candidates
    filtered = [(r, s) for r, s in candidates if passes_filters(r)]
    if len(filtered) < count:
        candidates2 = faiss_search(topK*2)
        filtered = [(r, s) for r, s in candidates2 if passes_filters(r)]
        # Final backfill: if still short on goals-only queries, use precomputed GOAL_MASKS directly
        only_goals = not prof["interests"] and bool(req.profile.get("developmental_goals"))
        if only_goals and len(filtered) < count:
            # collect rows satisfying ALL goals
            goals = [str(g).title() for g in req.profile.get("developmental_goals", [])]
            if goals:
                mask_all = np.ones(len(DATA), dtype=bool)
                for g in goals:
                    gm = GOAL_MASKS.get(g)
                    if gm is None:
                        mask_all = np.zeros(len(DATA), dtype=bool)
                        break
                    mask_all &= gm
                # price + active filter
                active = product_active_series()
                price_ok = price_ok_series(lo, hi)
                final_mask = mask_all & active & price_ok
                # optional age filter
                bands = overlapping_age_bands(req.profile)
                if bands:
                    age_any = np.zeros(len(DATA), dtype=bool)
                    for b in bands:
                        am = AGE_MASKS.get(b)
                        if am is not None:
                            age_any |= am
                    final_mask &= age_any
                idxs = np.nonzero(final_mask)[0]
                # If we have normalized embeddings, rank backfill by semantic similarity to goal centroid
                if EMB is not None and NORM_EMB is not None and goals:
                    cents = []
                    for g in goals:
                        q = GOAL_CENTROIDS.get(g)
                        if q is not None:
                            cents.append(q)
                    if cents:
                        qv = np.mean(np.vstack(cents), axis=0, keepdims=True)
                        qv /= (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-8)
                        sims = (NORM_EMB[idxs] @ qv.T).reshape(-1)
                        order = np.argsort(-sims)
                        idxs = idxs[order]
                # backfill with these rows
                for gi in idxs:
                    r = DATA.iloc[int(gi)].to_dict()
                    key = (r.get("id"), r.get("handle"))
                    if any((key[0] == rr.get("id") and key[1] == rr.get("handle")) for rr, _ in filtered):
                        continue
                    score = float((NORM_EMB[int(gi)] @ qv.T).item()) if (EMB is not None and NORM_EMB is not None and 'qv' in locals()) else 0.4
                    filtered.append((r, score))
                    if len(filtered) >= count:
                        break

    # Subtype-aware ranking for motor skills
    if prof["developmental_goals"] and "Motor Skills" in prof["developmental_goals"]:
        # Define motor skills subtypes
        motor_subtypes = {
            "fine_motor": ["fine motor", "lacing", "bead", "busy board", "puzzle", "threading", "fingers", "hand eye", "dexterity"],
            "gross_motor": ["building block", "construction", "ride-on", "physical", "movement", "gross motor", "coordination"]
        }
        
        # Categorize products by subtype
        fine_motor_products = []
        gross_motor_products = []
        other_products = []
        
        for r, score in filtered:
            text = row_text(r).lower()
            is_fine_motor = any(keyword in text for keyword in motor_subtypes["fine_motor"])
            is_gross_motor = any(keyword in text for keyword in motor_subtypes["gross_motor"])
            
            if is_fine_motor and not is_gross_motor:
                fine_motor_products.append((r, score))
            elif is_gross_motor and not is_fine_motor:
                gross_motor_products.append((r, score))
            else:
                other_products.append((r, score))
        
        # Sort each subtype by score
        fine_motor_products.sort(key=lambda x: (-x[1], str(x[0].get("id"))))
        gross_motor_products.sort(key=lambda x: (-x[1], str(x[0].get("id"))))
        other_products.sort(key=lambda x: (-x[1], str(x[0].get("id"))))
        
        # Interleave fine and gross motor products for balanced results
        balanced_filtered = []
        max_fine = len(fine_motor_products)
        max_gross = len(gross_motor_products)
        max_other = len(other_products)
        
        for i in range(max(max_fine, max_gross, max_other)):
            if i < max_fine:
                balanced_filtered.append(fine_motor_products[i])
            if i < max_gross:
                balanced_filtered.append(gross_motor_products[i])
            if i < max_other:
                balanced_filtered.append(other_products[i])
        
        filtered = balanced_filtered
    elif prof["interests"] and "Cars" in prof["interests"]:
        # Car-specific ranking: RC cars first, then diecast, then others
        rc_cars = []
        diecast_cars = []
        other_cars = []
        
        for r, score in filtered:
            priority = _get_car_priority(r)
            if priority == 1:  # RC cars
                rc_cars.append((r, score))
            elif priority == 2:  # Diecast cars
                diecast_cars.append((r, score))
            else:  # Other cars
                other_cars.append((r, score))
        
        # Sort each category by score
        rc_cars.sort(key=lambda x: (-x[1], str(x[0].get("id"))))
        diecast_cars.sort(key=lambda x: (-x[1], str(x[0].get("id"))))
        other_cars.sort(key=lambda x: (-x[1], str(x[0].get("id"))))
        
        # Combine in priority order: RC first, then diecast, then others
        filtered = rc_cars + diecast_cars + other_cars
    else:
        # Regular ranking for other goals
        filtered.sort(key=lambda x: (-x[1], str(x[0].get("id"))))
    
    total_hits = len(filtered)
    filtered_rows_all = [r for r, _ in filtered]

    # Strict pagination: build a globally diversified sequence up to skip+count, then slice
    interests = prof["interests"] or []
    target_len = min(total_hits, skip + count)
    global_rows: List[Dict[str,Any]] = []
    seen_global = set()

    if req.diversify and interests:
        # Build per-interest lists from full ranked pool
        per: Dict[str, List[Dict[str,Any]]] = {i: [] for i in interests}
        for r in filtered_rows_all:
            idx = KEY_TO_IDX.get((r.get("id"), r.get("handle")))
            hits = [i for i in interests if _match_interest_by_mask_or_regex(r, idx, i)]
            if not hits:
                continue
            # assign to least-filled interest list to keep per buckets balanced
            target = min(hits, key=lambda i: len(per[i]))
            per[target].append(r)
        # Round-robin to build a diversified global list
        import math as _math
        while len(global_rows) < target_len and any(per[i] for i in interests):
            for i in interests:
                if per[i]:
                    r = per[i].pop(0)
                    key = (r.get("id"), r.get("handle"))
                    if key in seen_global:
                        continue
                    global_rows.append(r)
                    seen_global.add(key)
                    if len(global_rows) >= target_len:
                        break
        # backfill from remaining ranked rows if needed
        if len(global_rows) < target_len:
            for r in filtered_rows_all:
                key = (r.get("id"), r.get("handle"))
                if key in seen_global:
                    continue
                global_rows.append(r)
                seen_global.add(key)
                if len(global_rows) >= target_len:
                    break
    else:
        # No interests: diversify by vendor and lightly by product_type
        # Build vendor buckets from full ranked pool, maintaining order
        vendor_buckets: Dict[str, List[Dict[str,Any]]] = {}
        for r in filtered_rows_all:
            v = str(r.get("vendor") or "").strip().lower()
            vendor_buckets.setdefault(v, []).append(r)
        # Round-robin across vendors with a per-vendor cap to avoid monopolies
        import math as _math
        per_vendor_cap = max(2, _math.ceil(max(1, target_len) / max(1, len(vendor_buckets) or 1)))
        vendor_counts: Dict[str,int] = {v: 0 for v in vendor_buckets}
        # Also try not to repeat the same product_type too often in sequence
        pt_counts: Dict[str,int] = {}
        while len(global_rows) < target_len and any(vendor_buckets[v] for v in vendor_buckets):
            for v in list(vendor_buckets.keys()):
                if len(global_rows) >= target_len:
                    break
                if not vendor_buckets[v]:
                    continue
                if vendor_counts.get(v, 0) >= per_vendor_cap:
                    continue
                # pick the next row from this vendor that minimally increases dominant product_type
                candidates = vendor_buckets[v]
                best_idx = 0
                best_score = None
                for idx_c, cand in enumerate(candidates[:5]):  # lookahead small window
                    pt = str(cand.get("product_type") or "").strip().lower() or "nan"
                    score = pt_counts.get(pt, 0)
                    if best_score is None or score < best_score:
                        best_score = score
                        best_idx = idx_c
                r = candidates.pop(best_idx)
                key = (r.get("id"), r.get("handle"))
                if key in seen_global:
                    continue
                global_rows.append(r)
                seen_global.add(key)
                vendor_counts[v] = vendor_counts.get(v, 0) + 1
                pt = str(r.get("product_type") or "").strip().lower() or "nan"
                pt_counts[pt] = pt_counts.get(pt, 0) + 1
        # backfill if still short
        if len(global_rows) < target_len:
            for r in filtered_rows_all:
                if len(global_rows) >= target_len:
                    break
                key = (r.get("id"), r.get("handle"))
                if key in seen_global:
                    continue
                global_rows.append(r)
                seen_global.add(key)

    # Final page slice strictly from skip to skip+count
    rows = global_rows[skip: skip + count]

    def shape_item(r: Dict[str,Any]) -> Dict[str,Any]:
        # price and discounted price
        def _to_float(x, default=0.0):
            try:
                val = float(x)
                if not math.isfinite(val):
                    return default
                return val
            except Exception:
                return default
        price = _to_float(r.get("variant_price") or r.get("price") or 0.0)
        discounted_price = _to_float(r.get("discounted_price") or price)

        # tags normalization
        tags: List[str] = []
        raw_tags = r.get("tags", "")
        if isinstance(raw_tags, list):
            tags = [str(t).strip().lower() for t in raw_tags if str(t).strip()]
        elif isinstance(raw_tags, str):
            tags = [t.strip().lower() for t in re.split(r"[;,|]", raw_tags) if t.strip()]

        # images extraction
        def _ensure_list(val):
            if val is None:
                return []
            if isinstance(val, list):
                return val
            if isinstance(val, (tuple, set)):
                return list(val)
            if isinstance(val, dict):
                return [val]
            if isinstance(val, str):
                v = val.strip()
                if v.startswith("[") and v.endswith("]"):
                    try:
                        parsed = json.loads(v)
                        return parsed if isinstance(parsed, list) else [parsed]
                    except Exception:
                        pass
                # split common separators
                return [p for p in re.split(r"[\s,;|]", v) if p]
            return []

        def _extract_images(row: Dict[str,Any]) -> List[str]:
            image_candidates = []
            for k in ("images", "image_urls", "image_url", "image_src", "image"):
                if k in row and row[k] is not None:
                    image_candidates.extend(_ensure_list(row[k]))
            urls: List[str] = []
            for it in image_candidates:
                if isinstance(it, dict):
                    src = it.get("src") or it.get("url")
                    if src and isinstance(src, str):
                        urls.append(src.strip())
                elif isinstance(it, str):
                    s = it.strip()
                    if s:
                        urls.append(s)
            # basic filtering and de-dup
            def _looks_like_url(u: str) -> bool:
                return u.startswith("http") or u.endswith(('.jpg','.jpeg','.png','.webp','.gif'))
            seen = set()
            out = []
            for u in urls:
                if not _looks_like_url(u):
                    continue
                if u in seen:
                    continue
                seen.add(u)
                out.append(u)
                if len(out) >= 6:
                    break
            return out

        def _extract_variants(row: Dict[str,Any]) -> List[Dict[str,Any]]:
            raw = row.get("variants")
            vs: List[Dict[str,Any]] = []
            parsed = None
            if isinstance(raw, list):
                parsed = raw
            elif isinstance(raw, str) and raw.strip():
                s = raw.strip()
                if s.startswith("[") and s.endswith("]"):
                    try:
                        parsed = json.loads(s)
                    except Exception:
                        parsed = None
            if isinstance(parsed, list):
                for v in parsed:
                    if isinstance(v, dict):
                        vs.append({
                            "id": v.get("id"),
                            "title": v.get("title") or v.get("name") or "",
                            "price": _to_float(v.get("price") or v.get("variant_price") or price),
                            "compare_at_price": _to_float(v.get("compare_at_price") or 0.0),
                            "sku": v.get("sku"),
                            "inventory_quantity": v.get("inventory_quantity")
                        })
            else:
                # synthesize a single variant from row-level fields
                vs.append({
                    "id": r.get("variant_id") or r.get("id"),
                    "title": r.get("variant_title") or "default",
                    "price": price,
                    "compare_at_price": _to_float(r.get("compare_at_price") or 0.0),
                    "sku": r.get("sku"),
                    "inventory_quantity": r.get("inventory_quantity")
                })
            return vs

        images_urls = _extract_images(r)
        # Normalize to list of objects with 'src' to avoid empty {} in some viewers
        images = [{"src": u} for u in images_urls if isinstance(u, str) and u]
        # Also build detailed images array as requested (url, altText, width, height)
        alt_candidate = str(r.get("image_alt") or r.get("altText") or "").strip()
        width_candidate = r.get("image_width"); height_candidate = r.get("image_height")
        def _to_int(x):
            try:
                return int(x)
            except Exception:
                return None
        images_v2 = [{"url": u, "altText": alt_candidate, "width": _to_int(width_candidate) or 800, "height": _to_int(height_candidate) or 800} for u in images_urls]

        variants = _extract_variants(r)
        # Also expose variants in the alt format (string prices)
        def _money_str(x: float) -> str:
            try:
                return f"{float(x):.2f}"
            except Exception:
                return "0.00"
        variants_alt: List[Dict[str,Any]] = []
        for v in variants:
            variants_alt.append({
                "id": v.get("id"),
                "title": v.get("title"),
                "price": _money_str(v.get("price")),
                "compare_at_price": _money_str(v.get("compare_at_price")),
                "sku": v.get("sku"),
                "image": v.get("image")
            })

        # collections
        def _extract_string_list(val) -> List[str]:
            if val is None:
                return []
            if isinstance(val, list):
                return [str(x).strip() for x in val if str(x).strip()]
            if isinstance(val, str):
                s = val.strip()
                if not s:
                    return []
                if s.startswith("[") and s.endswith("]"):
                    try:
                        arr = json.loads(s)
                        return [str(x).strip() for x in (arr if isinstance(arr, list) else [arr]) if str(x).strip()]
                    except Exception:
                        pass
                return [t.strip() for t in re.split(r"[;,|]", s) if t.strip()]
            return []
        collections = _extract_string_list(r.get("collections") or r.get("collection_titles") or r.get("collection"))

        # options (try native, else derive from variants' titles)
        options = []
        raw_options = r.get("options")
        if isinstance(raw_options, list):
            options = raw_options
        elif isinstance(raw_options, str) and raw_options.strip().startswith("["):
            try:
                options = json.loads(raw_options)
            except Exception:
                options = []
        if not options:
            # Fallback: single Title option with unique variant titles
            v_titles = list({str(v.get("title") or "Default Title") for v in variants})
            if v_titles:
                options = [{"name": "Title", "values": v_titles}]

        # discount and multiple price
        compare_at = _to_float(r.get("compare_at_price") or 0.0)
        discount = 0
        if compare_at and compare_at > 0 and price < compare_at:
            try:
                discount = int(round((compare_at - price) * 100.0 / compare_at))
            except Exception:
                discount = 0
        has_multiple_price = False
        if len(variants) > 1:
            prices_set = {round(float(v.get("price") or 0.0), 2) for v in variants}
            has_multiple_price = len(prices_set) > 1

        # extra review fields
        rc = r.get("reviews_count")
        ra = r.get("reviews_average")
        try:
            reviews_count = int(rc) if rc is not None and rc == rc else None
        except Exception:
            reviews_count = None
        reviews_average = _to_float(ra, None) if ra is not None else None

        return {
            "id": r.get("id"), "title": r.get("title",""), "vendor": r.get("vendor",""),
            "handle": r.get("handle",""), "tags": tags, "status": str(r.get("status","")).upper(),
            "product_type": r.get("product_type"), "category": r.get("category"),
            "price": price, "discounted_price": discounted_price, "discount": discount, "hasMultiplePrice": has_multiple_price,
            "images": images_v2, "variants": variants_alt,
            "collections": collections, "options": options,
            "reviews_count": reviews_count, "reviews_average": reviews_average,
            "_rank": 0.0, "_original_rank": 0.0,
            "isActive": 1 if str(r.get("status","")).lower()=="active" else 0
        }

    # sanitize items to avoid NaN/inf in JSON
    def _sanitize(v):
        if isinstance(v, dict):
            return {k: _sanitize(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_sanitize(x) for x in v]
        if isinstance(v, float):
            return float(v) if math.isfinite(v) else None
        return v
    items = [_sanitize(shape_item(r)) for r in rows]

    meta: Dict[str, Union[int,float,bool,dict]] = {
        "total_hits": total_hits, "returned": len(items),
        "skip": req.skip, "count": req.count,
        "has_more": (skip + count) < total_hits
    }

    if req.return_debug:
        # simple classification audit
        dbg = {"interests": interests}
        if interests:
            counts = {i: 0 for i in interests}
            details = []
            for r in rows:
                idx = KEY_TO_IDX.get((r.get("id"), r.get("handle")))
                hits = [i for i in interests if _match_interest_by_mask_or_regex(r, idx, i)]
                assigned = min(hits, key=lambda i: counts[i]) if hits else "Other"
                if assigned in counts: counts[assigned] += 1
                details.append({"id": r.get("id"), "handle": r.get("handle"), "matched_interests": hits, "assigned_interest": assigned})
            dbg["diversification"] = counts
            dbg["classification"] = details
        # goals per-item flags and counts
        sel_goals = [str(g).title() for g in req.profile.get("developmental_goals", [])]
        if sel_goals:
            gcounts: Dict[str,int] = {g: 0 for g in sel_goals}
            gdetails = []
            for r in rows:
                key = (r.get("id"), r.get("handle"))
                idx = KEY_TO_IDX.get(key)
                matched = []
                for g in sel_goals:
                    m = GOAL_MASKS.get(g)
                    ok = False
                    if m is not None and idx is not None:
                        ok = bool(m[idx])
                    else:
                        rx = GOAL_RX.get(g)
                        ok = bool(rx and rx.search(row_text(r)))
                    if ok:
                        matched.append(g)
                for g in matched:
                    gcounts[g] = gcounts.get(g, 0) + 1
                missing = [g for g in sel_goals if g not in matched]
                gdetails.append({"id": r.get("id"), "handle": r.get("handle"), "goals_matched": matched, "goals_missing": missing})
            dbg["goals_counts"] = gcounts
            dbg["goals_classification"] = gdetails
        # when no interests, expose vendor/product_type distribution on the page
        if not interests:
            vend_counts: Dict[str,int] = {}
            pt_counts: Dict[str,int] = {}
            for r in rows:
                v = str(r.get("vendor") or "").strip().lower()
                vend_counts[v] = vend_counts.get(v, 0) + 1
                pt = str(r.get("product_type") or "").strip().lower() or "nan"
                pt_counts[pt] = pt_counts.get(pt, 0) + 1
            dbg["diversification_vendor"] = vend_counts
            dbg["diversification_product_type"] = pt_counts
        meta["debug"] = dbg

    return {"results": items, "meta": meta}


@app.get("/health")
def health():
    return {"status": "ok", "rows": len(DATA), "faiss": bool(HAS_FAISS)}


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting server at http://localhost:8002/docs")
    uvicorn.run(app, host="0.0.0.0", port=8008)


