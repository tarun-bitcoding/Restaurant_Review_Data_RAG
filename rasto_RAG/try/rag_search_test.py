"""
RAG-ish search over restaurant reviews using a single global FAISS index.

Pipeline:
1) Retrieve top-N candidates by semantic similarity (FAISS over review/text_blob).
2) (Optional) Filter/boost by lexical match of dish keywords from the query (e.g., "maggi").
3) Compute sentiment for candidate reviews (VADER; falls back to a tiny rule-based scorer if NLTK unavailable).
4) (Optional) Cross-encoder re-ranking for query<->review pairs (if model available).
5) Combine normalized scores: cross-encoder (or semantic), sentiment positivity, and rating → final score.
6) Aggregate per-restaurant and return top results with supporting review snippets.

Files expected (from your previous build step):
  - indices/global_reviews.faiss
  - indices/global_reviews.pkl   (list[dict], includes columns: Restaurant, Review, Rating, ...)

Usage:
  python rag_search.py --query "which restaurant has good maggi?" --top_k 5

Install (typical):
  pip install sentence-transformers faiss-cpu pandas numpy nltk rapidfuzz

Notes:
- Cross-encoder is optional. If you want it, it will auto-download:
    from sentence_transformers import CrossEncoder
    model: "cross-encoder/ms-marco-MiniLM-L-12-v2"
- VADER sentiment requires NLTK data; this script auto-downloads it if needed.
"""

import os
import re
import sys
import math
import json
import pickle
import argparse
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import faiss

from sentence_transformers import SentenceTransformer

# --- Optional deps (graceful fallbacks) ---
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False

try:
    from sentence_transformers import CrossEncoder  # optional
    _HAS_CE = True
except Exception:
    _HAS_CE = False

try:
    from rapidfuzz import fuzz
    _HAS_RF = True
except Exception:
    _HAS_RF = False

# ----------------- Config -----------------
INDEX_FILE = "indices/global_reviews.faiss"
META_FILE = "indices/global_reviews.pkl"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# scoring weights (tweak as needed)
WEIGHTS_WITH_CE = {
    "rerank": 0.45,   # cross-encoder relevance
    "semantic": 0.25, # FAISS similarity
    "sentiment": 0.15,
    "rating": 0.15,
    "mention": 0.02,  # small boost if dish explicitly mentioned
}

WEIGHTS_NO_CE = {
    "rerank": 0.00,
    "semantic": 0.50,
    "sentiment": 0.25,
    "rating": 0.23,
    "mention": 0.02,
}

TOP_R = 200  # retrieve this many from FAISS before re-ranking/combining

# Minimal stopword list for keyword extraction
MINI_STOP = set("""
a an the and or but if then so to of in on for with by from is are was were be been being at as it this that those these there here i you he she they them we us our your
""".split())

# --------------- Helpers ------------------

def _maybe_init_vader():
    if not _HAS_NLTK:
        return None
    try:
        # make sure lexicon is available
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except LookupError:
            nltk.download('vader_lexicon', quiet=True)
        return SentimentIntensityAnalyzer()
    except Exception:
        return None


_DEF_POS_WORDS = {"good", "great", "best", "tasty", "delicious", "yummy", "loved", "awesome", "amazing"}
_DEF_NEG_WORDS = {"bad", "worst", "awful", "terrible", "hate", "disgusting", "bland"}


def rule_sentiment(text: str) -> float:
    """Very small fallback sentiment in [0,1].
    + counts simple positive words, - counts negative; sigmoid-map to [0,1].
    """
    t = text.lower()
    pos = sum(1 for w in _DEF_POS_WORDS if w in t)
    neg = sum(1 for w in _DEF_NEG_WORDS if w in t)
    score = pos - neg
    # map to 0..1 via sigmoid
    return 1 / (1 + math.exp(-score))


def sentiment_score(text: str, sia: SentimentIntensityAnalyzer | None) -> float:
    if sia is not None:
        try:
            sc = sia.polarity_scores(text or "")
            # compound in [-1,1] → map to [0,1]
            return (sc.get("compound", 0.0) + 1) / 2
        except Exception:
            pass
    return rule_sentiment(text or "")


def normalize01(arr: np.ndarray) -> np.ndarray:
    if len(arr) == 0:
        return arr
    a = arr.astype(float)
    mn, mx = np.min(a), np.max(a)
    if mx - mn < 1e-12:
        return np.ones_like(a) * (0.5 if mx == 0 else 1.0)
    return (a - mn) / (mx - mn)


def extract_keywords(query: str) -> List[str]:
    # very small heuristic keyword extractor
    toks = re.findall(r"[a-zA-Z]+", (query or "").lower())
    keywords = [t for t in toks if t not in MINI_STOP and len(t) > 2]
    return keywords


def contains_any(text: str, keys: List[str]) -> bool:
    if not keys:
        return True
    t = (text or "").lower()
    for k in keys:
        if not k:
            continue
        if k in t:
            return True
        if _HAS_RF and fuzz.partial_ratio(k, t) > 85:
            return True
    return False


# --------------- Core ---------------------

def load_resources():
    if not os.path.exists(INDEX_FILE):
        raise FileNotFoundError(f"Missing FAISS index: {INDEX_FILE}")
    if not os.path.exists(META_FILE):
        raise FileNotFoundError(f"Missing metadata pkl: {META_FILE}")

    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
    df = pd.DataFrame(meta)

    emb_model = SentenceTransformer(EMB_MODEL)

    sia = _maybe_init_vader()

    cross_encoder = None
    if _HAS_CE:
        try:
            cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        except Exception:
            cross_encoder = None

    return index, df, emb_model, sia, cross_encoder


def faiss_sim_to_similarity(distances: np.ndarray) -> np.ndarray:
    """Convert L2 distances to a soft similarity [0,1]."""
    # similarity ~ 1 / (1 + d)
    return 1.0 / (1.0 + distances)


def search(query: str, top_k: int = 5, require_keyword: bool = False,
           weights: Dict[str, float] | None = None) -> Tuple[pd.DataFrame, List[Dict]]:
    index, df, emb_model, sia, cross_encoder = load_resources()

    has_ce = cross_encoder is not None
    W = WEIGHTS_WITH_CE if has_ce else WEIGHTS_NO_CE
    if weights:
        W.update(weights)

    # encode query
    qv = emb_model.encode([query]).astype("float32")

    # FAISS retrieve
    k = min(TOP_R, len(df))
    D, I = index.search(qv, k)
    distances = D[0]
    indices = I[0]

    # assemble candidate frame
    cand = df.iloc[indices].copy()
    cand["_faiss_dist"] = distances
    cand["_faiss_sim"] = faiss_sim_to_similarity(cand["_faiss_dist"].values)

    # optional lexical filtering/boosting
    keys = extract_keywords(query)
    if require_keyword:
        cand = cand[cand["Review"].fillna("").astype(str).str.lower().apply(lambda t: contains_any(t, keys))]
    # mention boost if any key present
    cand["_mention"] = cand["Review"].fillna("").astype(str).str.lower().apply(lambda t: 1.0 if contains_any(t, keys) else 0.0)

    if cand.empty:
        return cand, []

    # sentiment for candidates
    cand["_sentiment"] = cand["Review"].fillna("").astype(str).apply(lambda t: sentiment_score(t, sia))

    # rating normalization
    if "Rating" in cand.columns:
        cand["_rating"] = pd.to_numeric(cand["Rating"], errors="coerce").fillna(0.0) / 5.0
    else:
        cand["_rating"] = 0.0

    # cross-encoder reranking (optional)
    if has_ce:
        pairs = [(query, str(t)) for t in cand["Review"].fillna("").tolist()]
        try:
            ce_scores = cross_encoder.predict(pairs)  # shape: [n]
            cand["_rerank"] = ce_scores.astype(float)
        except Exception:
            cand["_rerank"] = 0.0
    else:
        cand["_rerank"] = 0.0

    # normalize components to [0,1]
    for col in ["_faiss_sim", "_sentiment", "_rating", "_mention", "_rerank"]:
        cand[f"{col}_n"] = normalize01(cand[col].to_numpy())

    # final score
    cand["_final"] = (
        W["rerank"]   * cand["_rerank_n"] +
        W["semantic"] * cand["_faiss_sim_n"] +
        W["sentiment"]* cand["_sentiment_n"] +
        W["rating"]   * cand["_rating_n"] +
        W["mention"]  * cand["_mention_n"]
    )

    # Aggregate per restaurant: take the best-scoring review as representative
    if "Restaurant" in cand.columns:
        grp = cand.groupby("Restaurant", as_index=False)["_final"].max()
        # join a representative row (the one with max _final) to fetch a nice snippet & rating
        idx_best = cand.sort_values(["Restaurant", "_final"], ascending=[True, False]).drop_duplicates("Restaurant").index
        rep = cand.loc[idx_best, ["Restaurant", "Review", "Rating", "_final"]]
        out = grp.merge(rep, on=["Restaurant", "_final"], how="left")
    else:
        # no restaurant column? just use candidates
        out = cand[["Review", "Rating", "_final"]].copy()
        out.rename(columns={"_final": "FinalScore"}, inplace=True)
        out.sort_values("FinalScore", ascending=False, inplace=True)
        return out.head(top_k), out.head(top_k).to_dict(orient="records")

    out.rename(columns={"_final": "FinalScore"}, inplace=True)
    out.sort_values("FinalScore", ascending=False, inplace=True)

    # limit to top_k restaurants
    top = out.head(top_k).copy()

    # prepare a compact result list
    results = []
    for _, r in top.iterrows():
        results.append({
            "Restaurant": r.get("Restaurant", ""),
            "Rating": float(r.get("Rating", 0) or 0),
            "ReviewSnippet": (str(r.get("Review", ""))[:220] + ("..." if len(str(r.get("Review", ""))) > 220 else "")),
            "FinalScore": float(r.get("FinalScore", 0.0)),
        })

    return top, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="User question, e.g., 'which restaurant has good maggi?'")
    ap.add_argument("--top_k", type=int, default=5, help="Number of restaurants to return")
    ap.add_argument("--require_keyword", action="store_true", help="If set, require that dish keyword(s) appear in the review text")
    ap.add_argument("--weights", type=str, default=None, help="JSON dict to override weights, e.g. '{'semantic':0.4,'rating':0.3}'")
    args = ap.parse_args()

    weights = json.loads(args.weights) if args.weights else None

    df, results = search(
        query=args.query,
        top_k=args.top_k,
        require_keyword=args.require_keyword,
        weights=weights,
    )

    if not results:
        print("No results found.")
        sys.exit(0)

    print("\n🔎 Top matches:")
    for i, r in enumerate(results, 1):
        print(f"{i}. 🍽️ {r['Restaurant']} | ⭐ {r['Rating']} | ⛳ Score: {r['FinalScore']:.3f}\n   📝 {r['ReviewSnippet']}")


if __name__ == "__main__":
    main()
