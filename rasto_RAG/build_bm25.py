import utils.transport
import utils.session
import os
import re
import pickle
import pandas as pd
from rank_bm25 import BM25Okapi

DATA_PARQUET = "data/all_reviews.parquet"
BM25_META = "indices/bm25_meta.pkl"
os.makedirs("indices", exist_ok=True)

def tokenize(text):
    return re.findall(r"\w+", (text or "").lower())

def build_bm25():
    df = pd.read_parquet(DATA_PARQUET)
    docs = df["Review"].fillna("").astype(str).tolist()
    tokenized = [tokenize(d) for d in docs]
    bm25 = BM25Okapi(tokenized)
    meta = {
        "tokenized": tokenized,
        "docs": docs,
        "n_docs": len(docs)
    }
    with open(BM25_META, "wb") as f:
        pickle.dump({"bm25": bm25, **meta}, f)
    print(f"Saved BM25 meta -> {BM25_META}, docs={len(docs)}")

if __name__ == "__main__":
    build_bm25()
