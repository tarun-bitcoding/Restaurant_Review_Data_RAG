import os
import faiss
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_FILE = "data/all_reviews.parquet"
INDEX_DIR = "indices"

def build_index():
    os.makedirs(INDEX_DIR, exist_ok=True)

    print("🔹 Loading dataset...")
    df = pd.read_parquet(DATA_FILE)
    print(f"Dataset has {len(df)} rows")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("🔹 Encoding text...")
    embeddings = model.encode(
        df["text_blob"].tolist(),
        batch_size=64,
        show_progress_bar=True
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))

    index_file = os.path.join(INDEX_DIR, "global_reviews.faiss")
    meta_file = os.path.join(INDEX_DIR, "global_reviews.pkl")

    faiss.write_index(index, index_file)
    with open(meta_file, "wb") as f:
        pickle.dump(df.to_dict(orient="records"), f)

    print(f"✅ Saved global index: {index_file}")
    print(f"✅ Saved metadata: {meta_file}")

if __name__ == "__main__":
    build_index()
