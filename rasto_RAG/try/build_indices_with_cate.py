import os
import pickle
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

SHARD_DIR = "shards"
INDEX_DIR = "indices"

def build_indices():
    os.makedirs(INDEX_DIR, exist_ok=True)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    for shard_file in os.listdir(SHARD_DIR):
        if not shard_file.endswith(".parquet"):
            continue

        shard_path = os.path.join(SHARD_DIR, shard_file)
        df = pd.read_parquet(shard_path)

        print(f"🔹 Processing {shard_file} with {len(df)} rows")

        # Embeddings
        embeddings = model.encode(df["text_blob"].tolist(), batch_size=64, show_progress_bar=True)

        # Build FAISS
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype("float32"))

        # Save index + metadata
        base_name = shard_file.replace(".parquet", "")
        index_file = os.path.join(INDEX_DIR, f"{base_name}.faiss")
        meta_file = os.path.join(INDEX_DIR, f"{base_name}.pkl")

        faiss.write_index(index, index_file)
        with open(meta_file, "wb") as f:
            pickle.dump(df.to_dict(orient="records"), f)

        print(f"✅ Saved index: {index_file}, metadata: {meta_file}")

if __name__ == "__main__":
    build_indices()

#2 python build_indices.py