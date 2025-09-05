import pandas as pd
import os

DATA_FILE = "data/GoogleReview_data_cleaned.csv"
SHARD_DIR = "shards"

def preprocess():
    os.makedirs(SHARD_DIR, exist_ok=True)

    print("🔹 Loading dataset...")
    df = pd.read_csv(DATA_FILE)

    # Clean rating
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").fillna(0)

    # Create text blob (restaurant info + review)
    df["text_blob"] = (
        df["Restaurant"].fillna("") + " "
        + df["Review"].fillna("")
    )

    # Partition by city
    for Location, group in df.groupby("Location"):
        shard_path = os.path.join(SHARD_DIR, f"{Location.replace(' ', '_')}.parquet")
        group.to_parquet(shard_path, index=False)
        print(f"✅ Saved shard: {shard_path} ({len(group)} rows)")

if __name__ == "__main__":
    preprocess()

#1 python preprocess.py