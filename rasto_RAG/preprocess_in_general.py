import pandas as pd

DATA_FILE = "data/GoogleReview_data_cleaned.csv"
OUTPUT_FILE = "data/all_reviews.parquet"

def preprocess():
    print("🔹 Loading dataset...")
    df = pd.read_csv(DATA_FILE)

    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce").fillna(0)

    df["text_blob"] = (
        df["Restaurant"].fillna("") + " "
        + df["Review"].fillna("")
    )
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"✅ Saved unified dataset: {OUTPUT_FILE} ({len(df)} rows)")

if __name__ == "__main__":
    preprocess()
