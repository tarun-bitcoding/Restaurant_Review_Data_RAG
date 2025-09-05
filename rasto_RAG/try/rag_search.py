import os
import pickle
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Paths
INDEX_FILE = "indices/global_reviews.faiss"
META_FILE = "indices/global_reviews.pkl"

# Load sentiment analyzer
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

def load_resources():
    index = faiss.read_index(INDEX_FILE)
    with open(META_FILE, "rb") as f:
        metadata = pickle.load(f)
    df = pd.DataFrame(metadata)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return index, df, model

def sentiment_score(text):
    score = sia.polarity_scores(text)
    return (score["compound"] + 1) / 2  # map -1..1 → 0..1

def search(query, top_k=5, alpha=0.6, beta=0.2, gamma=0.2):
    """
    alpha = weight for semantic similarity
    beta = weight for rating
    gamma = weight for sentiment positivity
    Ensures one restaurant appears only once (best-scored review kept).
    """

    index, df, model = load_resources()

    # Encode query
    q_vec = model.encode([query]).astype("float32")
    D, I = index.search(q_vec, top_k * 10)  # retrieve more candidates

    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        row = df.iloc[idx]

        sim_score = 1 / (1 + dist)
        rating_score = row["Rating"] / 5.0
        sent_score = sentiment_score(str(row["Review"]))

        final_score = alpha * sim_score + beta * rating_score + gamma * sent_score

        results.append({
            "Restaurant": row["Restaurant"],
            "Review": row["Review"],
            "Rating": row["Rating"],
            "FinalScore": round(final_score, 3)
        })

    # --- Deduplicate by restaurant ---
    unique_restaurants = {}
    for r in results:
        rest = r["Restaurant"]
        # keep only the best review for each restaurant
        if rest not in unique_restaurants or r["FinalScore"] > unique_restaurants[rest]["FinalScore"]:
            unique_restaurants[rest] = r

    # Convert dict → list and sort
    final_results = sorted(unique_restaurants.values(), key=lambda x: x["FinalScore"], reverse=True)

    return final_results[:top_k]


if __name__ == "__main__":
    top_k = 5  
    while True:
        query = input("\n❓Enter your question (or type 'exit' to quit): ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print("👋 Exiting search. Goodbye..tata!")
            break

        results = search(query, top_k=top_k)

        print("\n🔎 Top Results:")
        for r in results:
            print(f"🍽️ {r['Restaurant']} | ⭐ {r['Rating']} | 📝 {r['Review'][:80]}... | 🔗 {r['FinalScore']}")


