"""
RAG search and summarization over restaurant reviews using hybrid retrieval and aspect-based sentiment analysis.

Pipeline:
1) Retrieve candidates with hybrid search:
   - Semantic similarity via FAISS embeddings (review/text_blob).
   - Lexical keyword relevance via BM25.
   - Combined scoring ensures both semantic match and exact keyword hits.
2) Apply Aspect-Based Sentiment Analysis (ABSA):
   - Extract sentiment across dimensions (Food, Service, Price, Ambience).
   - Build an aspect sentiment profile for each review.
3) Aggregate top-10 reviews per restaurant:
   - Select most relevant reviews per candidate restaurant.
   - Preserve diversity across aspects (avoid duplicates).
4) Summarization with Gemini LLM:
   - Prompt Gemini with the top reviews for each restaurant.
   - Generate concise summaries highlighting strengths and weaknesses.
5) Ranking & Final Suggestion:
   - Normalize and combine scores (semantic similarity + BM25 relevance + ABSA sentiment + rating).
   - Rank restaurants by final score.
   - Return top-5 summarized restaurants.
   - Send results to Gemini again for a final “best choice” suggestion based on user's query context.

Files expected:
  - indices/global_reviews.faiss   (semantic index)
  - indices/global_reviews.pkl     (list[dict], includes: Restaurant, Review, Rating, …)

Usage:
  python rag_pipeline.py 

Install (typical):
  pip install sentence-transformers faiss-cpu rank-bm25 pandas numpy nltk google-generativeai

Notes:
- ABSA can be swapped with pretrained HuggingFace models for richer aspect extraction.
- Gemini is used both for review summarization and for the final best-restaurant recommendation.
"""

import nltk
import faiss
import numpy as np
import pandas as pd
import os, pickle, re
from dotenv import load_dotenv
from transformers import pipeline
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from absa_utils import get_aspect_sentiments, detect_query_aspects 

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Paths vector file
FAISS_INDEX = "indices/global_reviews.faiss"
META_FILE = "indices/global_reviews.pkl"
BM25_META = "indices/bm25_meta.pkl"

# Models parameter
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOP_FAISS = 200
TOP_BM25 = 200
TOP_REVIEWS_PER_RESTAURANT = 10
TOP_RESTAURANTS = 5

# Scoring weights
ALPHA = 0.45  # semantic
BETA = 0.25   # BM25
GAMMA = 0.20  # rating
DELTA = 0.10  # aspect sentiment


# Ensure punkt models are available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


# Load resources
def load_resources():
    assert os.path.exists(FAISS_INDEX), "Missing FAISS index"
    assert os.path.exists(META_FILE), "Missing metadata pkl"
    assert os.path.exists(BM25_META), "Missing bm25 metadata"

    index = faiss.read_index(FAISS_INDEX)
    with open(META_FILE, "rb") as f:
        meta = pickle.load(f)
    df = pd.DataFrame(meta)

    with open(BM25_META, "rb") as f:
        bm25_blob = pickle.load(f)
    bm25 = bm25_blob["bm25"]
    bm25_tokenized = bm25_blob["tokenized"]

    emb = SentenceTransformer(EMB_MODEL)
    return index, df, bm25, bm25_tokenized, emb

def normalize01(arr):
    a = np.array(arr, dtype=float)
    mn, mx = np.min(a), np.max(a)
    if mx - mn < 1e-12:
        return np.ones_like(a) * 0.5
    return (a - mn) / (mx - mn)

def faiss_dist_to_sim(dists):
    return 1.0 / (1.0 + np.array(dists))

# Local summarizer
def load_local_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")

def chunk_reviews(reviews, chunk_size=20):
    """Split reviews into smaller chunks."""
    for i in range(0, len(reviews), chunk_size):
        yield reviews[i:i+chunk_size]


def summarize_with_gemini(restaurant_name, reviews, chunk_size=20):
    """Summarize all reviews for a restaurant using Gemini with chunking."""
    model = genai.GenerativeModel("gemini-2.0-flash-lite")

    # Step 1: Summarize each chunk
    partial_summaries = []
    for chunk in chunk_reviews(reviews, chunk_size):
        joined = "\n".join(chunk)
        prompt = f"""
        Summarize the following {len(chunk)} customer reviews for "{restaurant_name}".
        Capture both positives and negatives clearly.

        Reviews:
        {joined}
        """
        resp = model.generate_content(prompt)
        partial_summaries.append(resp.text.strip())

    # Step 2: Merge all partial summaries into a final structured summary
    merge_prompt = f"""
    You are summarizing customer reviews for "{restaurant_name}".
    Here are partial summaries of different review sets:

    {'\n---\n'.join(partial_summaries)}

    Please combine these into ONE structured summary:
    1. Overall summary (3–4 sentences)
    2. Positives (bullet list)
    3. Negatives (bullet list)
    4. Final verdict (short recommendation)
    """
    resp = model.generate_content(merge_prompt)
    return resp.text.strip()


def summarize_locally(restaurant_name, reviews, summarizer, chunk_size=20):
    """Summarize all reviews for a restaurant using local summarizer with chunking."""
    # Step 1: Summarize chunks
    partial_summaries = []
    for chunk in chunk_reviews(reviews, chunk_size):
        joined = "\n".join(chunk)
        summary = summarizer(
            joined,
            max_length=200,
            min_length=60,
            do_sample=False
        )[0]["summary_text"]
        partial_summaries.append(summary)

    # Step 2: Merge summaries into a final summary
    final_text = " ".join(partial_summaries)
    merged = summarizer(
        final_text,
        max_length=300,
        min_length=100,
        do_sample=False
    )[0]["summary_text"]

    return merged


def query_pipeline(query, use_gemini=True):
    index, df, bm25, bm25_tokenized, emb = load_resources()

    # 1. Aspect detection
    query_aspects = detect_query_aspects(query)
    q_tokens = re.findall(r"\w+", query.lower())

    # 2. FAISS retrieval
    qv = emb.encode([query]).astype("float32")
    k = min(TOP_FAISS, len(df))
    D, I = index.search(qv, k)
    faiss_inds, faiss_dists = I[0].tolist(), D[0].tolist()
    faiss_map = {idx: 1 / (1 + dist) for idx, dist in zip(faiss_inds, faiss_dists) if idx != -1}

    # 3. BM25 retrieval
    bm25_scores = bm25.get_scores(q_tokens)
    bm25_top_idx = np.argsort(bm25_scores)[::-1][:TOP_BM25]
    bm25_map = {int(i): float(bm25_scores[int(i)]) for i in bm25_top_idx if bm25_scores[int(i)] > 0}

    # 4. Candidate union
    candidate_ids = set(list(faiss_map.keys()) + list(bm25_map.keys()))
    if not candidate_ids:
        return [], "No relevant restaurants found."

    cand_list = sorted(list(candidate_ids))
    sem_scores = np.array([faiss_map.get(i, 0.0) for i in cand_list])
    bm25_raw = np.array([bm25_map.get(i, 0.0) for i in cand_list])
    ratings = np.array([float(df.iloc[i].get("Rating", 0) or 0) for i in cand_list]) / 5.0

    sem_n = normalize01(sem_scores)
    bm25_n = normalize01(bm25_raw)
    rating_n = normalize01(ratings)

    aspect_scores = []
    for i in cand_list:
        review = str(df.iloc[i].get("Review", ""))
        aspects = get_aspect_sentiments(review)
        if query_aspects:
            vals = [aspects.get(a) for a in query_aspects if aspects.get(a) is not None]
            val = sum(vals) / len(vals) if vals else 0.5
        else:
            vals = [v for v in aspects.values() if v is not None]
            val = sum(vals) / len(vals) if vals else 0.5
        aspect_scores.append(val)
    aspect_n = normalize01(np.array(aspect_scores))

    # Hybrid score
    hybrid = ALPHA * sem_n + BETA * bm25_n + GAMMA * rating_n + DELTA * aspect_n

    candidates = []
    for idx, score in zip(cand_list, hybrid):
        candidates.append({
            "idx": idx,
            "Restaurant": df.iloc[idx].get("Restaurant"),
            "Review": df.iloc[idx].get("Review"),
            "Rating": float(df.iloc[idx].get("Rating") or 0),
            "score": float(score)
        })


    # ---------------------------
    # Filter reviews inside each matched restaurant by query relevance
    # ---------------------------
    matched_restaurants = set(df.loc[list(candidate_ids), "Restaurant"].astype(str).tolist())
    if not matched_restaurants:
        return [], "No relevant restaurants found."

    filtered_candidates = []
    for ridx in df.index[df["Restaurant"].isin(matched_restaurants)]:
        review_text = str(df.iloc[ridx].get("Review", ""))
        rating_val = float(df.iloc[ridx].get("Rating") or 0)

        # raw scores (0 if not retrieved)
        sem_raw = faiss_map.get(ridx, 0.0)
        bm25_raw = bm25_map.get(ridx, 0.0)

        # check aspect relevance
        aspects = get_aspect_sentiments(review_text)
        aspect_val = None
        if query_aspects:
            vals = [aspects.get(a) for a in query_aspects if aspects.get(a) is not None]
            aspect_val = sum(vals) / len(vals) if vals else None

        # relevance decision:
        # keep review if semantic or lexical score > 0
        # OR it contains relevant aspects
        if sem_raw > 0 or bm25_raw > 0 or aspect_val is not None:
            filtered_candidates.append({
                "idx": ridx,
                "Restaurant": df.iloc[ridx].get("Restaurant"),
                "Review": review_text,
                "Rating": rating_val,
                "sem": sem_raw,
                "bm25": bm25_raw,
                "aspect": aspect_val if aspect_val is not None else 0.5
            })

    if not filtered_candidates:
        return [], "No relevant reviews matched user query."

    # Normalize and score filtered reviews
    sem_n = normalize01([c["sem"] for c in filtered_candidates])
    bm25_n = normalize01([c["bm25"] for c in filtered_candidates])
    rating_n = normalize01([c["Rating"]/5.0 for c in filtered_candidates])
    aspect_n = normalize01([c["aspect"] for c in filtered_candidates])

    for i, c in enumerate(filtered_candidates):
        c["score"] = (
            ALPHA * sem_n[i]
            + BETA * bm25_n[i]
            + GAMMA * rating_n[i]
            + DELTA * aspect_n[i]
        )

    df_cand = pd.DataFrame(filtered_candidates)
    rep = df_cand.sort_values(["Restaurant", "score"], ascending=[True, False])

    # group by restaurant, keep only reviews relevant to query
    agg = (
        rep.groupby("Restaurant")
           .agg(
               RestaurantScore=("score", "mean"),
               AvgRating=("Rating", "mean"),
               Reviews=("Review", lambda x: list(x))
           )
           .reset_index()
    )

    agg = agg.sort_values("RestaurantScore", ascending=False).reset_index(drop=True)
    topk = agg.head(TOP_RESTAURANTS)


    # Summaries
    results = []
    if use_gemini and GEMINI_API_KEY:
        for _, row in topk.iterrows():
            summary = summarize_with_gemini(row["Restaurant"], row["Reviews"], chunk_size=20)
            results.append({
                "Restaurant": row["Restaurant"],
                "RestaurantScore": row["RestaurantScore"],
                "AvgRating": row["AvgRating"],
                "Summary": summary
            })
    else:
        summarizer = load_local_summarizer()
        for _, row in topk.iterrows():
            summary = summarize_locally(row["Restaurant"], row["Reviews"], summarizer, chunk_size=20)
            results.append({
                "Restaurant": row["Restaurant"],
                "RestaurantScore": row["RestaurantScore"],
                "AvgRating": row["AvgRating"],
                "Summary": summary
            })

    # Final suggestion from Gemini
    final_suggestion = None
    if use_gemini and GEMINI_API_KEY:
        combined_prompt = f"User query: {query}\n\nTop restaurant summaries:\n"
        for r in results:
            combined_prompt += f"--- {r['Restaurant']} (score={r['RestaurantScore']:.3f}, rating={r['AvgRating']}) ---\n{r['Summary']}\n\n"
        combined_prompt += "Based only on the above summaries, recommend the best restaurant for the user and explain in 2-3 sentences."
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(combined_prompt)
        final_suggestion = resp.text.strip()

    return results, final_suggestion


if __name__ == "__main__":
    while True:
        query = input("\n❓ Enter your question (or type 'exit' to quit): ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print("👋 Exiting RAG pipeline. Goodbye!")
            break

        top_results, final_suggestion = query_pipeline(query, use_gemini=True)

        print("\n🔎 Top Restaurant Summaries:")
        for r in top_results:
            print(f"\n🍽️ {r['Restaurant']} (⭐ {r['AvgRating']})\n📝 {r['Summary']}")

        print("\n🤖 Final Suggestion:")
        print(final_suggestion or "No suggestion available.")

