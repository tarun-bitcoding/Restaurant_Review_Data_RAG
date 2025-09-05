import os
import math
import pickle
import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from reranker import rerank_candidates
import google.generativeai as genai
from dotenv import load_dotenv

INDEX_DIR = "indices"

# Load embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load Gemini API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel("gemini-1.5-flash")

# Load indices + metadata
indices, metas, cities = {}, {}, []
for file in os.listdir(INDEX_DIR):
    if file.endswith(".faiss"):
        city = file.replace(".faiss", "")
        faiss_file = os.path.join(INDEX_DIR, file)
        meta_file = os.path.join(INDEX_DIR, f"{city}.pkl")

        indices[city] = faiss.read_index(faiss_file)
        with open(meta_file, "rb") as f:
            metas[city] = pickle.load(f)

        cities.append(city)
        print(f"✅ Loaded {city} index with {len(metas[city])} restaurants")

# Precompute city embeddings
city_embeddings = embedding_model.encode(cities)

# FastAPI app
app = FastAPI()

class AskRequest(BaseModel):
    query: str
    top_k: int = 5

def detect_city(query: str):
    """Find closest city for the query"""
    query_emb = embedding_model.encode([query])
    sims = np.dot(city_embeddings, query_emb[0]) / (
        np.linalg.norm(city_embeddings, axis=1) * np.linalg.norm(query_emb)
    )
    best_idx = np.argmax(sims)
    return cities[best_idx], sims[best_idx]

def query_llm(query: str, results: list):
    context = "\n".join([
        f"- {r.get('restaurant_name')} | ⭐ {r.get('rating')} | {r.get('review')[:100]}..."
        for r in results
    ])
    prompt = f"""
You are a food guide assistant.
The user asked: "{query}"

Here are the top restaurants retrieved:
{context}

Based on reviews and ratings, suggest the BEST restaurant in natural language (2-3 sentences).
"""
    response = llm.generate_content(prompt)
    return response.text.strip() if response and response.text else "No recommendation available."

@app.post("/ask")
def ask(req: AskRequest):
    query = req.query
    top_k = req.top_k  

    # Detect city
    detected_city, score = detect_city(query)
    print(f"🔎 Detected city: {detected_city} (score={score:.3f})")

    results = []
    query_emb = embedding_model.encode([query])

    # Vector search in that city
    index = indices[detected_city]
    meta = metas[detected_city]

    D, I = index.search(np.array(query_emb).astype("float32"), min(200, len(meta)))
    candidates = [meta[i] for i in I[0]]

    # Rerank
    ranked = rerank_candidates(query, candidates)
    results = ranked[:top_k]

    # Clean dict
    def clean_dict(d):
        return {k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v) for k, v in d.items()}

    cleaned_results = [clean_dict(r) for r in results]

    best_answer = query_llm(query, cleaned_results)

    return {
        "query": query,
        "detected_city": detected_city,
        "best_answer": best_answer,
        "results": cleaned_results
    }

#3 uvicorn search_service:app --reload --port 8000
