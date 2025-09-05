from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Optional
from rag_pipeline import query_pipeline  # ✅ Import your existing pipeline

app = FastAPI(title="Restaurant RAG API")

class RestaurantSummary(BaseModel):
    restaurant: str
    rating: float
    AvgRating: float
    summary: str
    positives: List[str]
    negatives: List[str]
    final_verdict: str

class SearchResponse(BaseModel):
    top_restaurants: List[RestaurantSummary]
    final_suggestion: str


def parse_summary(summary_text: str):
    """
    Extract structured fields from the Gemini summary text.
    """
    positives, negatives, final_verdict = [], [], ""
    lines = summary_text.split("\n")

    section = None
    for line in lines:
        line = line.strip()
        if line.lower().startswith("positives"):
            section = "positives"
            continue
        elif line.lower().startswith("negatives"):
            section = "negatives"
            continue
        elif line.lower().startswith("final verdict"):
            section = "final_verdict"
            continue

        if section == "positives" and line.startswith("*"):
            positives.append(line.lstrip("* ").strip())
        elif section == "negatives" and line.startswith("*"):
            negatives.append(line.lstrip("* ").strip())
        elif section == "final_verdict" and line:
            final_verdict += line + " "

    return positives, negatives, final_verdict.strip()


@app.get("/search", response_model=SearchResponse)
def search(query: str = Query(..., description="User query")):
    """
    Run the RAG pipeline and return structured restaurant summaries + final suggestion.
    """
    top_results, final_suggestion = query_pipeline(query, use_gemini=True)

    formatted = []
    for r in top_results:
        # Parse positives/negatives/final_verdict from Gemini summary
        positives, negatives, final_verdict = parse_summary(r["Summary"])

        formatted.append(
            RestaurantSummary(
                restaurant=r["Restaurant"],
                rating=r["AvgRating"],
                AvgRating=r["AvgRating"],  # keep both
                summary=r["Summary"],
                positives=positives,
                negatives=negatives,
                final_verdict=final_verdict
            )
        )

    return SearchResponse(
        top_restaurants=formatted,
        final_suggestion=final_suggestion or "No suggestion available."
    )


