import numpy as np

def rerank_candidates(query, candidates):
    """
    Rerank by rating, number of reviews, and semantic score (if available).
    """
    for c in candidates:
        rating = c.get("rating", 0) or 0
        review_count = c.get("review_count", 0) or 0
        semantic = c.get("semantic_score", 0) or 0

        c["rank_score"] = (
            (rating * 0.6) +
            (np.log1p(review_count) * 0.3) +
            (semantic * 0.1)
        )

    return sorted(candidates, key=lambda x: x["rank_score"], reverse=True)
