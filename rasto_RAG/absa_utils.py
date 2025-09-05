import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("punkt", quiet=True)
nltk.download("vader_lexicon", quiet=True)
sia = SentimentIntensityAnalyzer()

ASPECT_KEYWORDS = {
    "food": ["food","taste","flavour","flavor","maggi","noodles","dish","meal","delicious","tasty"],
    "price": ["price","cost","expensive","cheap","value","money","costly","affordable"],
    "service": ["service","waiter","staff","server","friendly","rude","attendant","manager"],
    "ambience": ["ambience","ambiance","atmosphere","clean","crowd","decor","noise","space"]
}

def split_sentences(text):
    return nltk.sent_tokenize(text or "")

def get_aspect_sentiments(text, aspects=ASPECT_KEYWORDS):
    """
    Returns dict: {aspect: score_in_0_1_or_None}
    If no sentence mentions an aspect, returns None for that aspect.
    """
    out = {}
    sents = split_sentences(text)
    for a, kws in aspects.items():
        hits = []
        for s in sents:
            lower = s.lower()
            if any(k in lower for k in kws):
                comp = sia.polarity_scores(s)["compound"]  # -1..1
                hits.append((comp + 1)/2)  # map to 0..1
        out[a] = (sum(hits)/len(hits)) if hits else None
    return out

def detect_query_aspects(query, aspects=ASPECT_KEYWORDS):
    q = (query or "").lower()
    found = []
    for a, kws in aspects.items():
        if any(k in q for k in kws):
            found.append(a)
    return found 
