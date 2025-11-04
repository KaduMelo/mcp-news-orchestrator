import logging
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

LOG = logging.getLogger("analyst")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Agent - Analyst")


class ArticleIn(BaseModel):
    title: str | None = None
    description: str | None = None
    category: str | None = None


class ArticlesIn(BaseModel):
    articles: List[ArticleIn]


@app.post("/analyze")
async def analyze(payload: ArticlesIn) -> Dict[str, Any]:
    """Do simple sentiment/relevance analysis (heuristic)"""
    out = []
    for a in payload.articles:
        text = (a.title or "") + " " + (a.description or "")
        score = 0
        pos_words = ["good", "great", "innov", "grow", "positive", "success"]
        neg_words = ["crisis", "drop", "bad", "problem", "loss"]
        t = text.lower()
        for w in pos_words:
            if w in t:
                score += 1
        for w in neg_words:
            if w in t:
                score -= 1
        sentiment = "neutral"
        if score > 0:
            sentiment = "positive"
        elif score < 0:
            sentiment = "negative"
        out.append({"title": a.title, "category": a.category, "sentiment": sentiment, "score": score})

    return {"ok": True, "articles": out}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8003)
