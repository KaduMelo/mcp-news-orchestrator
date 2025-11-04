import logging
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

LOG = logging.getLogger("classifier")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Agent - Classifier")


class Article(BaseModel):
    title: str | None = None
    description: str | None = None


class ArticlesPayload(BaseModel):
    articles: List[Article]


@app.post("/classify")
async def classify(payload: ArticlesPayload) -> Dict[str, Any]:
    """Simple rule-based classifier as placeholder for an LLM-driven node."""
    out = []
    for a in payload.articles:
        text = (a.title or "") + " " + (a.description or "")
        txt = text.lower()
        if "tech" in txt or "ai" in txt or "intellig" in txt:
            cat = "technology"
        elif "econom" in txt or "business" in txt or "market" in txt:
            cat = "economy"
        else:
            cat = "other"
        out.append({"title": a.title, "description": a.description, "category": cat})

    return {"ok": True, "articles": out}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8002)
