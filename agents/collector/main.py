import os
import logging
from typing import Any

import httpx
from fastapi import FastAPI

LOG = logging.getLogger("collector")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Agent - Collector")

NEWSAPI = os.getenv("NEWSAPI_KEY")


@app.get("/fetch")
async def fetch(q: str = "technology") -> Any:
    """Fetch news from NewsAPI.org (if NEWSAPI_KEY provided) or return sample data."""
    if NEWSAPI:
        url = "https://newsapi.org/v2/everything"
        params = {"q": q, "language": "en", "pageSize": 5, "apiKey": NEWSAPI}
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url, params=params)
            r.raise_for_status()
            return r.json()

    # Fallback sample
    LOG.info("No NEWSAPI_KEY set — returning sample articles")
    sample = {
        "articles": [
            {"title": "Tech innovation: new AI", "description": "A new AI has been released."},
            {"title": "Local economy grows", "description": "Positive signals in business."},
        ]
    }
    return sample


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8001)
