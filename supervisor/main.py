import os
import asyncio
import logging
from typing import Any

import httpx
import yaml
from fastapi import FastAPI, HTTPException

try:
    from langsmith import Client
    _HAS_LANGSMITH = True
except Exception:
    _HAS_LANGSMITH = False

LOG = logging.getLogger("supervisor")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Supervisor - News Curation")

# Load graph.yaml if present
GRAPH_PATH = os.path.join(os.path.dirname(__file__), "graph.yaml")
if os.path.exists(GRAPH_PATH):
    with open(GRAPH_PATH, "r", encoding="utf-8") as f:
        GRAPH = yaml.safe_load(f)
else:
    GRAPH = {}

# LangSmith client (optional)
LANGSMITH_CLIENT = None
if _HAS_LANGSMITH and os.getenv("LANGCHAIN_API_KEY"):
    try:
        LANGSMITH_CLIENT = Client()
        LOG.info("LangSmith client initialized")
    except Exception as e:
        LOG.warning("LangSmith client failed to initialize: %s", e)


async def call_service(url: str, method: str = "GET", json: Any = None, params: dict | None = None):
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            if method == "GET":
                r = await client.get(url, params=params)
            else:
                r = await client.post(url, json=json)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            LOG.exception("Error calling %s: %s", url, e)
            raise


@app.get("/run")
async def run(q: str = "technology"):
    """Run a simple flow: collector -> classifier -> analyst"""
    collector_url = os.getenv("COLLECTOR_URL", "http://collector:8001/fetch")
    classifier_url = os.getenv("CLASSIFIER_URL", "http://classifier:8002/classify")
    analyst_url = os.getenv("ANALYST_URL", "http://analyst:8003/analyze")

    try:
        LOG.info("Calling collector: %s?q=%s", collector_url, q)
        collector_resp = await call_service(collector_url, method="GET", params={"q": q})

        articles = collector_resp.get("articles") if isinstance(collector_resp, dict) else collector_resp
        if not articles:
            return {"ok": False, "reason": "no articles returned by collector", "data": []}

        # Classify
        LOG.info("Calling classifier with %d articles", len(articles))
        class_resp = await call_service(classifier_url, method="POST", json={"articles": articles})

        # Analyze
        LOG.info("Calling analyst")
        analyst_resp = await call_service(analyst_url, method="POST", json={"articles": class_resp.get("articles", [])})

        consolidated = {
            "collected": len(articles),
            "classified": class_resp,
            "analyzed": analyst_resp,
        }

        # Optionally log tracing to LangSmith (best-effort)
        if LANGSMITH_CLIENT:
            try:
                LANGSMITH_CLIENT.create_run(name="news_curation_run", metadata={"q": q})
            except Exception:
                LOG.debug("LangSmith run creation failed, continuing")

        return {"ok": True, "result": consolidated}

    except Exception as exc:
        LOG.exception("Run failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8004, log_level="info")
