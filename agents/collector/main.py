import os
from langsmith import Client

# LangSmith tracing automatic (enabled via env)
# Keep migration notes for future reference.
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "mcp-news-orchestrator"

client = Client(api_key=os.getenv("LANGSMITH_API_KEY"))

# MIGRATION_NOTE_AGENT: Review Agent class usage and adapt to LangChain v1 create_agent patterns.
import os

# --- MIGRATION: LangChain v1 helper ---
# This helper tries to build an agent using langgraph.create_react_agent if available,
# otherwise it falls back to langchain.create_agent. Adjust parameters as needed.
try:
    from langgraph import create_react_agent  # preferred for LangGraph-based agents
except Exception:
    create_react_agent = None

try:
    from langchain import create_agent
except Exception:
    create_agent = None

def build_agent(model, tools, memory=None, **kwargs):
    """Return an agent built with the best available factory.
       - model: Chat model instance (e.g., ChatOpenAI)
       - tools: list of Tool objects or callables
       - memory: optional memory object
    """
    if create_react_agent is not None:
        return create_react_agent(model=model, tools=tools, memory=memory, **kwargs)
    if create_agent is not None:
        return create_agent(model=model, tools=tools, memory=memory, **kwargs)
    raise RuntimeError("No suitable agent factory available (langgraph or langchain create_agent)")
# --- end helper ---

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
