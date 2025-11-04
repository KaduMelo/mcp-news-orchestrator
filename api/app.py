import os
import logging

import httpx
from fastapi import FastAPI, HTTPException

LOG = logging.getLogger("api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="News Curation API Gateway")


@app.get("/curate")
async def curate(q: str = "technology"):
    supervisor_url = os.getenv("SUPERVISOR_URL", "http://supervisor:8004/run")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(supervisor_url, params={"q": q})
            r.raise_for_status()
            return r.json()
    except Exception as e:
        LOG.exception("Error calling supervisor: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000)
