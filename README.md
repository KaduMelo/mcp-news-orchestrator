# News Curation — POC

Proof-of-concept architecture with autonomous agents (Collector, Classifier, Analyst) orchestrated by a Supervisor and observability through LangSmith.

Prerequisites
- Docker & Docker Compose
- Python 3.11 (for local builds if needed)

Environment variables (create a `.env` file in this folder or fill `supervisor/.env`):

```env
NEWSAPI_KEY=your_newsapi_key
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=NewsCuration
```

How to run

1. From this folder (where `docker-compose.yml` is located):

```bash
docker compose up --build
```

2. Example endpoints

- API Gateway: http://localhost:8000/curate?q=technology
- Collector: http://localhost:8001/fetch?q=technology

Observability
- Set `LANGCHAIN_API_KEY` with your LangSmith key. The supervisor will attempt to create LangSmith runs if the `langsmith` package and key are available.

Next steps
- Replace heuristic classifier/analyst with real LLM nodes (LangChain / LangGraph).
- Add LangSmith tracing for each LLM call.
- Harden communication with Redis Streams (MCP) and add retries.
