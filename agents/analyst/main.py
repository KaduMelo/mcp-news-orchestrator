# MIGRATION_NOTE_AGENT: Review Agent class usage and adapt to LangChain v1 create_agent patterns.
# TODO: initialize LangSmith client for tracing
from langsmith.client import Clientimport os
import logging
from typing import List, Dict, Any

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


# --- MIGRATION: LangSmith tracing helper (TODO: set LANGSMITH_API_KEY in env) ---
# from langsmith.client import Client
# client = Client(api_key=os.environ.get('LANGSMITH_API_KEY'))
# Use client.create_run(...) and run.end(...) around agent executions to trace runs.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.callbacks import LangSmithCallbackHandler
from langsmith import Client

LOG = logging.getLogger("analyst")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Agent - Analyst")

# Initialize LangSmith (if configured)
try:
    if os.getenv("LANGCHAIN_API_KEY"):
        langsmith_client = Client()
        langsmith_callback = LangSmithCallbackHandler(
            project_name=os.getenv("LANGCHAIN_PROJECT", "NewsCuration")
        )
        LOG.info("LangSmith tracing enabled")
    else:
        langsmith_callback = None
        LOG.info("No LANGCHAIN_API_KEY set - running without tracing")
except Exception as e:
    LOG.warning("Failed to initialize LangSmith: %s", e)
    langsmith_callback = None

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4-1106-preview",  # or gpt-3.5-turbo for lower cost
    temperature=0.1,  # Low temperature for consistent analysis
)

# Define output schema
response_schemas = [
    ResponseSchema(
        name="sentiment",
        description="The sentiment of the article (positive, negative, or neutral)"
    ),
    ResponseSchema(
        name="score",
        description="Sentiment score from -5 (very negative) to +5 (very positive)"
    ),
    ResponseSchema(
        name="relevance",
        description="How relevant/impactful this news is (high, medium, or low)"
    ),
    ResponseSchema(
        name="reasoning",
        description="Brief explanation of the analysis"
    )
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Define analysis prompt
ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert news analyst. Analyze the article's sentiment, impact, and relevance.
    
    {format_instructions}
    
    Consider:
    - Overall tone and emotional impact
    - Significance of the news
    - Potential implications
    - Factual vs emotional content
    
    Be consistent in scoring and categorization."""),
    ("human", """Title: {title}
    Category: {category}
    Description: {description}
    
    Analyze this article's sentiment and relevance.""")
])

# Create the analysis chain
analysis_chain = LLMChain(
    llm=llm,
    prompt=ANALYSIS_PROMPT,
    verbose=True
)


class ArticleIn(BaseModel):
    title: str | None = None
    description: str | None = None
    category: str | None = None


class ArticlesIn(BaseModel):
    articles: List[ArticleIn]


@app.post("/analyze")
async def analyze(payload: ArticlesIn) -> Dict[str, Any]:
    """Analyze articles using LangChain + GPT-4"""
    try:
        out = []
        for idx, a in enumerate(payload.articles):
            # Skip empty articles
            if not a.title and not a.description:
                continue

            # Run analysis with tracing
            response = analysis_chain.invoke(
                {
                    "title": a.title or "",
                    "description": a.description or "",
                    "category": a.category or "unknown",
                    "format_instructions": output_parser.get_format_instructions()
                },
                config={
                    "callbacks": [langsmith_callback] if langsmith_callback else None,
                    "run_name": f"analyze_article_{idx+1}"
                }
            )

            try:
                parsed = output_parser.parse(response["text"])
                # Ensure score is numeric
                try:
                    score = float(parsed["score"])
                except (ValueError, TypeError):
                    score = 0
                
                out.append({
                    "title": a.title,
                    "category": a.category,
                    "sentiment": parsed["sentiment"].lower(),
                    "score": score,
                    "relevance": parsed["relevance"].lower(),
                    "reasoning": parsed["reasoning"]
                })
            except Exception as parse_err:
                LOG.error("Failed to parse LLM response: %s", parse_err)
                # Fallback to simple result
                out.append({
                    "title": a.title,
                    "category": a.category,
                    "sentiment": "neutral",
                    "score": 0,
                    "relevance": "medium",
                    "reasoning": "Failed to parse analysis"
                })

        return {"ok": True, "articles": out}

    except Exception as e:
        LOG.exception("Analysis failed")
        raise HTTPException(status_code=500, detail=str(e))

    return {"ok": True, "articles": out}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8003)
