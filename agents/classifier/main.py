import os
import logging
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import LangSmithCallbackHandler
from langsmith import Client

# Setup logging
LOG = logging.getLogger("classifier")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Agent - Classifier")

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
    temperature=0.2,  # Low temperature for more consistent categorization
)

# Define classification prompt
CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a news article classifier. Analyze the article and categorize it into one of these categories:
    - technology: Tech, AI, software, hardware, innovation
    - business: Economy, markets, companies, finance
    - politics: Government, policy, elections
    - science: Research, discoveries, space, medicine
    - other: Everything else
    
    Reply ONLY with the category name in lowercase, nothing else."""),
    ("human", "Title: {title}\nDescription: {description}\n\nClassify this article into one category.")
])

# Create the classification chain
classify_chain = LLMChain(
    llm=llm,
    prompt=CLASSIFY_PROMPT,
    verbose=True
)


class Article(BaseModel):
    title: str | None = None
    description: str | None = None


class ArticlesPayload(BaseModel):
    articles: List[Article]


@app.post("/classify")
async def classify(payload: ArticlesPayload) -> Dict[str, Any]:
    """Classify articles using LangChain + GPT-4"""
    try:
        out = []
        for a in payload.articles:
            # Skip empty articles
            if not a.title and not a.description:
                continue
                
            # Run classification with tracing
            response = classify_chain.invoke(
                {
                    "title": a.title or "",
                    "description": a.description or ""
                },
                config={
                    "callbacks": [langsmith_callback] if langsmith_callback else None,
                    "run_name": f"classify_article_{len(out)+1}"
                }
            )
            
            category = response.get("text", "other").strip().lower()
            
            # Ensure valid category
            if category not in {"technology", "business", "politics", "science", "other"}:
                category = "other"
                
            out.append({
                "title": a.title,
                "description": a.description,
                "category": category
            })
            
        return {"ok": True, "articles": out}
        
    except Exception as e:
        LOG.exception("Classification failed")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8002)
