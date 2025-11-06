FROM python:3.11-slim

# Copy environment variables from supervisor's .env
ARG LANGCHAIN_TRACING_V2
ARG LANGCHAIN_PROJECT
ARG LANGCHAIN_API_KEY
ARG OPENAI_API_KEY
ARG OPENAI_MODEL

ENV LANGCHAIN_TRACING_V2=${LANGCHAIN_TRACING_V2}
ENV LANGCHAIN_PROJECT=${LANGCHAIN_PROJECT}
ENV LANGCHAIN_API_KEY=${LANGCHAIN_API_KEY}
ENV OPENAI_API_KEY=${OPENAI_API_KEY}
ENV OPENAI_MODEL=${OPENAI_MODEL}

WORKDIR /app
COPY requirements.txt /app/
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt
COPY . /app
EXPOSE 8003
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003"]
