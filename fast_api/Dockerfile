FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download FastEmbed model (so container doesn't need internet at runtime)
RUN python -c "from fastembed import TextEmbedding; \
    model = TextEmbedding('sentence-transformers/all-MiniLM-L6-v2'); \
    list(model.passage_embed(['warmup']))"

COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8800"]
