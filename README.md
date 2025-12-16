# Qdrant Server with FastAPI

A FastAPI application that provides text embedding, vector storage, and search capabilities using Qdrant vector database. It includes PDF parsing functionality to convert PDF documents to Markdown.

## Features

- Text embedding using FastEmbed with sentence-transformers/all-MiniLM-L6-v2 model
- Vector storage and similarity search with Qdrant
- PDF to Markdown conversion
- RESTful API with authentication
- Docker containerization

## Prerequisites

- Docker and Docker Compose
- Python 3.11 (if running locally)

## Quick Start

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd qdrant-server
   ```

2. Start the services using Docker Compose:
   ```bash
   docker-compose up --build
   ```

3. The API will be available at `http://localhost:8800`
   - Qdrant dashboard at `http://localhost:3346/dashboard`

## API Endpoints

All endpoints require an API key in the `X-API-Key` header.

### POST /embed
Generate embeddings for a single text.

**Request Body:**
```json
{
  "collection": "my_collection",
  "text": "Your text here"
}
```

### POST /import
Embed and store multiple texts in a collection.

**Request Body:**
```json
{
  "collection": "my_collection",
  "texts": ["text1", "text2", "text3"]
}
```

### POST /search
Search for similar texts in a collection.

**Request Body:**
```json
{
  "collection": "my_collection",
  "query": "search query",
  "top_k": 5
}
```

### DELETE /collection
Delete an entire collection.

**Request Body:**
```json
{
  "collection": "my_collection"
}
```

### POST /parse-pdf
Parse a PDF file and convert it to Markdown.

**Request:** Multipart form with a PDF file.

## Environment Variables

- `API_KEY`: API key for authentication (default: b59cf76c-9e31-4520-bfb9-86a5d01b02cc)
- `QDRANT_URL`: Qdrant server URL (default: http://qdrant:6333)

## Local Development

1. Install dependencies:
   ```bash
   cd fast_api
   pip install -r requirements.txt
   ```

2. Start Qdrant locally or use the Docker Compose setup.

3. Run the FastAPI app:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8800
   ```

## Docker Services

- **qdrant**: Qdrant vector database (ports 3336:6333, 3346:6334)
- **fast_api**: FastAPI application (port 8800)

## License

[Add your license here]</content>
<parameter name="filePath">/Users/hmimeee/code/qdrant-server/README.md