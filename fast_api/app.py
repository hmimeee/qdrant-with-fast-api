from fastapi import FastAPI, Depends, HTTPException, status, Header, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance
import fitz  # PyMuPDF

import uuid
import os
import traceback


# ======================
# Config
# ======================
API_KEY = os.getenv("API_KEY")

def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return True

app = FastAPI(dependencies=[Depends(verify_api_key)])

# ======================
# Error Handlers
# ======================
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status": exc.status_code,
            "path": str(request.url),
        },
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "exception_type": type(exc).__name__,
            "traceback": tb,
            "path": str(request.url),
        },
    )


# ======================
# Qdrant + Embeddings
# ======================
embedding_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
qdrant = QdrantClient(url=os.getenv("QDRANT_URL"))


# ======================
# Schemas
# ======================
class BaseWithCollection(BaseModel):
    collection: str   # required


class TextsInput(BaseWithCollection):
    texts: list[str]


class SearchInput(BaseWithCollection):
    query: str
    top_k: int = 5


class SingleTextInput(BaseWithCollection):
    text: str


# ======================
# Utilities
# ======================
def ensure_collection(collection_name: str):
    if not qdrant.collection_exists(collection_name):
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=embedding_model.embedding_size,
                distance=Distance.COSINE
            )
        )


def pdf_to_md(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    md_content = []
    
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        text = span["text"]
                        font_size = span["size"]
                        font_name = span["font"]
                        flags = span["flags"]
                        
                        # Detect heading: large font size
                        if font_size > 14:
                            text = f"# {text}"
                        # Bold: check font name or flags
                        if "Bold" in font_name or (flags & 16):  # bold flag
                            text = f"**{text}**"
                        # Italics: flags & 2
                        if flags & 2:
                            text = f"*{text}*"
                        
                        line_text += text
                    
                    if line_text.strip():
                        md_content.append(line_text)
                md_content.append("")  # blank line after block
    
    doc.close()
    return "\n".join(md_content)

# ======================
# Endpoints
# ======================
@app.post("/embed")
def get_embedding(data: SingleTextInput):
    collection = data.collection
    ensure_collection(collection)

    embedding_vector = list(embedding_model.passage_embed([data.text]))[0]
    return {
        "collection": collection,
        "text": data.text,
        "embedding": list(embedding_vector)
    }


@app.post("/import")
def embed_and_store(data: TextsInput):
    collection = data.collection
    ensure_collection(collection)

    embeddings = list(embedding_model.passage_embed(data.texts))
    payloads = [{"text": t} for t in data.texts]
    ids = [str(uuid.uuid4()) for _ in data.texts]

    qdrant.upsert(
        collection_name=collection,
        points=[
            {"id": ids[i], "vector": embeddings[i], "payload": payloads[i]}
            for i in range(len(data.texts))
        ]
    )

    return {"status": "ok", "inserted": len(data.texts), "ids": ids, "collection": collection}


@app.post("/search")
def search_qdrant(data: SearchInput):
    collection = data.collection
    ensure_collection(collection)

    query_vector = list(embedding_model.passage_embed([data.query]))[0]

    results = qdrant.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=data.top_k
    )

    return {
        "query": data.query,
        "collection": collection,
        "results": [
            {"id": r.id, "score": r.score, "text": r.payload.get("text")}
            for r in results
        ]
    }


@app.delete("/collection")
def delete_collection(data: BaseWithCollection):
    collection = data.collection

    if not qdrant.collection_exists(collection):
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{collection}' does not exist"
        )

    qdrant.delete_collection(collection_name=collection)

    return {
        "status": "deleted",
        "collection": collection
    }

@app.post("/parse-pdf")
async def parse_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only PDF files are supported."
        )
    
    pdf_bytes = await file.read()
    md_content = pdf_to_md(pdf_bytes)

    return {
        "filename": file.filename,
        "markdown": md_content
    }
