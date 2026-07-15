import os
import shutil
import sys

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import re 

# scripts/ is a sibling of src/, not a package under src - add project root
# so "from scripts.ingest import ..." resolves regardless of working directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag_pipeline import process_chat_query
from src.db import (
    init_db, get_document_hash, upsert_document_record,
    delete_document_data, list_documents,
)
from src.llm_client import get_embedding
from src.parser import get_parser
from scripts.ingest import _make_doc_id, _file_hash, _ingest_tree_aware, _ingest_legacy
from src.db import insert_chunk
import uvicorn

from src.telemetry import init_telemetry_table
from src.config import settings
from src.db import get_graph_data

app = FastAPI(title=settings.app_name)

@app.get("/graph")
async def graph_endpoint(doc_id: str = None):
    """Returns entity co-occurrence graph data for visualization.
    Pass ?doc_id=... to scope to one document, omit for the full graph."""
    return get_graph_data(doc_id)

@app.on_event("startup")
async def startup_event():
    """Ensures the schema exists before the first request, mirroring
    scripts/ingest.py's init_db() call so /upload never hits a missing table."""
    init_db()
    init_telemetry_table()  # creates query_log table if it doesn't exist


class ChatRequest(BaseModel):
    message: str
    advanced_mode: bool = True  # was silently ignored before - UI already sends this field


@app.get("/health")
async def health_check():
    """Checks system availability and active pipeline models."""
    return {
        "status": "ok",
        "mode": settings.mode,
        "chat_model": settings.foundry_chat_model
    }


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Main RAG pipeline entrypoint for answering technical user queries."""
    try:
        result = process_chat_query(request.message, advanced_mode=request.advanced_mode)
        return result
    except Exception as e:
        return {"reply": f"An error occurred: {str(e)}", "thinking": "Pipeline failure.", "telemetry": {}, "chunks_matrix": []}


@app.get("/documents")
async def documents_endpoint():
    """Returns the ingestion registry, used by the UI sidebar to show
    what's actually indexed instead of a hardcoded example list."""
    return {"documents": list_documents()}


# --- DYNAMIC DOCUMENT INGESTION (UPLOAD ENDPOINT) ---
@app.post("/upload")
async def upload_file_endpoint(file: UploadFile = File(...)):
    """
    Saves, parses, chunks, embeds, and indexes an uploaded document.

    Runs the exact same pipeline as scripts/ingest.py's per-file logic
    (tree-aware parsing for PDFs, hash-based dedup, doc_id tracking) so
    a file uploaded through the UI is indexed identically to one placed
    in docs/sample_docs and ingested via the CLI script. Previously this
    endpoint used a separate, simpler code path (chunk_document only,
    no tree/heading/table structure, no dedup tracking) which meant
    UI-uploaded PDFs lost table/figure/heading nodes entirely.
    """
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()

    supported_extensions = ['.md', '.pdf', '.docx', '.xlsx', '.xls', '.csv']
    if ext not in supported_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported types: {', '.join(supported_extensions)}"
        )

    target_dir = "docs/sample_docs"
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        doc_id = _make_doc_id(filename)
        current_hash = _file_hash(file_path)
        stored_hash = get_document_hash(doc_id)

        if stored_hash == current_hash:
            return {
                "status": "unchanged",
                "message": f"'{filename}' is identical to the already-indexed version - skipped.",
                "analytics": {"chunk_count": 0},
            }

        if stored_hash is not None:
            delete_document_data(doc_id)

        parser = get_parser(file_path)

        if hasattr(parser, "parse_to_tree"):
            chunks_data = _ingest_tree_aware(parser, file_path, doc_id, filename)
        else:
            chunks_data = _ingest_legacy(parser, file_path, filename, doc_id)

        if not chunks_data:
            raise HTTPException(status_code=400, detail="The file content is empty or could not be chunked properly.")

        indexed_count = 0
        for item in chunks_data:
            if not item["chunk_text"].strip():
                continue
            embedding = get_embedding(item["chunk_text"])
            insert_chunk(item, embedding)
            indexed_count += 1

        upsert_document_record(
            doc_id=doc_id,
            filename=filename,
            file_hash=current_hash,
            node_count=len(chunks_data),
            chunk_count=indexed_count,
            embedding_model=EMBEDDING_MODEL_NAME,
        )

        return {
            "status": "success",
            "message": f"'{filename}' successfully indexed into {indexed_count} chunks.",
            "analytics": {"chunk_count": indexed_count},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")


@app.get("/")
async def serve_ui():
    """Serves the central static frontend index layout."""
    ui_path = os.path.join("static", "index.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    return {"message": "Interface file not found. Please create the static/index.html file."}

def _safe_filename(filename: str) -> str:
    """
    Prevents path traversal (e.g. "../../etc/passwd") when serving files
    by name. Only allows the plain filename, no directory separators.
    """
    return os.path.basename(filename)


@app.get("/source/{filename}")
async def serve_source_document(filename: str):
    """
    Serves an ingested source file for the citation viewer. The UI links
    to this with a #page=N fragment, which browser-native PDF viewers
    (Chrome, Firefox, Edge) honor to jump straight to that page - no
    PDF.js integration needed for this simple version.
    """
    safe_name = _safe_filename(filename)
    file_path = os.path.join("docs/sample_docs", safe_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Source file '{safe_name}' not found.")

    return FileResponse(file_path)

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=True)