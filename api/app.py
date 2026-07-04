import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.rag_pipeline import process_chat_query
from src.chunking import chunk_document          
from src.db import insert_chunk                 
from src.llm_client import get_embedding
import uvicorn

from src.config import settings

app = FastAPI(title=settings.app_name)

class ChatRequest(BaseModel):
    message: str

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
        result = process_chat_query(request.message)
        return result
    except Exception as e:
        return {"reply": f"An error occurred: {str(e)}", "thinking": "Pipeline failure."}

# --- DYNAMIC DOCUMENT INGESTION (UPLOAD ENDPOINT) ---
@app.post("/upload")
async def upload_file_endpoint(file: UploadFile = File(...)):
    """Saves, tokens, embeds, and indexes an uploaded Markdown file locally into SQLite."""
    if not file.filename.endswith('.md'):
        raise HTTPException(status_code=400, detail="Only Markdown (.md) files are supported.")
    
    # Save the file temporarily to local disk storage
    target_dir = "docs/sample_docs"
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Process the document into token-aware semantic sections
        chunks = chunk_document(file_path)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="The file content is empty or could not be chunked properly.")
        
        # Compute vector embeddings and persist chunks sequentially to the database
        for chunk_text in chunks:
            embedding = get_embedding(chunk_text)
            insert_chunk(
                source_file=file.filename,
                chunk_text=chunk_text,
                embedding=embedding
            )
        
        return {
            "status": "success", 
            "message": f"'{file.filename}' successfully uploaded and indexed into {len(chunks)} token-aware chunks."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

@app.get("/")
async def serve_ui():
    """Serves the central static frontend index layout."""
    ui_path = os.path.join("static", "index.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    return {"message": "Interface file not found. Please create the static/index.html file."}

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=True)