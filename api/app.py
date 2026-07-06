import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.rag_pipeline import process_chat_query
from src.chunking import chunk_document          
from src.db import insert_chunk                 
from src.llm_client import get_embedding
from src.parsers import get_parser                  # Import the central parser factory
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
    """Saves, parses, tokens, embeds, and indexes multi-format documents locally into SQLite."""
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    
    # Supported file extensions validation guardrail
    supported_extensions = ['.md', '.pdf', '.docx', '.xlsx', '.xls', '.csv']
    if ext not in supported_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file format. Supported types: {', '.join(supported_extensions)}"
        )
    
    # Save the file temporarily to local disk storage
    target_dir = "docs/sample_docs"
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 1. Dynamically retrieve the dedicated parser via the factory module
        parser = get_parser(file_path)
        parsed_data = parser.parse(file_path)
        
        raw_content = parsed_data["content"]
        file_metadata = parsed_data["metadata"]
        
        # 2. Process the extracted document content into token-aware semantic chunks
        # Uses standard configurations (size=500, overlap=50)
        chunks_data = chunk_document(file_path, raw_content)
        
        if not chunks_data:
            raise HTTPException(status_code=400, detail="The file content is empty or could not be chunked properly.")
        
        # 3. Compute vector embeddings and persist chunks sequentially with rich metadata
        for item in chunks_data:
            chunk_text = item["chunk_text"]
            chunk_metadata = item["metadata"]
            
            # Generate the vector representation via Foundry Local embedding client
            embedding = get_embedding(chunk_text)
            
            # Since our native insert_chunk takes discrete parameters, we pass the extracted metadata
            # You can expand the local DB schema later to store page_number explicitly
            insert_chunk(
                source_file=chunk_metadata["source_file"],
                chunk_text=chunk_text,
                embedding=embedding
            )
        
        return {
            "status": "success", 
            "message": f"'{filename}' successfully parsed via {file_metadata['parser_used']} and indexed into {len(chunks_data)} metadata-rich chunks.",
            "analytics": {
                "chunk_count": len(chunks_data),
                "file_metadata": file_metadata
            }
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