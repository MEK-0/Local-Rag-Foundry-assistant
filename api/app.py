import os
import shutil
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.rag_pipeline import process_chat_query
from src.chunking import chunk_document          # src/chunking.py'ye eklediğimiz fonksiyon
from src.db import insert_chunk                 # Senin db.py içindeki orijinal fonksiyonun
from src.llm_client import get_embedding
import uvicorn

from src.config import settings

app = FastAPI(title=settings.app_name)

class ChatRequest(BaseModel):
    message: str

@app.get("/health")
async def health_check():
    return {
        "status": "ok", 
        "mode": settings.mode, 
        "chat_model": settings.foundry_chat_model
    }

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        result = process_chat_query(request.message)
        return result
    except Exception as e:
        return {"reply": f"An error occurred: {str(e)}", "thinking": "Pipeline failure."}

# --- DİNAMİK DOSYA YÜKLEME UÇ NOKTASI (UPLOAD ENDPOINT) ---
@app.post("/upload")
async def upload_file_endpoint(file: UploadFile = File(...)):
    """Kullanıcının arayüzden yüklediği .md dosyasını kaydeder, parçalar ve DB'ye yazar."""
    if not file.filename.endswith('.md'):
        raise HTTPException(status_code=400, detail="Sadece Markdown (.md) dosyaları desteklenmektedir.")
    
    # 1. Dosyayı geçici olarak sakla
    target_dir = "docs/sample_docs"
    os.makedirs(target_dir, exist_ok=True)
    file_path = os.path.join(target_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 2. Token tabanlı parçalayıcıyı çağırıyoruz
        chunks = chunk_document(file_path)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Dosya içeriği boş veya parçalara ayrılamadı.")
        
        # 3. Her bir parçanın embedding'ini alıp doğrudan senin insert_chunk fonksiyonunla DB'ye yazıyoruz
        for chunk_text in chunks:
            embedding = get_embedding(chunk_text)
            insert_chunk(
                source_file=file.filename,
                chunk_text=chunk_text,
                embedding=embedding
            )
        
        return {"status": "success", "message": f"'{file.filename}' başarıyla yüklendi ve {len(chunks)} parçaya ayrıldı."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dosya işlenirken hata oluştu: {str(e)}")

@app.get("/")
async def serve_ui():
    ui_path = os.path.join("static", "index.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    return {"message": "Arayüz dosyası bulunamadı. Lütfen static/index.html dosyasını oluşturun."}

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=True)