import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from src.rag_pipeline import process_chat_query
import uvicorn

# Bir önceki adımda oluşturduğumuz config dosyasından ayarları çekiyoruz
from src.config import settings

app = FastAPI(title=settings.app_name)

# --- Veri Modelleri ---
class ChatRequest(BaseModel):
    message: str

# --- Uç Noktalar (Endpoints) ---

@app.get("/health")
async def health_check():
    """Sistemin ayakta olup olmadığını ve hangi modda çalıştığını kontrol eder."""
    return {
        "status": "ok", 
        "mode": settings.mode, 
        "chat_model": settings.foundry_chat_model
    }

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Kullanıcıdan gelen soruları RAG boru hattına (pipeline) iletecek ana uç nokta."""
    try:
        reply = process_chat_query(request.message)
        return {"reply": reply}
    except Exception as e:
        return {"reply": f"An error occurred: {str(e)}"}

@app.post("/ingest")
async def ingest_endpoint():
    """Dokümanların vektör veritabanına aktarımını tetikleyecek uç nokta."""
    # İleride src/chunking.py ve veri tabanı entegrasyonu buraya gelecek.
    return {"status": "success", "message": "Doküman aktarımı başlatıldı (Placeholder)."}

# --- Statik Arayüz Sunumu ---

@app.get("/")
async def serve_ui():
    """Kullanıcı arayüzünü (static/index.html) sunar."""
    ui_path = os.path.join("static", "index.html")
    if os.path.exists(ui_path):
        return FileResponse(ui_path)
    return {"message": "Arayüz dosyası bulunamadı. Lütfen static/index.html dosyasını oluşturun."}

if __name__ == "__main__":
    uvicorn.run("api.app:app", host="127.0.0.1", port=8000, reload=True)