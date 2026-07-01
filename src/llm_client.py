from openai import OpenAI, AzureOpenAI
from src.config import settings
from typing import List

if settings.mode == "local":
    client = OpenAI(
        base_url="http://127.0.0.1:50124/v1",
        api_key="not-needed-for-local"
    )
    # Yerel vektörleştirme için sentence-transformers (Çevrimdışı çalışır)
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
else:
    client = AzureOpenAI(
        api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_endpoint or "",
        api_key=settings.azure_openai_key or ""
    )

def get_embedding(text: str) -> List[float]:
    """
    Generates a vector embedding for the provided text.
    """
    clean_text = text.replace("\n", " ")
    
    if settings.mode == "local":
        # Yerel cihazda anında vektör üretir
        return embedder.encode(clean_text).tolist()
    else:
        response = client.embeddings.create(
            input=clean_text,
            model=settings.azure_openai_deployment
        )
        return response.data[0].embedding

def generate_chat_response(prompt: str) -> str:
    """
    Placeholder for the generation step.
    """
    pass