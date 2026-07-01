from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # --- Mode Switch ---
    mode: str = "local"

    # --- App Settings ---
    app_name: str = "local-rag-assistant"
    top_k: int = 3
    chunk_size_tokens: int = 200
    chunk_overlap_tokens: int = 40

    # --- Local Models ---
    foundry_chat_model: str = "qwen3-0.6b-generic-gpu:2"
    foundry_embedding_model: str = "qwen3-0.6b"

    # --- Cloud Variables (Bulut modunda doldurulacak) ---
    azure_search_endpoint: Optional[str] = None
    azure_search_key: Optional[str] = None
    azure_search_index: Optional[str] = "rag-index"

    azure_openai_endpoint: Optional[str] = None
    azure_openai_key: Optional[str] = None
    azure_openai_deployment: Optional[str] = "gpt-4o-mini"
    azure_openai_api_version: Optional[str] = "2024-08-01-preview"

    # .env dosyasını otomatik okuması için yapılandırma
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

# Uygulama genelinde kullanılacak ayar objesi
settings = Settings()