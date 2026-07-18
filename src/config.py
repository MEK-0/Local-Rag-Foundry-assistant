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

    # --- Generation Settings ---
    # NOTE: rag_pipeline.py's prompt explicitly instructs "do not truncate"
    # for list/table synthesis answers, so this needs headroom - 500 was too
    # tight for multi-chunk technical answers and was silently cutting them off.
    generation_max_tokens: int = 1000
    generation_temperature: float = 0.3

    # --- Local Models ---
    foundry_chat_model: str = "Phi-4-mini-instruct-generic-gpu:5"
    foundry_base_url: str = "http://127.0.0.1:49327/v1"
    # NOTE: currently unused - get_embedding() in local mode always uses
    # SentenceTransformer("all-MiniLM-L6-v2") directly, ignoring this field.
    # Kept here as a placeholder until embedding model selection is wired up;
    # do not assume changing this value has any effect yet.
    #foundry_embedding_model: str = "qwen3-0.6b"

    # --- Entity graph (opt-in, adds one LLM call per document section) ---
    enable_entity_graph: bool = False

    # --- Azure AI Search (cloud mode) ----------------------------------
    azure_search_endpoint: Optional[str] = None
    azure_search_key: Optional[str] = None
    azure_search_index: Optional[str] = "rag-index"

    # --- Azure Blob Storage (cloud mode, document sync) -----------------
    azure_storage_connection_string: Optional[str] = None
    azure_storage_container: str = "rag-documents"

    # --- Azure OpenAI (cloud mode) --------------------------------------
    azure_openai_endpoint: Optional[str] = None
    azure_openai_key: Optional[str] = None
    azure_openai_deployment: Optional[str] = "gpt-4o-mini"
    azure_openai_api_version: Optional[str] = "2024-08-01-preview"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()