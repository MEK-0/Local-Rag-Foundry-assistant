"""
Azure Blob Storage sync for cloud mode.

Scope: raw document sync only (upload the original files docs/sample_docs
so they're available if Azure AI Search needs re-indexing, or for a future
document-download feature). This does NOT store chunks, embeddings, or the
knowledge tree - those stay in local SQLite in both modes (see README's
"Local & cloud strategy" section for why).
"""

import os
from typing import List

from azure.storage.blob import BlobServiceClient, ContentSettings

from src.config import settings


def get_blob_service_client() -> BlobServiceClient:
    if not settings.azure_storage_connection_string:
        raise RuntimeError(
            "AZURE_STORAGE_CONNECTION_STRING is not set. Required for cloud mode document sync."
        )
    return BlobServiceClient.from_connection_string(settings.azure_storage_connection_string)


def ensure_container_exists() -> None:
    client = get_blob_service_client()
    container_client = client.get_container_client(settings.azure_storage_container)
    if not container_client.exists():
        container_client.create_container()


def upload_document(file_path: str, filename: str) -> str:
    """
    Uploads a single document to Blob Storage. Returns the blob URL.
    Overwrites any existing blob with the same name (re-sync is idempotent,
    matching the same "re-running is safe" property as local ingestion).
    """
    client = get_blob_service_client()
    container_client = client.get_container_client(settings.azure_storage_container)

    with open(file_path, "rb") as f:
        blob_client = container_client.upload_blob(
            name=filename,
            data=f,
            overwrite=True,
            content_settings=ContentSettings(content_type="application/octet-stream"),
        )
    return blob_client.url


def sync_documents_directory(source_dir: str) -> List[str]:
    """
    Uploads every file in source_dir to Blob Storage. Returns the list of
    filenames successfully uploaded. Failures on individual files are
    logged and skipped rather than aborting the whole sync - one corrupt
    file (see ingest.py's handling of the same issue locally) shouldn't
    block every other document from syncing.
    """
    ensure_container_exists()
    uploaded = []

    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        if not os.path.isfile(file_path):
            continue
        try:
            upload_document(file_path, filename)
            uploaded.append(filename)
        except Exception as e:
            print(f"   Failed to upload {filename} to Blob Storage: {e}")

    return uploaded