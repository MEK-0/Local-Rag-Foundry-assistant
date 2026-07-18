import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.db import get_all_chunks_for_sparse
from src.azure_storage import sync_documents_directory
from src.azure_search import ensure_index_exists, upload_chunks


def run_sync():
    """
    Syncs local state to Azure for cloud mode:
      1. Raw document files -> Azure Blob Storage (docs/sample_docs/*)
      2. Indexed chunks + embeddings (already computed locally) -> Azure AI Search

    This does NOT sync knowledge_nodes or the entity graph - per the
    "Local & cloud strategy" design, those remain local-SQLite-only in
    both modes. Only the flat retrieval index moves to Azure.

    Chunks must already be embedded (i.e. scripts/ingest.py has run)
    before this script is useful - it reads existing embeddings from
    SQLite rather than recomputing them, since embeddings were already
    paid for once during local ingestion.
    """
    if settings.mode != "cloud":
        print(f"MODE is '{settings.mode}', not 'cloud'. Set MODE=cloud in .env before syncing to Azure.")
        print("Proceeding anyway - this only pushes data, it doesn't change how /chat behaves.")

    source_dir = "docs/sample_docs"

    print("Step 1/2: Syncing raw documents to Azure Blob Storage...")
    try:
        if not os.path.exists(source_dir):
            print(f"   Source directory '{source_dir}' not found - skipping blob sync.")
        else:
            uploaded = sync_documents_directory(source_dir)
            print(f"   Uploaded {len(uploaded)} file(s) to container '{settings.azure_storage_container}'.")
    except Exception as e:
        print(f"   Blob Storage sync failed: {e}")
        print("   Check AZURE_STORAGE_CONNECTION_STRING in .env.")

    print("\nStep 2/2: Syncing indexed chunks to Azure AI Search...")
    try:
        ensure_index_exists()
        print(f"   Index '{settings.azure_search_index}' ready.")

        all_chunks = get_all_chunks_for_sparse()
        if not all_chunks:
            print("   No chunks found in local SQLite. Run 'python scripts/ingest.py' first.")
            return

        # Azure AI Search recommends batching uploads rather than sending
        # everything in one request - 1000 is comfortably under its
        # per-request document limit.
        batch_size = 1000
        total_indexed = 0
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            indexed_count = upload_chunks(batch)
            total_indexed += indexed_count
            print(f"   Indexed batch {i // batch_size + 1}: {indexed_count}/{len(batch)} chunks.")

        print(f"\n   Total: {total_indexed}/{len(all_chunks)} chunks synced to Azure AI Search.")

    except RuntimeError as e:
        # Raised by _require_search_config() when endpoint/key are missing
        print(f"   Azure AI Search sync failed: {e}")
        print("   Check AZURE_SEARCH_ENDPOINT / AZURE_SEARCH_KEY in .env.")
    except Exception as e:
        print(f"   Azure AI Search sync failed: {e}")


if __name__ == "__main__":
    run_sync()