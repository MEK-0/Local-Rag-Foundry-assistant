import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.db import get_all_chunks_for_sparse
from src.llm_client import get_embedding
from src.azure_storage import sync_documents_directory
from src.azure_search import ensure_index_exists, upload_chunks


def run_sync():
    """
    Syncs local state to Azure for cloud mode:
      1. Raw document files -> Azure Blob Storage (docs/sample_docs/*)
      2. Indexed chunks -> Azure AI Search, re-embedded via Azure OpenAI

    This does NOT sync knowledge_nodes or the entity graph - per the
    "Local & cloud strategy" design, those remain local-SQLite-only in
    both modes. Only the flat retrieval index moves to Azure.

    Re-embedding is required, not optional: local chunks were embedded
    with the local SentenceTransformer (384-dim), but Azure AI Search's
    vector field is sized for Azure OpenAI embeddings (1536-dim) - the
    two are not interchangeable (same issue hybrid.py's embedding-model
    consistency guard protects against on the retrieval side). This
    script re-embeds chunk_text via get_embedding(), which automatically
    uses Azure OpenAI when MODE=cloud - this re-pays the embedding cost
    once per sync, but is the only correct way to get compatible vectors.
    """
    if settings.mode != "cloud":
        print(f"MODE is '{settings.mode}', not 'cloud'. Set MODE=cloud in .env before syncing to Azure.")
        print("Re-embedding below will use whatever backend the current MODE selects,")
        print("which would produce vectors of the wrong dimension for Azure AI Search if MODE=local.")
        return

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

    print("\nStep 2/2: Re-embedding and syncing chunks to Azure AI Search...")
    try:
        ensure_index_exists()
        print(f"   Index '{settings.azure_search_index}' ready.")

        all_chunks = get_all_chunks_for_sparse()
        if not all_chunks:
            print("   No chunks found in local SQLite. Run 'python scripts/ingest.py' first.")
            return

        print(f"   Re-embedding {len(all_chunks)} chunks via Azure OpenAI (this re-pays embedding cost)...")
        re_embedded_chunks = []
        failed_count = 0
        for idx, chunk in enumerate(all_chunks, start=1):
            try:
                chunk["embedding"] = get_embedding(chunk["chunk_text"])
                re_embedded_chunks.append(chunk)
            except Exception as e:
                failed_count += 1
                print(f"      [{idx}/{len(all_chunks)}] embedding failed for chunk {chunk.get('id')}: {e}")

            if idx % 100 == 0:
                print(f"      ...{idx}/{len(all_chunks)} embedded")

        if failed_count:
            print(f"   Warning: {failed_count} chunk(s) failed re-embedding and will be skipped.")

        batch_size = 1000
        total_indexed = 0
        for i in range(0, len(re_embedded_chunks), batch_size):
            batch = re_embedded_chunks[i:i + batch_size]
            indexed_count = upload_chunks(batch)
            total_indexed += indexed_count
            print(f"   Indexed batch {i // batch_size + 1}: {indexed_count}/{len(batch)} chunks.")

        print(f"\n   Total: {total_indexed}/{len(all_chunks)} chunks synced to Azure AI Search.")

    except RuntimeError as e:
        print(f"   Azure AI Search sync failed: {e}")
        print("   Check AZURE_SEARCH_ENDPOINT / AZURE_SEARCH_KEY in .env.")
    except Exception as e:
        print(f"   Azure AI Search sync failed: {e}")


if __name__ == "__main__":
    run_sync()