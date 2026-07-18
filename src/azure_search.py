"""
Azure AI Search integration for cloud mode.

Design constraint (see README's "Local & cloud strategy"): this module
mirrors document_chunks into a flat Azure AI Search index for retrieval
only. It does NOT attempt to represent the knowledge_nodes hierarchy or
the entity graph in Azure - those remain local-SQLite-only in both modes.

azure_hybrid_retrieve() returns results in the exact same shape as
hybrid.py's hybrid_retrieve(), so rag_pipeline.py's rerank/grade/compress/
parent-expansion/follow-up-hop logic runs unmodified regardless of MODE.
"""

from typing import List, Dict, Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

from src.config import settings

EMBEDDING_DIMENSIONS = 1536  # Azure OpenAI text-embedding-3-small / ada-002 output size
VECTOR_PROFILE_NAME = "rag-vector-profile"
VECTOR_ALGORITHM_NAME = "rag-hnsw"


def _require_search_config() -> None:
    if not (settings.azure_search_endpoint and settings.azure_search_key):
        raise RuntimeError(
            "AZURE_SEARCH_ENDPOINT / AZURE_SEARCH_KEY are not set. Required for cloud mode retrieval."
        )


def get_search_index_client() -> SearchIndexClient:
    _require_search_config()
    return SearchIndexClient(
        endpoint=settings.azure_search_endpoint,
        credential=AzureKeyCredential(settings.azure_search_key),
    )


def get_search_client() -> SearchClient:
    _require_search_config()
    return SearchClient(
        endpoint=settings.azure_search_endpoint,
        index_name=settings.azure_search_index,
        credential=AzureKeyCredential(settings.azure_search_key),
    )


def ensure_index_exists() -> None:
    """
    Creates the Azure AI Search index if it doesn't exist yet. Field set
    mirrors document_chunks' columns (minus embedding-as-BLOB, which
    becomes a native vector field here) - this keeps the shape returned
    by azure_hybrid_retrieve() consistent with hybrid_retrieve()'s output.
    """
    index_client = get_search_index_client()

    existing = [idx.name for idx in index_client.list_indexes()]
    if settings.azure_search_index in existing:
        return

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="doc_id", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="source_file", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="page_number", type=SearchFieldDataType.Int32, filterable=True),
        SearchableField(name="chunk_text", type=SearchFieldDataType.String),
        SimpleField(name="node_type", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="heading_path", type=SearchFieldDataType.String),
        SimpleField(name="parent_id", type=SearchFieldDataType.String),
        SimpleField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name=VECTOR_PROFILE_NAME,
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name=VECTOR_ALGORITHM_NAME)],
        profiles=[
            VectorSearchProfile(
                name=VECTOR_PROFILE_NAME,
                algorithm_configuration_name=VECTOR_ALGORITHM_NAME,
            )
        ],
    )

    index = SearchIndex(name=settings.azure_search_index, fields=fields, vector_search=vector_search)
    index_client.create_index(index)


def upload_chunks(chunks: List[Dict[str, Any]]) -> int:
    """
    Uploads a batch of chunks (same dict shape as db.get_all_chunks_for_sparse()
    output, with an "embedding" list of floats) to the search index.
    Returns the count of successfully indexed documents.

    Azure AI Search requires a document "id" - reuses the local SQLite
    row id (str-cast, since Azure keys must be strings) so re-syncing the
    same chunk overwrites rather than duplicates.
    """
    client = get_search_client()

    documents = [
        {
            "id": str(chunk["id"]),
            "doc_id": chunk.get("doc_id") or "",
            "source_file": chunk.get("source_file", ""),
            "page_number": chunk.get("page_number", 1),
            "chunk_text": chunk["chunk_text"],
            "node_type": chunk.get("node_type", "paragraph"),
            "heading_path": chunk.get("heading_path", ""),
            "parent_id": chunk.get("parent_id") or "",
            "embedding": chunk["embedding"],
        }
        for chunk in chunks
    ]

    result = client.upload_documents(documents=documents)
    return sum(1 for r in result if r.succeeded)


def azure_hybrid_retrieve(query_text: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Azure AI Search's native hybrid search (BM25 + vector, fused server-side)
    used as a drop-in replacement for hybrid.py's hybrid_retrieve().

    Returns the same dict shape (id, chunk_text, source_file, page_number,
    node_type, heading_path, parent_id, rrf_score) so every downstream
    stage in rag_pipeline.py (rerank, grade, compress, parent-expansion,
    follow-up hop) works without modification.
    """
    client = get_search_client()

    vector_query = VectorizedQuery(vector=query_embedding, k_nearest_neighbors=top_k, fields="embedding")

    results = client.search(
        search_text=query_text,       # BM25/keyword side of hybrid search
        vector_queries=[vector_query],  # vector side of hybrid search
        top=top_k,
        select=["id", "doc_id", "source_file", "page_number", "chunk_text", "node_type", "heading_path", "parent_id"],
    )

    retrieved = []
    for r in results:
        retrieved.append({
            "id": r["id"],
            "doc_id": r.get("doc_id"),
            "chunk_text": r["chunk_text"],
            "source_file": r.get("source_file"),
            "page_number": r.get("page_number"),
            "node_type": r.get("node_type", "paragraph"),
            "heading_path": r.get("heading_path", ""),
            "parent_id": r.get("parent_id") or None,
            # Azure's @search.score is its own fused relevance score -
            # named rrf_score here (not a literal RRF computation) purely
            # so the field name matches what rag_pipeline.py/compression.py
            # already read from hybrid_retrieve()'s local-mode results.
            "rrf_score": r["@search.score"],
        })

    return retrieved