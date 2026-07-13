import numpy as np
from typing import List, Dict, Any, Optional
from rank_bm25 import BM25Okapi

from src.db import get_all_chunks_for_sparse, get_db_connection

# --------------------------------------------------------------------------- #
# In-memory index cache
#
# Rebuilding BM25 from scratch and computing cosine similarity in a Python
# loop on every single query does not scale past a few hundred chunks - at
# a few thousand chunks this alone can take seconds per query. The index
# only actually changes when ingest.py adds/removes chunks, so it should be
# built once and reused, not rebuilt per request.
#
# Staleness check: a cheap COUNT(*) against document_chunks. If it differs
# from what the cache was built with, the corpus changed (new ingestion
# run) and we rebuild. This avoids needing to manually call an invalidate
# function from ingest.py, at the cost of one lightweight COUNT query per
# hybrid_retrieve() call.
# --------------------------------------------------------------------------- #

_cache: Dict[str, Any] = {
    "chunk_count": None,
    "chunks": None,          # List[Dict] - full chunk payloads, index-aligned with embedding_matrix
    "bm25": None,            # BM25Okapi instance built once
    "embedding_matrix": None,  # np.ndarray, shape (n_chunks, dim), L2-normalized rows
}


def _current_chunk_count() -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM document_chunks")
    count = cursor.fetchone()["cnt"]
    conn.close()
    return count


def _build_index() -> None:
    """Fetches all chunks once and builds both the BM25 index and a
    normalized dense embedding matrix, cached for reuse across queries."""
    all_chunks = get_all_chunks_for_sparse()

    if not all_chunks:
        _cache.update(chunk_count=0, chunks=[], bm25=None, embedding_matrix=None)
        return

    corpus_tokenized = [chunk["chunk_text"].lower().split() for chunk in all_chunks]
    bm25 = BM25Okapi(corpus_tokenized)

    # Build a single (n_chunks, dim) matrix and normalize each row once here,
    # instead of computing norms per-chunk on every query. With normalized
    # rows, cosine similarity against a normalized query vector reduces to
    # a single matrix-vector dot product (matrix @ query_vec).
    embedding_matrix = np.array([chunk["embedding"] for chunk in all_chunks], dtype=np.float32)
    norms = np.linalg.norm(embedding_matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10  # guard against zero-vector edge case
    embedding_matrix = embedding_matrix / norms

    _cache.update(
        chunk_count=len(all_chunks),
        chunks=all_chunks,
        bm25=bm25,
        embedding_matrix=embedding_matrix,
    )


def _get_index() -> Dict[str, Any]:
    """Returns the cached index, rebuilding it if the corpus size changed
    since it was last built (i.e. ingest.py added/removed chunks)."""
    live_count = _current_chunk_count()
    if _cache["chunk_count"] != live_count:
        _build_index()
    return _cache


def invalidate_index_cache() -> None:
    """
    Explicit cache invalidation hook. The COUNT(*) check in _get_index()
    already catches added/removed chunks automatically, but it would miss
    an edit that replaces content without changing the row count (e.g. a
    future update-in-place feature). Call this after any bulk write to
    document_chunks to be safe, e.g. at the end of ingest.py's run.
    """
    _cache.update(chunk_count=None, chunks=None, bm25=None, embedding_matrix=None)


def hybrid_retrieve(query_text: str, query_embedding: List[float], top_k: int = 5, rrf_k: int = 60) -> List[Dict[str, Any]]:
    """
    Executes hybrid search combining BM25 (sparse) and dense cosine
    similarity, fused via Reciprocal Rank Fusion (RRF).

    Performance notes vs. the previous implementation:
      - BM25 index and the embedding matrix are built once and cached,
        not rebuilt from scratch on every call.
      - Dense similarity is computed as a single vectorized matrix
        multiplication instead of a per-chunk Python loop.

    Args:
        query_text: Raw string question for sparse keyword matching.
        query_embedding: Query vector for dense semantic similarity.
        top_k: Number of top fused results to return.
        rrf_k: RRF penalty constant controlling rank-position weighting.
    """
    index = _get_index()
    all_chunks: Optional[List[Dict[str, Any]]] = index["chunks"]

    if not all_chunks:
        return []

    bm25: BM25Okapi = index["bm25"]
    embedding_matrix: np.ndarray = index["embedding_matrix"]

    # --- LAYER 1: SPARSE RETRIEVAL (BM25) ---
    query_tokens = query_text.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)
    sparse_ranked_indices = np.argsort(bm25_scores)[::-1]

    sparse_ranks: Dict[Any, int] = {}
    for rank_idx, chunk_arr_idx in enumerate(sparse_ranked_indices):
        if bm25_scores[chunk_arr_idx] > 0:
            chunk_id = all_chunks[chunk_arr_idx]["id"]
            sparse_ranks[chunk_id] = rank_idx + 1

    # --- LAYER 2: DENSE RETRIEVAL (vectorized cosine similarity) ---
    query_vec = np.array(query_embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        query_norm = 1e-10
    query_vec = query_vec / query_norm

    # embedding_matrix rows are already normalized -> this dot product IS
    # the cosine similarity for every chunk at once, no Python loop needed
    dense_scores = embedding_matrix @ query_vec
    dense_ranked_indices = np.argsort(dense_scores)[::-1]

    dense_ranks: Dict[Any, int] = {
        all_chunks[chunk_arr_idx]["id"]: rank_idx + 1
        for rank_idx, chunk_arr_idx in enumerate(dense_ranked_indices)
    }

    # --- LAYER 3: RECIPROCAL RANK FUSION ---
    rrf_scores: Dict[Any, float] = {}
    for chunk in all_chunks:
        c_id = chunk["id"]
        sparse_rank = sparse_ranks.get(c_id)
        sparse_score = 1.0 / (rrf_k + sparse_rank) if sparse_rank is not None else 0.0
        dense_rank = dense_ranks.get(c_id)
        dense_score = 1.0 / (rrf_k + dense_rank) if dense_rank is not None else 0.0
        rrf_scores[c_id] = sparse_score + dense_score

    sorted_chunks_by_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    chunk_map = {chunk["id"]: chunk for chunk in all_chunks}
    final_retrieved_chunks = []
    for chunk_id, rrf_final_score in sorted_chunks_by_rrf[:top_k]:
        target_chunk = dict(chunk_map[chunk_id])  # shallow copy, avoid mutating the cached chunk
        target_chunk["rrf_score"] = rrf_final_score
        target_chunk["dense_rank"] = dense_ranks.get(chunk_id, -1)
        target_chunk["sparse_rank"] = sparse_ranks.get(chunk_id, -1)
        final_retrieved_chunks.append(target_chunk)

    return final_retrieved_chunks