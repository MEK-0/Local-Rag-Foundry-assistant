from typing import List, Dict, Any

# Cross-encoder relevance threshold. bge-reranker-base outputs raw logits
# (typically -10..+10 range) rather than a 0-1 probability. Relevant
# query/chunk pairs cluster positive, irrelevant pairs cluster negative —
# so 0.0 is the meaningful cutoff. The previous -3.0 threshold was below
# the model's practical output floor, meaning this filter never rejected
# anything and only the Jaccard dedup step was actually pruning chunks.
RERANK_SCORE_THRESHOLD = 0.0

# Node types treated as atomic, authoritative sources (a full table, a full
# warning box). Two different small tables can share boilerplate header
# rows ("| Parameter | Value |") and get falsely flagged as Jaccard
# duplicates despite holding different data. These types are already
# id-deduplicated upstream (seen_chunk_ids in rag_pipeline.py), so they
# skip content-similarity dedup here.
DEDUP_EXEMPT_TYPES = {"table", "figure", "warning", "note", "code"}


def grade_retrieved_chunks(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Grades retrieved chunks: enforces semantic uniqueness using Jaccard
    similarity to prevent redundant context, and prunes chunks that pass
    neither the cross-encoder relevance bar nor a basic keyword-hit check.
    """
    valid_chunks = []
    seen_texts = set()

    for chunk in chunks:
        text_normalized = " ".join(chunk.get("chunk_text", "").lower().split())
        if not text_normalized:
            # Empty content (e.g. a figure node awaiting vision captioning)
            # has nothing to grade or dedup against — drop it here.
            continue

        node_type = chunk.get("node_type", "paragraph")
        words_current = set(text_normalized.split())

        is_duplicate = False
        if node_type not in DEDUP_EXEMPT_TYPES:
            for seen_text in seen_texts:
                words_seen = set(seen_text.split())
                intersection = words_current.intersection(words_seen)
                union = words_current.union(words_seen)
                similarity = len(intersection) / len(union) if union else 0.0

                if similarity > 0.70:
                    is_duplicate = True
                    break

        if is_duplicate:
            continue

        # Basic keyword-hit ratio as a secondary relevance signal
        query_words = [w.lower() for w in query.split() if len(w) > 3]
        hit_count = sum(1 for word in query_words if word in text_normalized)
        hit_ratio = hit_count / len(query_words) if query_words else 1.0

        # Keep the chunk if the cross-encoder scored it as relevant OR it
        # has meaningful keyword overlap with the query
        if chunk.get("rerank_score", 0.0) > RERANK_SCORE_THRESHOLD or hit_ratio >= 0.2:
            valid_chunks.append(chunk)
            seen_texts.add(text_normalized)

    return valid_chunks