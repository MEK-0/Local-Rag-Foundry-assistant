from typing import List

from src.llm_client import generate_chat_response

MAX_QUERY_TRACKS = 3

REWRITE_PROMPT_TEMPLATE = """Rewrite the following technical question in {n} different ways, \
preserving its exact meaning but varying the vocabulary (e.g. synonyms, \
alternate phrasing, different terminology for the same concept). This is \
for a document search system, not for a human reader.

Rules:
- Do not answer the question.
- Do not add or remove any information.
- Output ONLY the rewritten variants, one per line, no numbering, no extra text.

Original question:
{query}

Rewritten variants:"""


def rewrite_query(original_query: str) -> List[str]:
    """
    Expands a query into multiple phrasings for multi-track hybrid
    retrieval, using the local/cloud chat model rather than a fixed
    domain-specific synonym dictionary. This generalizes across any
    subject matter instead of only recognizing a handful of hardcoded
    industrial-automation terms (the previous approach only expanded
    queries containing "grease", "spindle", "scara", or "maintenance" -
    every other topic silently got zero expansion).

    Falls back to returning just the original query if the LLM call
    fails or returns something unusable - this keeps retrieval working
    (single-track instead of multi-track) rather than raising an error.
    """
    num_variants = MAX_QUERY_TRACKS - 1  # 1 slot reserved for the original query
    prompt = REWRITE_PROMPT_TEMPLATE.format(n=num_variants, query=original_query)

    try:
        raw_response = generate_chat_response(prompt)
    except Exception:
        return [original_query]

    if not raw_response or "error during generation" in raw_response.lower():
        return [original_query]

    variants = [line.strip("-* \t") for line in raw_response.strip().split("\n")]
    variants = [v for v in variants if v and v.lower() != original_query.lower()]

    expanded_queries = [original_query] + variants
    # Dedup while preserving order (set() would not guarantee order,
    # and query-track order affects which candidates get merged first
    # in rag_pipeline.py's seen_chunk_ids loop)
    seen = set()
    unique_queries = []
    for q in expanded_queries:
        key = q.lower()
        if key not in seen:
            seen.add(key)
            unique_queries.append(q)

    return unique_queries[:MAX_QUERY_TRACKS]

SUBQUERY_PROMPT_TEMPLATE = """The following question may require combining multiple separate \
pieces of information to answer fully. Break it down into {n} simpler, \
self-contained sub-questions that together would let someone answer the \
original question.

Rules:
- Do not answer the question.
- Each sub-question must be answerable on its own.
- Output ONLY the sub-questions, one per line, no numbering, no extra text.

Original question:
{query}

Sub-questions:"""


def decompose_into_subqueries(original_query: str, num_subqueries: int = 2) -> List[str]:
    """
    Splits a complex question into simpler sub-questions for a follow-up
    retrieval hop. Used only when the first retrieval pass is judged
    insufficient (see _is_context_sufficient in rag_pipeline.py) - this
    keeps the extra LLM call rare rather than paid on every query.

    Falls back to an empty list on any failure, so the caller can safely
    skip the extra hop instead of crashing.
    """
    prompt = SUBQUERY_PROMPT_TEMPLATE.format(n=num_subqueries, query=original_query)

    try:
        raw_response = generate_chat_response(prompt)
    except Exception:
        return []

    if not raw_response or "error during generation" in raw_response.lower():
        return []

    subqueries = [line.strip("-* \t") for line in raw_response.strip().split("\n")]
    subqueries = [q for q in subqueries if q and q.lower() != original_query.lower()]
    return subqueries[:num_subqueries]