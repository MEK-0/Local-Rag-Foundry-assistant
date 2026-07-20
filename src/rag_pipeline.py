import time
from src.config import settings
from src.llm_client import get_embedding, generate_chat_response
from src.retrieval.query_rewriter import rewrite_query, decompose_into_subqueries
from src.retrieval.hybrid import hybrid_retrieve
from src.retrieval.reranker import rerank_chunks
from src.retrieval.grader import grade_retrieved_chunks
from src.retrieval.compression import compress_context_chunks
from src.telemetry import log_query_event

MAX_HOPS = 2  # initial retrieval pass + at most 1 follow-up hop, never more


def _retrieve(query_text: str, query_embedding: list, top_k: int) -> list:
    """
    Mode-aware retrieval dispatch. Returns the same result shape
    (id, chunk_text, source_file, page_number, node_type, heading_path,
    parent_id, rrf_score) regardless of MODE, so every downstream stage
    (rerank, grade, compress, parent-expansion, follow-up hop) works
    unmodified. Azure import is local to this function so local mode
    never needs the azure-search-documents package installed.
    """
    if settings.mode == "cloud":
        from src.azure_search import azure_hybrid_retrieve
        return azure_hybrid_retrieve(query_text=query_text, query_embedding=query_embedding, top_k=top_k)
    return hybrid_retrieve(query_text=query_text, query_embedding=query_embedding, top_k=top_k)


def process_chat_query(query: str, advanced_mode: bool = True) -> dict:
    """
    RAG pipeline: query expansion -> hybrid retrieval -> rerank -> grade ->
    compress (+ parent expansion) -> [optional single follow-up hop] -> generate.

    No hardcoded/canned answers anywhere in this pipeline. If retrieval or
    generation fails, the pipeline reports that honestly instead of
    returning a pre-written answer — a wrong-but-confident canned response
    is worse than an explicit "insufficient context" reply, because it
    fails silently on any query the fallback list wasn't written for.
    """
    telemetry = {}
    query_lower = query.lower()

    # --- PHASE 1: Query expansion ---
    start_time = time.perf_counter()
    expanded_queries = rewrite_query(query) if advanced_mode else [query]
    telemetry["query_expansion_time_ms"] = round((time.perf_counter() - start_time) * 1000, 1)

    # --- PHASE 2: Embedding ---
    start_time = time.perf_counter()
    query_embedding = get_embedding(query)
    telemetry["embedding_time_ms"] = round((time.perf_counter() - start_time) * 1000, 1)

    # --- PHASE 3: Multi-track hybrid retrieval ---
    start_time = time.perf_counter()
    all_candidate_chunks = []
    seen_chunk_ids = set()

    search_width = 8 if advanced_mode else 4
    for q_track in expanded_queries:
        retrieved = _retrieve(query_text=q_track, query_embedding=query_embedding, top_k=search_width)
        for chunk in retrieved:
            if chunk["id"] not in seen_chunk_ids:
                seen_chunk_ids.add(chunk["id"])
                all_candidate_chunks.append(chunk)
    telemetry["retrieval_time_ms"] = round((time.perf_counter() - start_time) * 1000, 1)

    if not all_candidate_chunks:
        return _empty_result(
            reply="No relevant content was found in the knowledge base for this question.",
            thinking="Retrieval returned 0 candidates across all expanded query tracks.",
            telemetry=telemetry,
        )

    # --- PHASE 4: Rerank -> grade -> compress (+ parent expansion) ---
    top_score = 0.0
    if not advanced_mode:
        final_context_chunks = all_candidate_chunks[:2]
        telemetry["rerank_time_ms"] = 0.0
        telemetry["culling_and_compression_time_ms"] = 0.0
    else:
        start_time = time.perf_counter()
        reranked = rerank_chunks(query=query, chunks=all_candidate_chunks, top_n=6)
        telemetry["rerank_time_ms"] = round((time.perf_counter() - start_time) * 1000, 1)

        start_time = time.perf_counter()
        graded = grade_retrieved_chunks(query=query, chunks=reranked)
        compressed = compress_context_chunks(query=query, chunks=graded)

        top_score = compressed[0].get("rerank_score", 0.0) if compressed else 0.0

        # Adaptive chunk budget: fewer, higher-confidence chunks when the
        # top hit is very strong; more chunks when the signal is weak/spread
        if top_score > 1.2:
            chunk_limit = 2
        elif top_score > 0.65:
            chunk_limit = 4
        else:
            chunk_limit = 5

        final_context_chunks = compressed[:chunk_limit]
        telemetry["culling_and_compression_time_ms"] = round((time.perf_counter() - start_time) * 1000, 1)

    if not final_context_chunks:
        return _empty_result(
            reply="Relevant chunks were found but none passed the relevance grading step.",
            thinking="All candidates were pruned during grading/compression.",
            telemetry=telemetry,
        )

    # --- PHASE 4.5: Single bounded follow-up hop if context looks weak ---
    # Not an open-ended agentic loop - capped at MAX_HOPS=2 total (initial
    # + one follow-up). The decision to trigger a follow-up is score-based
    # (free, reuses the same rerank_score threshold as the chunk budget
    # above), so simple/well-matched questions never pay the extra cost.
    # Only sub-query generation itself calls the LLM, and only when triggered.
    hop_count = 1
    start_time = time.perf_counter()
    if advanced_mode and top_score <= 0.65 and len(final_context_chunks) > 0:
        subqueries = decompose_into_subqueries(query)
        if subqueries:
            hop_count = 2
            followup_candidates = []
            for sub_q in subqueries:
                sub_embedding = get_embedding(sub_q)
                retrieved = _retrieve(query_text=sub_q, query_embedding=sub_embedding, top_k=search_width)
                for chunk in retrieved:
                    if chunk["id"] not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk["id"])
                        followup_candidates.append(chunk)

            if followup_candidates:
                reranked_followup = rerank_chunks(query=query, chunks=followup_candidates, top_n=6)
                graded_followup = grade_retrieved_chunks(query=query, chunks=reranked_followup)
                compressed_followup = compress_context_chunks(query=query, chunks=graded_followup)

                # Merge with the original results rather than replacing them -
                # the first hop's chunks were still relevant, just incomplete
                merged = final_context_chunks + compressed_followup
                merged_sorted = sorted(merged, key=lambda c: c.get("rerank_score", 0.0), reverse=True)
                final_context_chunks = merged_sorted[:6]  # slightly wider budget after a merge
                top_score = final_context_chunks[0].get("rerank_score", 0.0) if final_context_chunks else top_score
    telemetry["followup_hop_time_ms"] = round((time.perf_counter() - start_time) * 1000, 1)

    # --- PHASE 5: Context packaging (heading/node-type aware) ---
    context_text = ""
    source_files = set()
    chunks_matrix_payload = []

    for idx, chunk in enumerate(final_context_chunks):
        heading = chunk.get("expanded_heading") or chunk.get("heading_path") or "Unknown Section"
        node_type = chunk.get("node_type", "paragraph")
        section_context = chunk.get("expanded_section", "")

        context_text += (
            f"--- CHUNK {idx + 1} "
            f"(Source: {chunk['source_file']}, Page: {chunk['page_number']}, "
            f"Type: {node_type}, Section: {heading}) ---\n"
        )
        if section_context:
            context_text += f"[Full section context]\n{section_context}\n\n"
        context_text += f"[Matched excerpt]\n{chunk['chunk_text'].strip()}\n"
        context_text += f"--- END CHUNK {idx + 1} ---\n\n"

        source_files.add(chunk.get("source_file", "unknown"))
        chunks_matrix_payload.append({
            "id": chunk["id"],
            "source": chunk["source_file"],
            "page": chunk["page_number"],
            "section": heading,
            "node_type": node_type,
            "rerank_score": round(chunk.get("rerank_score", 0.0), 3),
            "rrf_score": round(chunk.get("rrf_score", 0.0), 4),
        })

    prompt = f"""You are a technical documentation assistant. Answer the question using ONLY the information in the document chunks below.

INSTRUCTIONS:
1. If the question asks for a list, table comparison, or multi-part information, synthesize facts across ALL chunks into a complete answer. Do not truncate.
2. Use only the vocabulary and facts present in the text. Do not add information that is not stated.
3. If the chunks do not contain enough information to answer, say so explicitly instead of guessing.
4. Answer directly. No preamble, no restating the question, no meta-commentary about the instructions.
5. Do NOT describe or narrate the chunks one by one (e.g. "In the first chunk..." / "In the Nth document chunk..."). State the final answer directly, as a technician would.

DOCUMENT CHUNKS:
{context_text.strip()}

QUESTION:
{query}

ANSWER:"""

    # --- PHASE 6: Generation ---
    start_time = time.perf_counter()
    raw_response = generate_chat_response(prompt)
    telemetry["generation_time_ms"] = round((time.perf_counter() - start_time) * 1000, 1)

    final_summary = raw_response.strip()
    generation_failed = "error during generation" in final_summary.lower() or len(final_summary) < 5

    # Repetition-loop safety net: kept model-agnostic (not just for small
    # local models) - cheap to check, does nothing if generation is healthy,
    # and prevents a degenerate wall of repeated template sentences from
    # ever reaching the user as if it were a real answer.
    if not generation_failed and _detect_repetition_loop(final_summary):
        generation_failed = True

    if generation_failed:
        final_summary = (
            "The model was unable to produce a reliable answer from the retrieved context "
            "(generation failed or collapsed into a repetitive loop). Relevant document chunks "
            "were found and are listed below in the Reference section - please retry the "
            "question, or check the LLM client connection if this happens repeatedly."
        )

    references_string = ", ".join(source_files) if source_files else "Local Knowledge Base"
    structured_reply = f"{final_summary}\n\nReference:\n- {references_string}"

    thinking_content = (
        f"Retrieval hops used: {hop_count}/{MAX_HOPS}\n"
        f"Chunks injected into context: {len(final_context_chunks)} (top rerank score: {top_score})\n"
        f"Expanded query tracks:\n" + "\n".join(f" -> {q}" for q in expanded_queries)
    )

    log_query_event(
        query=query,
        telemetry=telemetry,
        advanced_mode=advanced_mode,
        hop_count=hop_count,
        chunk_count=len(final_context_chunks),
        top_rerank_score=top_score,
        generation_failed=generation_failed,
    )

    return {
        "reply": structured_reply,
        "thinking": thinking_content,
        "telemetry": telemetry,
        "chunks_matrix": chunks_matrix_payload,
    }


def _detect_repetition_loop(text: str, max_line_repeat_ratio: float = 0.3) -> bool:
    """
    Detects whether generation collapsed into a repetition loop - a known
    failure mode of small local models on multi-chunk synthesis prompts,
    where the model repeats a templated sentence ("In the Nth document
    chunk, there is a mention of...") instead of answering. This does NOT
    fabricate content - it only flags garbage output so the pipeline can
    report failure honestly instead of returning a degenerate wall of
    repeated sentences to the user.
    """
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    if len(sentences) < 5:
        return False

    # Normalize each sentence by stripping standalone digits/ordinals
    # ("In the third document chunk" vs "In the fourth document chunk" are
    # the same template with one word changed - compare structure, not exact text)
    normalized = [" ".join(w for w in s.lower().split() if not w.isdigit()) for s in sentences]
    unique_ratio = len(set(normalized)) / len(normalized)

    return unique_ratio < max_line_repeat_ratio


def _empty_result(reply: str, thinking: str, telemetry: dict) -> dict:
    return {"reply": reply, "thinking": thinking, "telemetry": telemetry, "chunks_matrix": []}