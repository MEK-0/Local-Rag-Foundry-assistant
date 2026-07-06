import re
import time
from src.llm_client import get_embedding, generate_chat_response
from src.retrieval.query_rewriter import rewrite_query
from src.retrieval.hybrid import hybrid_retrieve
from src.retrieval.reranker import rerank_chunks
from src.retrieval.grader import grade_retrieved_chunks
from src.retrieval.compression import compress_context_chunks

def process_chat_query(query: str) -> dict:
    """
    Orchestrates the advanced offline RAG pipeline while generating 
    granular telemetry tracking for the frontend observability panels.
    """
    telemetry = {}
    
    # --- PHASE 1: QUERY REWRITING ---
    start_time = time.perf_counter()
    expanded_queries = rewrite_query(query)
    # If the user toggles off advanced mode, we can bypass extension layers if needed,
    # but to support the UI toggle natively, we track the full stack execution.
    telemetry["query_expansion_time_ms"] = round((time.perf_counter() - start_time) * 1000, 1)

    # --- PHASE 2: EMBEDDING GENERATION ---
    start_time = time.perf_counter()
    query_embedding = get_embedding(query)
    telemetry["embedding_time_ms"] = round((time.perf_counter() - start_time) * 1000, 1)
    
    # --- PHASE 3: HYBRID RETRIEVAL ---
    start_time = time.perf_counter()
    all_candidate_chunks = []
    seen_chunk_ids = set()
    
    for q_track in expanded_queries:
        retrieved_candidates = hybrid_retrieve(query_text=q_track, query_embedding=query_embedding, top_k=6)
        for chunk in retrieved_candidates:
            if chunk["id"] not in seen_chunk_ids:
                seen_chunk_ids.add(chunk["id"])
                all_candidate_chunks.append(chunk)
    telemetry["retrieval_time_ms"] = round((time.perf_counter() - start_time) * 1000, 1)
                
    if not all_candidate_chunks:
        return {
            "reply": "Summary: I couldn't find any relevant verification parameters in my current database.",
            "thinking": f"Expanded Queries Used: {expanded_queries}\nNo chunks discovered during retrieval.",
            "telemetry": telemetry,
            "chunks_matrix": []
        }
        
    # --- PHASE 4: CROSS-ENCODER RE-RANKING ---
    start_time = time.perf_counter()
    reranked_candidates = rerank_chunks(query=query, chunks=all_candidate_chunks, top_n=5)
    telemetry["rerank_time_ms"] = round((time.perf_counter() - start_time) * 1000, 1)
    
    # --- PHASE 5: RETRIEVAL GRADER & COMPRESSION ---
    start_time = time.perf_counter()
    graded_candidates = grade_retrieved_chunks(query=query, chunks=reranked_candidates)
    final_context_chunks = compress_context_chunks(query=query, chunks=graded_candidates)
    telemetry["culling_and_compression_time_ms"] = round((time.perf_counter() - start_time) * 1000, 1)
    
    if not final_context_chunks:
        return {
            "reply": "Summary: Documents were found but failed the semantic safety relevance thresholds.",
            "thinking": "All candidate chunks were pruned during the grading phase.",
            "telemetry": telemetry,
            "chunks_matrix": []
        }

    # --- PHASE 6: CONTEXT COMPILING ---
    context_text = ""
    source_files = set()
    chunks_matrix_payload = []
    
    for idx, chunk in enumerate(final_context_chunks):
        context_text += f"--- Document {idx+1} (Source: {chunk['source_file']}, Page: {chunk['page_number']}) ---\n"
        context_text += f"{chunk['chunk_text']}\n\n"
        if "source_file" in chunk:
            source_files.add(chunk["source_file"])
            
        # Packaging explainability matrix for JavaScript extraction loops
        chunks_matrix_payload.append({
            "id": chunk["id"],
            "source": chunk["source_file"],
            "page": chunk["page_number"],
            "rerank_score": round(chunk.get("rerank_score", 0.0), 3),
            "rrf_score": round(chunk.get("rrf_score", 0.0), 4)
        })
        
    prompt = f"""You are an industrial automation expert. Answer the user's question directly, accurately, and concisely using ONLY the provided context. 

Provide a direct answer focusing strictly on exact technical specifications, part names, or hours. Do not use conversational preambles.

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:"""

    # --- PHASE 7: TOKEN GENERATION ---
    start_time = time.perf_counter()
    raw_response = generate_chat_response(prompt)
    telemetry["generation_time_ms"] = round((time.perf_counter() - start_time) * 1000, 1)
    
    # Post-processing to clean up raw conversational leakages
    clean_reply = raw_response.strip()
    fluff_patterns = [
        r"okay,\s*let.*?\.", r"first,\s*i\s*need.*?\.", r"let's\s*see.*?\.", 
        r"the\s*user\s*is\s*asking.*?\.", r"i\s*need\s*to\s*confirm.*?\."
    ]
    for pattern in fluff_patterns:
        clean_reply = re.sub(pattern, "", clean_reply, flags=re.IGNORECASE).strip()

    answer_block_match = re.search(r'(?:answer):\s*(.*)', clean_reply, re.IGNORECASE)
    if answer_block_match:
        final_summary = answer_block_match.group(1).strip()
    else:
        lines = [line.strip() for line in clean_reply.split("\n") if len(line.strip()) > 10 and not line.startswith("**")]
        final_summary = lines[-1] if lines else clean_reply

    final_summary = re.sub(r'\*\*Answer:\*\*\s*', '', final_summary, flags=re.IGNORECASE).strip()
    references_string = ", ".join(source_files) if source_files else "Local Knowledge Base"
    
    # High-grade structured corporate response package
    structured_reply = (
        f"Summary: {final_summary}\n\n"
        f"Safety Warnings:\n- Always cross-reference extracted values with system schematics prior to maintenance scheduling.\n\n"
        f"Step-by-step Guidance:\n1. Open the referenced automated technical systems engineering manual.\n2. Apply the verified parameter action directly: {final_summary}\n\n"
        f"Reference:\n- {references_string}"
    )

    # Dynamic explanation logs for the active UI accordion
    thinking_content = (
        f"Expanded Queries Loaded:\n" + "\n".join([f"↳ {q}" for q in expanded_queries]) + "\n\n"
        f"Pipeline Search Trace:\n"
        f"✔ Hybrid multi-query search generated {len(all_candidate_chunks)} candidate partitions.\n"
        f"✔ Cross-Encoder evaluated and preserved the top {len(final_context_chunks)} high-density fragments."
    )
            
    return {
        "reply": structured_reply,
        "thinking": thinking_content,
        "telemetry": telemetry,
        "chunks_matrix": chunks_matrix_payload
    }