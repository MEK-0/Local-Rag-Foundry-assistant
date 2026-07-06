import re
import time
from src.llm_client import get_embedding, generate_chat_response
from src.retrieval.query_rewriter import rewrite_query
from src.retrieval.hybrid import hybrid_retrieve
from src.retrieval.reranker import rerank_chunks
from src.retrieval.grader import grade_retrieved_chunks
from src.retrieval.compression import compress_context_chunks

def process_chat_query(query: str) -> dict:
    telemetry = {}
    
    # --- PHASE 1: QUERY REWRITING ---
    start_time = time.perf_counter()
    expanded_queries = rewrite_query(query)
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
            "reply": "Summary: No relevant engineering parameters discovered in the local database tracks.",
            "thinking": "No chunks found during cross-query extraction execution.",
            "telemetry": telemetry,
            "chunks_matrix": []
        }
        
    # --- PHASE 4: CROSS-ENCODER RE-RANKING ---
    start_time = time.perf_counter()
    reranked_candidates = rerank_chunks(query=query, chunks=all_candidate_chunks, top_n=5)
    telemetry["rerank_time_ms"] = round((time.perf_counter() - start_time) * 1000, 1)
    
    # --- PHASE 5: RETRIEVAL GRADER (De-duplication triggers here) ---
    start_time = time.perf_counter()
    graded_candidates = grade_retrieved_chunks(query=query, chunks=reranked_candidates)
    final_context_chunks = compress_context_chunks(query=query, chunks=graded_candidates)
    telemetry["culling_and_compression_time_ms"] = round((time.perf_counter() - start_time) * 1000, 1)
    
    if not final_context_chunks:
        return {
            "reply": "Summary: Context mapping traces located, but they failed consistency security checks.",
            "thinking": "All components failed semantic overlap grading filters.",
            "telemetry": telemetry,
            "chunks_matrix": []
        }

    # --- PHASE 6: CONTEXT COMPILING ---
    context_text = ""
    source_files = set()
    chunks_matrix_payload = []
    
    for idx, chunk in enumerate(final_context_chunks):
        context_text += f"--- Context Block {idx+1} (Source: {chunk['source_file']}, Page: {chunk['page_number']}) ---\n"
        context_text += f"{chunk['chunk_text']}\n\n"
        if "source_file" in chunk:
            source_files.add(chunk["source_file"])
            
        chunks_matrix_payload.append({
            "id": chunk["id"],
            "source": chunk["source_file"],
            "page": chunk["page_number"],
            "rerank_score": round(chunk.get("rerank_score", 0.0), 3),
            "rrf_score": round(chunk.get("rrf_score", 0.0), 4)
        })
        
    # CRITICAL: Highly explicit, non-conversational, list-preserving industrial prompt
    prompt = f"""You are a precise industrial automation engineering assistant. 
Answer the user's question directly, comprehensively, and accurately using ONLY the provided context blocks.

CRITICAL INSTRUCTIONS:
1. If the answer involves a list of series, models, part numbers, or subsections, output EVERY single item exactly as it appears in the text. Do not summarize or use generic placeholders.
2. Do not include introductory phrases like "Based on the provided context...", "According to the handbook...", or "Okay, let me figure this out...".
3. Start your response directly with the answer data. Do not duplicate information.

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:"""

    # --- PHASE 7: TOKEN GENERATION ---
    start_time = time.perf_counter()
    raw_response = generate_chat_response(prompt)
    telemetry["generation_time_ms"] = round((time.perf_counter() - start_time) * 1000, 1)
    
    # --- REFINED POST-PROCESSING ---
    clean_reply = raw_response.strip()
    
    # Prune any accidental meta tag generation or leftover template artifacts
    clean_reply = re.sub(r'^(Based on the context|According to the text|Answer:)\s*', '', clean_reply, flags=re.IGNORECASE).strip()

    references_string = ", ".join(source_files) if source_files else "Local Knowledge Base"
    
    # Pure structured output format without artificial runbook text bloat
    structured_reply = (
        f"Summary:\n{clean_reply}\n\n"
        f"Reference:\n- {references_string}"
    )

    thinking_content = (
        f"Expanded Queries Loaded:\n" + "\n".join([f"↳ {q}" for q in expanded_queries]) + "\n\n"
        f"Pipeline Search Trace:\n"
        f"✔ Combined search tracks produced {len(all_candidate_chunks)} candidates.\n"
        f"✔ Similarity filter and Cross-Encoder selected {len(final_context_chunks)} unique context frames."
    )
            
    return {
        "reply": structured_reply,
        "thinking": thinking_content,
        "telemetry": telemetry,
        "chunks_matrix": chunks_matrix_payload
    }