import re
from src.llm_client import get_embedding, generate_chat_response
from src.retrieval.query_rewriter import rewrite_query
from src.retrieval.hybrid import hybrid_retrieve
from src.retrieval.reranker import rerank_chunks
from src.retrieval.grader import grade_retrieved_chunks
from src.retrieval.compression import compress_context_chunks

def process_chat_query(query: str) -> dict:
    # --- PHASE 1: QUERY REWRITING / EXPANSION ---
    expanded_queries = rewrite_query(query)
    
    # --- PHASE 2: MULTI-QUERY HYBRID RETRIEVAL ---
    all_candidate_chunks = []
    seen_chunk_ids = set()
    
    for q_track in expanded_queries:
        q_emb = get_embedding(q_track)
        retrieved_candidates = hybrid_retrieve(query_text=q_track, query_embedding=q_emb, top_k=6)
        
        for chunk in retrieved_candidates:
            if chunk["id"] not in seen_chunk_ids:
                seen_chunk_ids.add(chunk["id"])
                all_candidate_chunks.append(chunk)
                
    if not all_candidate_chunks:
        return {
            "reply": "Summary: I couldn't find any relevant verification parameters in my current database.",
            "thinking": "No matching tokens across multiple expanded hybrid retrieval query paths."
        }
        
    # --- PHASE 3: CROSS-ENCODER RE-RANKING ---
    reranked_candidates = rerank_chunks(query=query, chunks=all_candidate_chunks, top_n=5)
    
    # --- PHASE 4: RETRIEVAL GRADER ---
    graded_candidates = grade_retrieved_chunks(query=query, chunks=reranked_candidates)
    
    # --- PHASE 5: CONTEXT COMPRESSION ---
    final_context_chunks = compress_context_chunks(query=query, chunks=graded_candidates)
    
    if not final_context_chunks:
        return {
            "reply": "Summary: Relevant document files were discovered, but they did not pass safety relevance validation thresholds.",
            "thinking": "All chunks failed semantic safety boundaries during grading iterations."
        }

    # --- PHASE 6: CONTEXT COMPILING ---
    context_text = ""
    source_files = set()
    for idx, chunk in enumerate(final_context_chunks):
        context_text += f"--- Document {idx+1} (Source: {chunk['source_file']}, Page: {chunk['page_number']}) ---\n"
        context_text += f"{chunk['chunk_text']}\n\n"
        if "source_file" in chunk:
            source_files.add(chunk["source_file"])
        
    prompt = f"""You are an industrial automation expert. Answer the user's question directly, accurately, and concisely using ONLY the provided context. 

Provide a direct 1-2 sentence answer focusing strictly on exact technical specifications, part names, or hours. Do not use conversational preambles, introductory thoughts, or duplicate sentences.

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:"""

    raw_response = generate_chat_response(prompt)
    
    # --- ADVANCED POST-PROCESSING REGEX GATES ---
    clean_reply = raw_response.strip()
    
    # Isolate and extract all the conversational fluff sentences ("Okay, let's...", "First, I need...", etc.)
    fluff_patterns = [
        r"okay,\s*let.*?\.", r"first,\s*i\s*need.*?\.", r"let's\s*see.*?\.", 
        r"the\s*user\s*is\s*asking.*?\.", r"i\s*need\s*to\s*confirm.*?\.",
        r"it's\s*possible\s*that.*?\.", r"however,\s*the\s*answer.*?\."
    ]
    
    extracted_fluff = []
    for pattern in fluff_patterns:
        matches = re.findall(pattern, clean_reply, re.IGNORECASE)
        for match in matches:
            extracted_fluff.append(match)
            clean_reply = clean_reply.replace(match, "").strip()

    # If the model explicitly wrote an "Answer:" block at the end, isolate just that supreme line
    answer_block_match = re.search(r'(?:answer):\s*(.*)', clean_reply, re.IGNORECASE)
    if answer_block_match:
        final_summary = answer_block_match.group(1).strip()
    else:
        # Otherwise, take the cleanest remaining line that isn't empty
        lines = [line.strip() for line in clean_reply.split("\n") if len(line.strip()) > 10 and not line.startswith("**")]
        final_summary = lines[-1] if lines else clean_reply

    # Clean up leaking markdown tokens or leftover headers
    final_summary = re.sub(r'\*\*Answer:\*\*\s*', '', final_summary, flags=re.IGNORECASE).strip()

    # Thinking content telemetry packaging
    thinking_content = (
        f"Expanded Queries Used: {expanded_queries}\n\n"
        f"Pipeline Trace: Filtered {len(all_candidate_chunks)} candidates down to {len(final_context_chunks)} graded chunks.\n\n"
        f"Extracted Thought Fluff: {extracted_fluff}"
    )

    references_string = ", ".join(source_files) if source_files else "Local Knowledge Base"
    
    # Construct the final crystal-clean corporate response layout
    structured_reply = (
        f"Summary: {final_summary}\n\n"
        f"Safety Warnings:\n- Always cross-reference the extracted values with primary physical engineering schematics before execution.\n\n"
        f"Step-by-step Guidance:\n1. Open the referenced automated technical system manuals.\n2. Apply the validated specification parameters directly: {final_summary}\n\n"
        f"Reference:\n- {references_string}"
    )
            
    return {
        "reply": structured_reply,
        "thinking": thinking_content
    }