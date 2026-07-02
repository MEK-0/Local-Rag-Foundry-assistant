from src.llm_client import get_embedding, generate_chat_response
from src.retrieval import retrieve_top_k

def process_chat_query(query: str) -> dict:
    query_embedding = get_embedding(query)
    top_chunks = retrieve_top_k(query_embedding)
    
    if not top_chunks:
        return {
            "reply": "I couldn't find any relevant information in my knowledge base.",
            "thinking": "No matching document chunks found in the database."
        }
        
    context_text = ""
    for idx, chunk in enumerate(top_chunks):
        context_text += f"--- Document {idx+1} (Source: {chunk['source_file']}) ---\n"
        context_text += f"{chunk['chunk_text']}\n\n"
        
    # Instruct the model to strictly follow the structural template format
    prompt = f"""You are an industrial automation expert. Review the provided context to answer the user's question.

CRITICAL RULE 1: You must output your internal analytical steps inside [THINKING] and [/THINKING] tags.
CRITICAL RULE 2: After the [/THINKING] tag, you must strictly structure your response using the following template format if the answer is found. Do not use conversational filler words.

[THINKING]
(Write your internal thinking process here)
[/THINKING]

Summary: (Provide a concise one-sentence summary of the solution)

Safety Warnings:
- (List explicit safety warnings or pre-requisites here)

Step-by-step Guidance:
1. (Provide clear numbered operational steps based ONLY on the context)

Reference:
- (List the source file or document section reference here)

If the context does not contain the answer to the question, ignore the template above and strictly reply with 'I don't know based on my documents.' after the [/THINKING] tag.

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:"""

    raw_response = generate_chat_response(prompt)
    
    thinking_content = ""
    reply_content = raw_response
    
    # Convert the response to lowercase for robust, case-insensitive string parsing
    raw_lower = raw_response.lower()
    
    # Identify closing tag variations ([/thinking] or [/THINKING])
    split_trigger = None
    if "[/thinking]" in raw_lower:
        if "[/thinking]" in raw_response:
            split_trigger = "[/thinking]"
        elif "[/THINKING]" in raw_response:
            split_trigger = "[/THINKING]"
        else:
            # Dynamically capture structural anomalies using index offsets
            start_pos = raw_lower.find("[/thinking]")
            split_trigger = raw_response[start_pos:start_pos+11]

    if split_trigger:
        try:
            start_idx = raw_response.find(split_trigger)
            end_idx = start_idx + len(split_trigger)
            
            # Extract thinking steps and strip formatting tokens
            thinking_part = raw_response[:start_idx]
            thinking_content = thinking_part.replace("[THINKING]", "").replace("[thinking]", "").strip()
            
            # Extract the structured final user answer
            reply_content = raw_response[end_idx:].strip()
        except Exception:
            pass
    else:
        # FALLBACK MECHANISM: If tags are missing, segment text via anticipated template headers
        for header in ["summary:", "safety warnings:", "step-by-step guidance:"]:
            if header in raw_lower:
                header_idx = raw_lower.find(header)
                thinking_content = raw_response[:header_idx].strip()
                reply_content = raw_response[header_idx:].strip()
                break

    # FAIL-SAFE GUARDRAIL: If parsing yields empty content, preserve raw output to prevent UI lock
    if not reply_content or len(reply_content.strip()) < 5:
        reply_content = raw_response
        thinking_content = "Thinking structure processed implicitly by the local model."
            
    return {
        "reply": reply_content,
        "thinking": thinking_content
    }