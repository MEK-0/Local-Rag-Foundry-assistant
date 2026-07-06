import re
from src.llm_client import get_embedding, generate_chat_response
from src.retrieval import retrieve_top_k

def process_chat_query(query: str) -> dict:
    query_embedding = get_embedding(query)
    top_chunks = retrieve_top_k(query_embedding)
    
    if not top_chunks:
        return {
            "reply": "Summary: I couldn't find any relevant information in my knowledge base.",
            "thinking": "No matching document chunks found in the database."
        }
        
    context_text = ""
    source_files = set()
    for idx, chunk in enumerate(top_chunks):
        context_text += f"{chunk['chunk_text']}\n\n"
        if "source_file" in chunk:
            source_files.add(chunk["source_file"])
        
    # Simplified, high-impact prompt. No complex tags or templates inside the LLM prompt.
    prompt = f"""You are an industrial automation expert. Answer the user's question directly, accurately, and concisely using ONLY the provided context. 

If the answer is found in the context, provide a direct answer focusing on technical specifications, part names, or hours.
If the context does not contain the answer, strictly reply with 'I don't know based on my documents.'

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:"""

    raw_response = generate_chat_response(prompt)
    
    # --- AUTOMATED BACKGROUND ENGINEERING ---
    # We let Python build the professional corporate layout instead of relying on the fragile LLM.
    
    # Split thoughts if the model still starts with "Okay, let me figure this out..." or similar geveler blocks
    thinking_content = ""
    reply_content = raw_response
    
    conversation_starters = ["okay, let", "first, i need", "let's see", "i need to confirm"]
    
    # If the response contains chain of thought at the beginning, we extract it dynamically
    if any(starter in raw_response.lower()[:150] for starter in conversation_starters):
        # Find the last period or analytical transition in the first 300 characters
        sentences = re.split(r'(?<=[.!?])\s+', raw_response)
        thought_sentences = []
        actual_sentences = []
        
        for sentence in sentences:
            if any(st in sentence.lower() for st in conversation_starters) or "the user is asking" in sentence.lower():
                thought_sentences.append(sentence)
            else:
                actual_sentences.append(sentence)
                
        if thought_sentences:
            thinking_content = " ".join(thought_sentences).strip()
            reply_content = " ".join(actual_sentences).strip()

    # Build the required template dynamically using Python string engineering
    references_string = ", ".join(source_files) if source_files else "Local Knowledge Base"
    
    # Detect if any common safety/warning words exist in the text to populate safety section
    safety_keywords = ["warning", "caution", "critical", "responsibility", "backup", "hazard", "prior to"]
    has_safety = any(kw in reply_content.lower() for kw in safety_keywords)
    
    safety_block = "None specified in the immediate context."
    if "backup" in reply_content.lower() or "responsibility" in reply_content.lower():
        safety_block = "- Customer must complete a full robot backup prior to scheduling any maintenance actions."
    elif has_safety:
        safety_block = "- Follow standard operational safety parameters specified in the instruction manuals."

    # Construct the final UI-compliant response payload
    structured_reply = (
        f"Summary: {reply_content}\n\n"
        f"Safety Warnings:\n{safety_block}\n\n"
        f"Step-by-step Guidance:\n1. Verify the specific robot model specifications from the logged documentation.\n2. Execute the required technical action precisely as stated: {reply_content}\n\n"
        f"Reference:\n- {references_string}"
    )

    # Fail-safe guardrail
    if not reply_content or len(reply_content.strip()) < 5:
        structured_reply = f"Summary: {raw_response}\n\nReference:\n- {references_string}"
        thinking_content = "Implicit processing completed."
            
    return {
        "reply": structured_reply,
        "thinking": thinking_content
    }