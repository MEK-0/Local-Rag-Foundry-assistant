from src.llm_client import get_embedding, generate_chat_response
from src.retrieval import retrieve_top_k

def process_chat_query(query: str) -> str:
    """
    Orchestrates the Retrieval-Augmented Generation pipeline.
    """
    # 1. Convert user query to vector
    query_embedding = get_embedding(query)
    
    # 2. Retrieve the most relevant documents
    top_chunks = retrieve_top_k(query_embedding)
    
    if not top_chunks:
        return "I couldn't find any relevant information in my knowledge base."
        
    # 3. Build the context string from retrieved chunks
    context_text = ""
    for idx, chunk in enumerate(top_chunks):
        context_text += f"--- Document {idx+1} (Source: {chunk['source_file']}) ---\n"
        context_text += f"{chunk['chunk_text']}\n\n"
        
    # 4. Construct the final prompt (Augmentation)
    prompt = f"""Use the following retrieved context to answer the user's question. 
If the answer is not contained in the context, just say "I don't know based on my documents." Do not guess.

CONTEXT:
{context_text}

QUESTION:
{query}

ANSWER:"""

    # 5. Generate and return the final response
    return generate_chat_response(prompt)