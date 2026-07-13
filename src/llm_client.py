from openai import OpenAI, AzureOpenAI
from src.config import settings
from typing import List

# Embedding model identity - tracked so ingest.py / db.py can later detect a
# mode switch that would produce incompatible vector dimensions (local
# MiniLM = 384-dim, Azure embeddings = 1536-dim). Mixing them in the same
# DB breaks cosine_similarity in hybrid.py with a silent shape-mismatch crash.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" if settings.mode == "local" else settings.azure_openai_deployment

if settings.mode == "local":
    client = OpenAI(
        base_url=settings.foundry_base_url,
        api_key="not-needed-for-local"
    )

    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
else:
    client = AzureOpenAI(
        api_version=settings.azure_openai_api_version,
        azure_endpoint=settings.azure_openai_endpoint or "",
        api_key=settings.azure_openai_key or ""
    )


def get_embedding(text: str) -> List[float]:
    """
    Generates a vector embedding for the provided text.
    """
    clean_text = text.replace("\n", " ")

    if settings.mode == "local":
        return embedder.encode(clean_text).tolist()
    else:
        response = client.embeddings.create(
            input=clean_text,
            model=settings.azure_openai_deployment
        )
        return response.data[0].embedding

def generate_chat_response(prompt: str) -> str:
    """
    Sends the augmented prompt to the selected LLM and returns the response.
    """
    if settings.mode == "local":
        model_name = settings.foundry_chat_model
    else:
        model_name = settings.azure_openai_deployment

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": f"You are an intelligent offline AI assistant.\n\n{prompt}"}
            ],
            temperature=settings.generation_temperature,
            max_tokens=settings.generation_max_tokens,
            # Small local models (e.g. qwen3-0.6b) are prone to repetition
            # loops on multi-chunk synthesis tasks. frequency_penalty
            # discourages reusing the same tokens repeatedly;
            # presence_penalty discourages reusing the same phrases/topics.
            # Not all OpenAI-compatible local servers honor these - if the
            # loop persists even with this set, the model itself is the
            # limiting factor (see note in rag_pipeline.py).
            frequency_penalty=0.4,
            presence_penalty=0.3,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during generation: {str(e)}"