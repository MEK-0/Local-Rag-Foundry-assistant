import re
import json
from typing import List, Dict, Any

from src.llm_client import generate_chat_response
from src.db import upsert_entity, upsert_edge, get_children

ENTITY_EXTRACTION_PROMPT = """Extract 2 to 8 key technical terms, part names, or \
named concepts from the text below. Output ONLY a JSON array of short strings.

Example output: ["Emergency Stop", "SCARA", "600 hours"]

If there are no meaningful technical terms, output exactly: []

Do not repeat the word "JSON" or "array" in your output. Do not add any text \
before or after the array.

Text:
{text}
"""

MIN_SECTION_LENGTH_FOR_EXTRACTION = 100


def extract_entities_from_text(text: str) -> List[str]:
    """
    Extracts a small set of key entities/terms from a block of text using
    the configured chat model. Falls back to an empty list on any failure -
    entity extraction is a best-effort enrichment step, never allowed to
    break ingestion.

    Handles two known small-model failure modes seen in practice:
      - Repetition loop (e.g. "JSON array: []\\n\\nJSON array: []\\n..." repeated
        dozens of times) - detected and treated as "no entities found".
      - JSON array embedded inside extra text despite instructions - extracted
        via regex instead of requiring the whole response to be valid JSON.
    """
    if not text or len(text.strip()) < MIN_SECTION_LENGTH_FOR_EXTRACTION:
        return []

    prompt = ENTITY_EXTRACTION_PROMPT.format(text=text[:2000])

    try:
        raw_response = generate_chat_response(prompt, max_tokens=150)
    except Exception:
        return []

    if not raw_response or "error during generation" in raw_response.lower():
        return []

    # Repetition-loop guard: if the same short line repeats many times,
    # treat it as a failed/degenerate generation rather than parsing it.
    lines = [l.strip() for l in raw_response.strip().split("\n") if l.strip()]
    if len(lines) > 5 and len(set(lines)) <= 2:
        return []

    # Extract the first [...] block from the response instead of requiring
    # the entire response to be valid JSON - small models often add stray
    # text around the array despite instructions not to.
    match = re.search(r"\[.*?\]", raw_response, re.DOTALL)
    if not match:
        return []

    try:
        entities = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return []

    if not isinstance(entities, list):
        return []

    return [str(e).strip() for e in entities if isinstance(e, (str, int, float)) and str(e).strip()][:8]


def build_graph_for_section(doc_id: str, section_node_id: str, section_children: List[Dict[str, Any]]) -> None:
    combined_text = "\n".join(
        (child.get("content") or "").strip()
        for child in section_children
        if child.get("content")
    )
    if not combined_text:
        return

    entity_names = extract_entities_from_text(combined_text)
    if len(entity_names) < 2:
        return

    entity_ids = [
        upsert_entity(doc_id=doc_id, name=name, node_id=section_node_id)
        for name in entity_names
    ]
    entity_ids = sorted(set(entity_ids))

    for i in range(len(entity_ids)):
        for j in range(i + 1, len(entity_ids)):
            upsert_edge(doc_id=doc_id, source_entity_id=entity_ids[i], target_entity_id=entity_ids[j])


def build_document_graph(doc_id: str, section_nodes: List[Dict[str, Any]]) -> int:
    processed = 0
    total = len(section_nodes)
    for section in section_nodes:
        section_id = section["id"] if hasattr(section, "get") else section.id
        children = get_children(section_id)
        build_graph_for_section(doc_id=doc_id, section_node_id=section_id, section_children=children)
        processed += 1
        print(f"      entity graph: section {processed}/{total} processed", end="\r")
    print()
    return processed