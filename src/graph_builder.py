import json
from typing import List, Dict, Any

from src.llm_client import generate_chat_response
from src.db import upsert_entity, upsert_edge, get_children

ENTITY_EXTRACTION_PROMPT = """Extract the key technical terms, concepts, part names, \
or named entities mentioned in the following section of a technical document. \
Return between 2 and 8 entities - only the ones that matter (e.g. component \
names, safety terms, model numbers, procedures) - not generic words.

Rules:
- Output ONLY a JSON array of strings, nothing else.
- No explanations, no markdown code fences.
- If there are no meaningful entities, output an empty array: []

Section text:
{text}

JSON array:"""


def extract_entities_from_text(text: str) -> List[str]:
    """
    Extracts a small set of key entities/terms from a block of text using
    the configured chat model. Falls back to an empty list on any failure
    (parsing error, generation error, empty response) - entity extraction
    is a best-effort enrichment step, not something that should ever break
    ingestion.
    """
    if not text or len(text.strip()) < 20:
        return []

    prompt = ENTITY_EXTRACTION_PROMPT.format(text=text[:2000])  # cap input size

    try:
        raw_response = generate_chat_response(prompt)
    except Exception:
        return []

    if not raw_response or "error during generation" in raw_response.lower():
        return []

    cleaned = raw_response.strip().strip("`")
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()

    try:
        entities = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return []

    if not isinstance(entities, list):
        return []

    return [str(e).strip() for e in entities if isinstance(e, (str, int, float)) and str(e).strip()]


def build_graph_for_section(doc_id: str, section_node_id: str, section_children: List[Dict[str, Any]]) -> None:
    """
    Extracts entities from a section's combined text (its child nodes -
    paragraphs, tables, etc.) and records them plus co-occurrence edges
    between every pair found in the same section.

    One LLM call per section rather than per entity-pair keeps ingestion
    cost bounded by section count, not by the combinatorial explosion of
    entity relationships - this is the key cost tradeoff vs. full GraphRAG
    relationship-typing (e.g. "related_to" / "located_in"), traded for a
    much cheaper "co-occurs_with" signal that still supports graph
    traversal and visualization.
    """
    combined_text = "\n".join(
        (child.get("content") or "").strip()
        for child in section_children
        if child.get("content")
    )
    if not combined_text:
        return

    entity_names = extract_entities_from_text(combined_text)
    if len(entity_names) < 2:
        return  # need at least 2 entities to form an edge

    entity_ids = [
        upsert_entity(doc_id=doc_id, name=name, node_id=section_node_id)
        for name in entity_names
    ]
    entity_ids = sorted(set(entity_ids))  # dedup + stable order for edge direction

    # Co-occurrence edges: every pair of entities found in this section.
    # Sorting ids ensures (a, b) and (b, a) never both get inserted -
    # upsert_edge's UNIQUE(source, target) constraint relies on this.
    for i in range(len(entity_ids)):
        for j in range(i + 1, len(entity_ids)):
            upsert_edge(doc_id=doc_id, source_entity_id=entity_ids[i], target_entity_id=entity_ids[j])


def build_document_graph(doc_id: str, section_nodes: List[Dict[str, Any]]) -> int:
    """
    Runs build_graph_for_section() for every section node in a document.
    section_nodes should be the SECTION-type nodes from the parsed tree.
    Returns the number of sections processed (for ingest reporting).
    """
    processed = 0
    for section in section_nodes:
        children = get_children(section["id"] if hasattr(section, "get") else section.id)
        build_graph_for_section(
            doc_id=doc_id,
            section_node_id=section["id"] if hasattr(section, "get") else section.id,
            section_children=children,
        )
        processed += 1
    return processed