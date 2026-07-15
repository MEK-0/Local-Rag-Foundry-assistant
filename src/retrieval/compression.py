import re
from typing import List, Dict, Any

from src.db import get_node, get_children

ATOMIC_NODE_TYPES = {"table", "figure", "warning", "note", "code"}

# Cap on reconstructed section length (characters). Full child-node
# reconstruction pulls every sibling under the same parent - without a
# cap, a long section (e.g. a 10-paragraph maintenance chapter) would
# blow up prompt size for a single chunk. This keeps it bounded while
# still giving far more surrounding context than a bare heading lookup.
MAX_EXPANDED_SECTION_CHARS = 1500


def compress_context_chunks(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Same responsibilities as before (sentence-window compression +
    parent-context attachment), now with full child-node section
    reconstruction instead of a heading-only lookup.
    """
    compressed_chunks = []
    query_words = [w.lower() for w in query.split() if len(w) > 3]

    for chunk in chunks:
        node_type = chunk.get("node_type", "paragraph")

        if node_type in ATOMIC_NODE_TYPES:
            _attach_parent_context(chunk)
            compressed_chunks.append(chunk)
            continue

        if chunk.get("page_number") in (1, 2, "1", "2"):
            lines = chunk["chunk_text"].split("\n")
            clean_lines = [
                line.strip() for line in lines
                if line.strip()
                and not any(noise in line.lower() for noise in
                            ["no part of this", "subject to change", "all rights"])
            ]
            chunk["chunk_text"] = "\n".join(clean_lines)
            _attach_parent_context(chunk)
            compressed_chunks.append(chunk)
            continue

        if chunk.get("rerank_score", 0.0) > 1.5:
            _attach_parent_context(chunk)
            compressed_chunks.append(chunk)
            continue

        sentences = re.split(r"(?<=[.!?])\s+", chunk["chunk_text"])
        relevant_indices = set()
        for idx, sentence in enumerate(sentences):
            if any(word in sentence.lower() for word in query_words):
                relevant_indices.add(idx)
                if idx > 0:
                    relevant_indices.add(idx - 1)
                if idx < len(sentences) - 1:
                    relevant_indices.add(idx + 1)

        if relevant_indices:
            sorted_indices = sorted(relevant_indices)
            chunk["chunk_text"] = " ".join(sentences[i] for i in sorted_indices)
            _attach_parent_context(chunk)
            compressed_chunks.append(chunk)

    return compressed_chunks


def _attach_parent_context(chunk: Dict[str, Any]) -> None:
    """
    Full parent-context expansion: instead of only fetching the parent's
    heading text, this reconstructs the surrounding section by pulling
    all sibling nodes under the same parent (get_children) and joining
    their content in document order. This gives the LLM the actual
    surrounding paragraphs/tables of a section, not just its title -
    e.g. a chunk that only contains "...apply after 600 hours" now
    arrives with the preceding sentence that names the grease type,
    even if that sentence landed in a different chunk during splitting.

    Adds two fields to chunk:
      - expanded_heading: the parent section's heading/title text
      - expanded_section: reconstructed sibling content, capped at
        MAX_EXPANDED_SECTION_CHARS
    """
    parent_id = chunk.get("parent_id")
    if not parent_id:
        chunk["expanded_heading"] = chunk.get("heading_path", "")
        chunk["expanded_section"] = ""
        return

    parent_node = get_node(parent_id)
    if not parent_node:
        chunk["expanded_heading"] = chunk.get("heading_path", "")
        chunk["expanded_section"] = ""
        return

    chunk["expanded_heading"] = parent_node.get("heading_path") or parent_node.get("content", "")

    siblings = get_children(parent_id)
    section_parts = []
    total_len = 0
    for sibling in siblings:
        content = (sibling.get("content") or "").strip()
        if not content:
            continue
        if total_len + len(content) > MAX_EXPANDED_SECTION_CHARS:
            break
        section_parts.append(content)
        total_len += len(content)

    chunk["expanded_section"] = "\n".join(section_parts)