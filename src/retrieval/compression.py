import re
from typing import List, Dict, Any

from src.db import get_node, get_children

# Node types that are already atomic units (a full table, a full warning box,
# a figure caption). Sentence-window pruning on these would cut a table row
# in half or drop the second sentence of a warning — never prune them.
ATOMIC_NODE_TYPES = {"table", "figure", "warning", "note", "code"}


def compress_context_chunks(query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Two responsibilities:
      1. Sentence-window compression for long paragraph chunks (unchanged
         behavior from before, but now gated by node_type instead of a
         page-number heuristic).
      2. Parent-child expansion: for each surviving chunk, walk up to its
         parent node (heading/section) and attach the parent's heading text
         as `expanded_heading`, so the LLM sees which section the fact came
         from even if the chunk itself doesn't repeat the heading.
    """
    compressed_chunks = []
    query_words = [w.lower() for w in query.split() if len(w) > 3]

    for chunk in chunks:
        node_type = chunk.get("node_type", "paragraph")

        # Atomic nodes (table/figure/warning/note/code): never sentence-prune,
        # pass through as-is so the full table/warning stays intact
        if node_type in ATOMIC_NODE_TYPES:
            _attach_parent_context(chunk)
            compressed_chunks.append(chunk)
            continue

        # Cover-page noise cleanup (copyright boilerplate, doc numbers) —
        # kept for pages 1-2 regardless of node_type
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

        # High-confidence cross-encoder hits: keep raw text intact so list
        # structure / numbering isn't broken by sentence-window slicing
        if chunk.get("rerank_score", 0.0) > 1.5:
            _attach_parent_context(chunk)
            compressed_chunks.append(chunk)
            continue

        # Standard sentence-window compression for ordinary paragraphs
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
    Parent-child expansion: looks up the chunk's parent node (the section
    heading it lives under) and attaches its heading text. This is what
    lets the LLM say "this comes from Section 5.1.1 External E-STOP" even
    when the chunk text itself is just a raw paragraph or table.

    Mutates chunk in place, adding "expanded_heading".
    """
    parent_id = chunk.get("parent_id")
    if not parent_id:
        chunk["expanded_heading"] = chunk.get("heading_path", "")
        return

    parent_node = get_node(parent_id)
    if parent_node:
        chunk["expanded_heading"] = parent_node.get("heading_path") or parent_node.get("content", "")
    else:
        chunk["expanded_heading"] = chunk.get("heading_path", "")