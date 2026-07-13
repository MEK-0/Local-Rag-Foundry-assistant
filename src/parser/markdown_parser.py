import os
import re
from typing import List

from src.parser.base import BaseParser, KnowledgeNode, NodeType

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
CODE_FENCE_RE = re.compile(r"^```")
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


class MarkdownDocumentParser(BaseParser):
    def parse_to_tree(self, file_path: str, doc_id: str) -> List[KnowledgeNode]:
        """
        Converts a Markdown file into a KnowledgeNode tree.
        - '#'..'######' headings become SECTION nodes (level = number of '#').
        - Fenced code blocks (```...```) become atomic CODE nodes.
        - Contiguous '|...|' lines become atomic TABLE nodes (kept as-is,
          markdown tables are already in the target format).
        - Everything else accumulates into PARAGRAPH nodes, flushed on
          blank lines or when a heading/code/table boundary is hit.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        nodes: List[KnowledgeNode] = []
        heading_stack: List[KnowledgeNode] = []
        order_counter = 0
        buffer: List[str] = []

        i = 0
        while i < len(lines):
            line = lines[i]

            heading_match = HEADING_RE.match(line)
            if heading_match:
                order_counter = self._flush_paragraph(buffer, nodes, heading_stack, doc_id, order_counter)
                buffer = []

                level = len(heading_match.group(1))
                order_counter += 1
                while heading_stack and heading_stack[-1].level >= level:
                    heading_stack.pop()
                parent = heading_stack[-1] if heading_stack else None

                node = KnowledgeNode(
                    doc_id=doc_id,
                    parent_id=parent.id if parent else None,
                    type=NodeType.SECTION,
                    level=level,
                    content=heading_match.group(2).strip(),
                    order=order_counter,
                )
                node.heading_path = self._path_for(parent, node.content)
                heading_stack.append(node)
                nodes.append(node)
                i += 1
                continue

            if CODE_FENCE_RE.match(line):
                order_counter = self._flush_paragraph(buffer, nodes, heading_stack, doc_id, order_counter)
                buffer = []
                code_lines = []
                i += 1
                while i < len(lines) and not CODE_FENCE_RE.match(lines[i]):
                    code_lines.append(lines[i])
                    i += 1
                i += 1  # skip closing fence
                order_counter += 1
                parent = heading_stack[-1] if heading_stack else None
                nodes.append(
                    KnowledgeNode(
                        doc_id=doc_id,
                        parent_id=parent.id if parent else None,
                        type=NodeType.CODE,
                        level=(parent.level + 1) if parent else 1,
                        heading_path=self._path_for(parent),
                        content="\n".join(code_lines),
                        order=order_counter,
                    )
                )
                continue

            if TABLE_ROW_RE.match(line):
                order_counter = self._flush_paragraph(buffer, nodes, heading_stack, doc_id, order_counter)
                buffer = []
                table_lines = []
                while i < len(lines) and TABLE_ROW_RE.match(lines[i]):
                    table_lines.append(lines[i])
                    i += 1
                order_counter += 1
                parent = heading_stack[-1] if heading_stack else None
                nodes.append(
                    KnowledgeNode(
                        doc_id=doc_id,
                        parent_id=parent.id if parent else None,
                        type=NodeType.TABLE,
                        level=(parent.level + 1) if parent else 1,
                        heading_path=self._path_for(parent),
                        content="\n".join(table_lines),
                        order=order_counter,
                    )
                )
                continue

            if not line.strip():
                order_counter = self._flush_paragraph(buffer, nodes, heading_stack, doc_id, order_counter)
                buffer = []
            else:
                buffer.append(line)
            i += 1

        self._flush_paragraph(buffer, nodes, heading_stack, doc_id, order_counter)
        return nodes

    @staticmethod
    def _flush_paragraph(buffer, nodes, heading_stack, doc_id, order_counter) -> int:
        text = " ".join(buffer).strip()
        if not text:
            return order_counter
        order_counter += 1
        parent = heading_stack[-1] if heading_stack else None
        nodes.append(
            KnowledgeNode(
                doc_id=doc_id,
                parent_id=parent.id if parent else None,
                type=NodeType.PARAGRAPH,
                level=(parent.level + 1) if parent else 1,
                heading_path=MarkdownDocumentParser._path_for(parent),
                content=text,
                order=order_counter,
            )
        )
        return order_counter

    @staticmethod
    def _path_for(parent, current_title: str = None) -> str:
        base = parent.heading_path if parent else ""
        if parent and not base:
            base = parent.content
        if current_title:
            return f"{base} > {current_title}" if base else current_title
        return base