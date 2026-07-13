import re
from typing import Any, Dict, List, Optional

import pdfplumber

from src.parser.base import BaseParser, KnowledgeNode, NodeType

# "5.1.1 External E-STOP", "Section 3.8", "A.2 Notes" gibi başlıkları yakalar
HEADING_PATTERN = re.compile(
    r"^(?P<num>\d+(\.\d+){0,4}|[A-Z]\.\d+)\s+(?P<title>[A-ZÇĞİÖŞÜ][^\n]{2,80})$"
)

WARNING_PATTERN = re.compile(r"^\s*(WARNING|UYARI|CAUTION|DİKKAT)\b", re.IGNORECASE)
NOTE_PATTERN = re.compile(r"^\s*(NOTE|NOT)\b", re.IGNORECASE)


class PDFDocumentParser(BaseParser):
    """
    PDF'i düz text yerine hiyerarşik KnowledgeNode ağacına çevirir.
    - Heading tespiti: numaralandırma regex + font-size/bold heuristiği
    - Table extraction: pdfplumber.extract_tables() -> markdown table node
    - Figure: page.images -> placeholder node (vision captioning sonraki fazda doldurulacak)
    """

    def __init__(self, min_heading_font_ratio: float = 1.08):
        # heading font'u, sayfadaki baskın (gövde) font boyutunun en az bu kadar
        # katı olmalı; taramalı/garip PDF'lerde saf regex'e güvenmemek için ek sinyal
        self.min_heading_font_ratio = min_heading_font_ratio

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def parse_to_tree(self, file_path: str, doc_id: str) -> List[KnowledgeNode]:
        nodes: List[KnowledgeNode] = []
        heading_stack: List[KnowledgeNode] = []  # aktif ata zinciri (level'a göre)
        order_counter = 0

        with pdfplumber.open(file_path) as pdf:
            body_font_size = self._estimate_body_font_size(pdf)

            for page_num, page in enumerate(pdf.pages, start=1):
                # 1) Tabloları önce çıkar; bbox'larını not et ki text extraction
                #    onları tekrar paragraf gibi yutmasın
                table_bboxes = []
                for table_obj in page.find_tables():
                    table_bboxes.append(table_obj.bbox)
                    md_table = self._table_to_markdown(table_obj.extract())
                    if not md_table:
                        continue
                    parent = heading_stack[-1] if heading_stack else None
                    order_counter += 1
                    nodes.append(
                        KnowledgeNode(
                            doc_id=doc_id,
                            parent_id=parent.id if parent else None,
                            type=NodeType.TABLE,
                            level=(parent.level + 1) if parent else 1,
                            heading_path=self._path_for(parent),
                            content=md_table,
                            page=page_num,
                            bbox=self._bbox_dict(table_obj.bbox),
                            order=order_counter,
                            metadata={"rows": len(table_obj.extract())},
                        )
                    )

                # 2) Görselleri figure placeholder olarak çıkar
                for img in page.images:
                    parent = heading_stack[-1] if heading_stack else None
                    order_counter += 1
                    nodes.append(
                        KnowledgeNode(
                            doc_id=doc_id,
                            parent_id=parent.id if parent else None,
                            type=NodeType.FIGURE,
                            level=(parent.level + 1) if parent else 1,
                            heading_path=self._path_for(parent),
                            content="",  # vision captioning fazında doldurulacak
                            page=page_num,
                            bbox=self._bbox_dict(
                                (img["x0"], img["top"], img["x1"], img["bottom"])
                            ),
                            order=order_counter,
                            metadata={"needs_captioning": True},
                        )
                    )

                # 3) Kalan text'i satır satır işleyip heading/paragraph/warning/note ayır
                words = page.extract_words(extra_attrs=["size", "fontname"])
                lines = self._group_words_into_lines(words)
                lines = self._drop_lines_inside_bboxes(lines, table_bboxes)

                buffer: List[str] = []
                for line_text, avg_size, is_bold in lines:
                    heading_match = HEADING_PATTERN.match(line_text.strip())
                    looks_like_heading = heading_match and (
                        avg_size >= body_font_size * self.min_heading_font_ratio or is_bold
                    )

                    if looks_like_heading:
                        self._flush_paragraph(
                            buffer, nodes, heading_stack, doc_id, page_num, order_counter
                        )
                        order_counter += 1
                        buffer = []

                        level = heading_match.group("num").count(".") + 1
                        node = KnowledgeNode(
                            doc_id=doc_id,
                            type=NodeType.SECTION,
                            level=level,
                            content=line_text.strip(),
                            page=page_num,
                            order=order_counter,
                        )
                        # heading_stack'i level'a göre kırp, parent bağla
                        while heading_stack and heading_stack[-1].level >= level:
                            heading_stack.pop()
                        parent = heading_stack[-1] if heading_stack else None
                        node.parent_id = parent.id if parent else None
                        node.heading_path = self._path_for(parent, node.content)
                        heading_stack.append(node)
                        nodes.append(node)
                        continue

                    # warning/note ayrı node tipi olarak işaretlenir ama buffer'a girmez
                    if WARNING_PATTERN.match(line_text) or NOTE_PATTERN.match(line_text):
                        self._flush_paragraph(
                            buffer, nodes, heading_stack, doc_id, page_num, order_counter
                        )
                        buffer = []
                        order_counter += 1
                        parent = heading_stack[-1] if heading_stack else None
                        node_type = NodeType.WARNING if WARNING_PATTERN.match(line_text) else NodeType.NOTE
                        nodes.append(
                            KnowledgeNode(
                                doc_id=doc_id,
                                parent_id=parent.id if parent else None,
                                type=node_type,
                                level=(parent.level + 1) if parent else 1,
                                heading_path=self._path_for(parent),
                                content=line_text.strip(),
                                page=page_num,
                                order=order_counter,
                            )
                        )
                        continue

                    buffer.append(line_text)

                self._flush_paragraph(
                    buffer, nodes, heading_stack, doc_id, page_num, order_counter
                )

        return nodes

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _flush_paragraph(
        self,
        buffer: List[str],
        nodes: List[KnowledgeNode],
        heading_stack: List[KnowledgeNode],
        doc_id: str,
        page_num: int,
        order_counter: int,
    ) -> None:
        text = " ".join(buffer).strip()
        if not text:
            return
        parent = heading_stack[-1] if heading_stack else None
        nodes.append(
            KnowledgeNode(
                doc_id=doc_id,
                parent_id=parent.id if parent else None,
                type=NodeType.PARAGRAPH,
                level=(parent.level + 1) if parent else 1,
                heading_path=self._path_for(parent),
                content=text,
                page=page_num,
                order=order_counter,
            )
        )

    @staticmethod
    def _path_for(parent: Optional[KnowledgeNode], current_title: Optional[str] = None) -> str:
        base = parent.heading_path if parent else ""
        if parent and not base:
            base = parent.content
        if current_title:
            return f"{base} > {current_title}" if base else current_title
        return base

    @staticmethod
    def _bbox_dict(bbox) -> Dict[str, float]:
        x0, y0, x1, y1 = bbox
        return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

    @staticmethod
    def _table_to_markdown(rows: List[List[Optional[str]]]) -> str:
        rows = [r for r in rows if any(c for c in r)]
        if not rows:
            return ""
        header, *body = rows
        header = [c or "" for c in header]
        md = "| " + " | ".join(header) + " |\n"
        md += "| " + " | ".join(["---"] * len(header)) + " |\n"
        for row in body:
            row = [(c or "").replace("\n", " ") for c in row]
            md += "| " + " | ".join(row) + " |\n"
        return md

    @staticmethod
    def _estimate_body_font_size(pdf) -> float:
        sizes = []
        for page in pdf.pages[:5]:  # ilk 5 sayfa yeterli sample
            for w in page.extract_words(extra_attrs=["size"]):
                sizes.append(w["size"])
        if not sizes:
            return 10.0
        sizes.sort()
        return sizes[len(sizes) // 2]  # median = gövde font'u

    @staticmethod
    def _group_words_into_lines(words: List[Dict[str, Any]]):
        """pdfplumber word listesini y-koordinatına göre satırlara gruplar."""
        lines: List[Dict[str, Any]] = []
        current_y = None
        current_words: List[Dict[str, Any]] = []
        for w in sorted(words, key=lambda x: (round(x["top"], 1), x["x0"])):
            if current_y is None or abs(w["top"] - current_y) > 2:
                if current_words:
                    lines.append(current_words)
                current_words = [w]
                current_y = w["top"]
            else:
                current_words.append(w)
        if current_words:
            lines.append(current_words)

        result = []
        for line_words in lines:
            text = " ".join(w["text"] for w in line_words)
            avg_size = sum(w["size"] for w in line_words) / len(line_words)
            is_bold = any("bold" in w.get("fontname", "").lower() for w in line_words)
            result.append((text, avg_size, is_bold))
        return result

    @staticmethod
    def _drop_lines_inside_bboxes(lines, table_bboxes):
        # basit versiyon: satır metnini tablo hücreleriyle karıştırmamak için
        # bbox filtrelemesi şu an word-level koordinat taşımadığından
        # (satırlar text+size+bold olarak sadeleştirildi) tam filtre sonraki iterasyonda
        # word-level bbox taşıyarak eklenecek. Şimdilik pas geçiyoruz.
        return lines