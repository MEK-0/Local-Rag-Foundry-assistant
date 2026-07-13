from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class NodeType(str, Enum):
    DOCUMENT = "document"
    SECTION = "section"          # heading node (any level)
    PARAGRAPH = "paragraph"
    TABLE = "table"
    TABLE_ROW = "table_row"        # opsiyonel: satır bazlı granülerlik istersen
    FIGURE = "figure"
    WARNING = "warning"
    NOTE = "note"
    CODE = "code"


@dataclass
class KnowledgeNode:
    """
    Chunk yerine kullanılan temel bilgi birimi.
    Her parser bu tipte bir liste (flat, parent_id ile bağlı) döndürür.
    """
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    parent_id: Optional[str] = None
    doc_id: str = ""
    type: NodeType = NodeType.PARAGRAPH
    level: int = 0                          # heading derinliği (0 = root/document)
    heading_path: str = ""                  # "5 Safety > 5.1 Emergency Stop"
    content: str = ""                       # text veya markdown table
    page: Optional[int] = None
    bbox: Optional[Dict[str, float]] = None  # {"x0":.., "y0":.., "x1":.., "y1":..}
    order: int = 0                          # aynı parent altında sıralama
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["type"] = self.type.value if isinstance(self.type, NodeType) else self.type
        return d


class BaseParser(ABC):
    @abstractmethod
    def parse_to_tree(self, file_path: str, doc_id: str) -> List[KnowledgeNode]:
        """
        Belgeyi hiyerarşik KnowledgeNode listesine dönüştürür.
        Liste flat'tir; hiyerarşi parent_id üzerinden kurulur.

        Args:
            file_path: Belge yolu.
            doc_id: db.py'da bu belgeye atanan doc id (ingest.py tarafından üretilir).

        Returns:
            List[KnowledgeNode]
        """
        raise NotImplementedError

    # --- Geriye dönük uyumluluk ---
    # Eski pipeline (rag_pipeline.py, run_eval.py) hala parse()->{"content","metadata"}
    # bekliyorsa kırılmasın diye default implementasyon: tree'yi düz text'e indirger.
    def parse(self, file_path: str) -> Dict[str, Any]:
        nodes = self.parse_to_tree(file_path, doc_id="_legacy")
        content = "\n\n".join(
            n.content for n in nodes if n.type in (NodeType.PARAGRAPH, NodeType.SECTION, NodeType.TABLE)
        )
        return {
            "content": content,
            "metadata": {"node_count": len(nodes), "parser_used": self.__class__.__name__},
        }