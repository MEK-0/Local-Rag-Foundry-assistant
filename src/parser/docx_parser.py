import docx
from typing import Dict, Any
from src.parsers.base import BaseParser

class DocxDocumentParser(BaseParser):
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Extracts operational text from Word documents (.docx).
        """
        doc = docx.Document(file_path)
        full_text = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                full_text.append(paragraph.text)
                
        return {
            "content": "\n".join(full_text),
            "metadata": {
                "paragraphs_count": len(doc.paragraphs),
                "parser_used": "python-docx"
            }
        }