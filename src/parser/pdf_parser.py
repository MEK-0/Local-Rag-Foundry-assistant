import pdfplumber
from typing import Dict, Any
from src.parser.base import BaseParser

class PDFDocumentParser(BaseParser):
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Extracts text from PDF files page by page using pdfplumber.
        """
        extracted_text = []
        total_pages = 0
        
        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    # Append a structural marker for tracking boundaries later
                    extracted_text.append(f"[Page {page_num}]\n{page_text}")
                    
        return {
            "content": "\n\n".join(extracted_text),
            "metadata": {
                "total_pages": total_pages,
                "parser_used": "pdfplumber"
            }
        }