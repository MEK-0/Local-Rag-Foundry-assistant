from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> Dict[str, Any]:
        """
        Parses the given file and returns a structured dictionary.
        
        Args:
            file_path (str): The absolute or relative path to the document.
            
        Returns:
            Dict[str, Any]: A dictionary containing:
                - "content": str (The extracted raw text content)
                - "metadata": dict (File specific metadata like page count, sheet names, etc.)
        """
        pass