import logging
import re
from typing import Dict, List, Any
import PyPDF2

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None
    logging.warning("PyMuPDF (fitz) is not installed. Will default to PyPDF2.")

class PDFParser:
    """
    A robust PDF parser for ML research papers.
    Uses PyMuPDF as the primary engine with a fallback to PyPDF2.
    Includes heuristic-based extraction for common paper sections and citations.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def parse(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parses a PDF file and returns its text, pages, and metadata.
        Handles scanned PDFs gracefully by checking the extracted text volume.
        """
        result = {
            "text": "",
            "pages": [],
            "metadata": {},
            "num_pages": 0,
            "file_path": pdf_path
        }

        parsed_successfully = False

        # Attempt 1: PyMuPDF (fitz)
        if fitz:
            try:
                doc = fitz.open(pdf_path)
                result["num_pages"] = len(doc)
                result["metadata"] = doc.metadata or {}
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    page_text = page.get_text("text")
                    result["pages"].append(page_text)
                    
                result["text"] = "\n\n".join(result["pages"])
                doc.close()
                parsed_successfully = True
            except Exception as e:
                self.logger.warning(f"PyMuPDF failed on {pdf_path}: {e}. Falling back to PyPDF2.")

        # Attempt 2: PyPDF2 Fallback
        if not parsed_successfully:
            try:
                with open(pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    result["num_pages"] = len(reader.pages)
                    
                    if reader.metadata:
                        result["metadata"] = {str(k).strip('/'): str(v) for k, v in reader.metadata.items()}
                        
                    for page in reader.pages:
                        page_text = page.extract_text() or ""
                        result["pages"].append(page_text)
                        
                result["text"] = "\n\n".join(result["pages"])
                parsed_successfully = True
            except Exception as e2:
                self.logger.error(f"PyPDF2 fallback also failed for {pdf_path}. Error: {e2}")
                return result

        # Handle scanned PDFs gracefully
        # Heuristic: If there is extremely little text compared to the number of pages,
        # it is likely a scanned document (images) without an OCR layer.
        total_chars = len(result["text"].strip())
        if result["num_pages"] > 0 and total_chars < (result["num_pages"] * 50):
            self.logger.warning(f"File {pdf_path} appears to be a scanned PDF (little to no text extracted).")
            result["text"] = ""
            result["pages"] = ["" for _ in range(result["num_pages"])]

        return result

    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Identifies and extracts common paper sections using regex and heuristics.
        Returns a dictionary with keys: abstract, introduction, related_work,
        methodology, experiments, results, conclusion, references.
        """
        sections = {
            "abstract": "",
            "introduction": "",
            "related_work": "",
            "methodology": "",
            "experiments": "",
            "results": "",
            "conclusion": "",
            "references": ""
        }

        if not text:
            return sections

        # Map header variations to canonical keys
        aliases = {
            "abstract": "abstract",
            "introduction": "introduction",
            "related work": "related_work", "background": "related_work", "literature review": "related_work",
            "methodology": "methodology", "method": "methodology", "approach": "methodology", "model": "methodology", "architecture": "methodology",
            "experiments": "experiments", "evaluation": "experiments", "experimental setup": "experiments",
            "results": "results", "discussion": "results", "results and discussion": "results",
            "conclusion": "conclusion", "conclusions": "conclusion",
            "references": "references", "bibliography": "references"
        }

        # Regex to match common academic headers, allowing for leading numbers (e.g., "1. Introduction", "IV. Background")
        header_pattern = re.compile(
            r'^\s*(?:(?:[IVXLCDM]+|[0-9]+)(?:\.[0-9]+)*\.?\s+)?(Abstract|Introduction|Related Work|Background|Literature Review|Methodology|Method|Approach|Model|Architecture|Experiments|Evaluation|Experimental Setup|Results|Discussion|Results and Discussion|Conclusion|Conclusions|References|Bibliography)\s*$',
            re.IGNORECASE | re.MULTILINE
        )

        matches = list(header_pattern.finditer(text))

        # Fallback Abstract Extraction if no header is found
        if not matches or matches[0].group(1).lower() not in ["abstract"]:
            abstract_match = re.search(r'(?i)abstract[\.\:\s]+(.*?)(?=\n\s*(?:(?:[IVXLCDM]+|[0-9]+)\.?\s*)?introduction|\Z)', text, re.DOTALL)
            if abstract_match:
                sections["abstract"] = abstract_match.group(1).strip()

        for i, match in enumerate(matches):
            header_name = match.group(1).lower()
            canonical_name = aliases.get(header_name)
            
            start_idx = match.end()
            end_idx = matches[i+1].start() if i + 1 < len(matches) else len(text)
            
            section_content = text[start_idx:end_idx].strip()
            
            if canonical_name and canonical_name in sections:
                # If a section appears multiple times (e.g. Results, Discussion), combine them
                if sections[canonical_name]:
                    sections[canonical_name] += "\n\n" + section_content
                else:
                    sections[canonical_name] = section_content

        return sections

    def extract_citations(self, text: str) -> List[str]:
        """
        Extracts cited paper titles/IDs found in the references section.
        """
        sections = self.extract_sections(text)
        ref_text = sections.get("references", "")

        # Fallback if the references section wasn't captured cleanly by section extraction
        if not ref_text:
            ref_match = re.search(r'(?i)^\s*(?:[0-9]+\.?\s*)?(?:references|bibliography)\s*$(.*)', text, re.DOTALL | re.MULTILINE)
            if ref_match:
                ref_text = ref_match.group(1)

        if not ref_text:
            return []

        citations = []
        
        # Strategy 1: Look for bracketed numbers like [1], [2], [3]
        if re.search(r'\[[0-9]+\]', ref_text):
            chunks = re.split(r'\[[0-9]+\]', ref_text)
            for chunk in chunks:
                cleaned = " ".join(chunk.strip().split()) # Normalize whitespace/newlines
                if len(cleaned) > 15:  # Filter out noise
                    citations.append(cleaned)
        
        # Strategy 2: Look for numbered lists like "1.", "2.", "3."
        elif re.search(r'^\s*[0-9]+\.\s', ref_text, re.MULTILINE):
            chunks = re.split(r'^\s*[0-9]+\.\s', ref_text, flags=re.MULTILINE)
            for chunk in chunks:
                cleaned = " ".join(chunk.strip().split())
                if len(cleaned) > 15:
                    citations.append(cleaned)
                    
        # Strategy 3: Split by large whitespace gaps or double newlines
        else:
            chunks = re.split(r'\n\s*\n', ref_text)
            for chunk in chunks:
                cleaned = " ".join(chunk.strip().split())
                if len(cleaned) > 15:
                    citations.append(cleaned)

        return citations
