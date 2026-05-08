import re
from pathlib import Path
from typing import List, Dict, Any

class PaperChunker:
    """
    Chunks academic papers using a paragraph-based strategy.
    Discards noise (headers/footers, captions, short fragments) and merges/splits paragraphs 
    based on word count thresholds.
    """
    
    def split_into_paragraphs(self, text: str) -> List[str]:
        """
        Splits text into paragraphs by detecting double newlines (\n\n) 
        and single newlines followed by indentation.
        """
        # \n\s*\n matches double newlines with optional spaces/tabs
        # \n[ \t]+ matches single newline followed by spaces or tabs
        paras = re.split(r'\n\s*\n|\n[ \t]+', text)
        return [p for p in paras if p.strip()]

    def clean_paragraph(self, para: str) -> str:
        """
        Cleans the paragraph by removing hyphenation artifacts, normalizing whitespace,
        and filtering out figure/table captions and lone page numbers.
        """
        # Remove hyphenation artifacts across newlines (e.g. "word-\nword" -> "wordword")
        para = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', para)
        
        # Normalize whitespace (replaces newlines and multiple spaces with a single space)
        para = " ".join(para.split())
        
        if not para:
            return ""
            
        # Return empty string if paragraph is a figure or table caption
        if re.match(r'^(?:Figure|Table|Fig\.|Eq\.)', para, flags=re.IGNORECASE):
            return ""
            
        # Remove lone page numbers or strictly numeric header/footer repetitions
        if re.match(r'^\d+$', para):
            return ""
            
        return para

    def assign_section(self, para: str, section_boundaries: Dict[str, str]) -> str:
        """
        Takes the section dict from PDFParser.extract_sections() and returns 
        which section this paragraph belongs to by matching the paragraph text 
        against the extracted sections.
        """
        # Use the first 100 characters of the cleaned paragraph to find its section
        para_prefix = para[:100]
        
        for sec_name, sec_text in section_boundaries.items():
            if not sec_text:
                continue
            # Clean the section text identically to ensure substring matching works
            clean_sec = " ".join(sec_text.split())
            if para_prefix in clean_sec:
                return sec_name
                
        return "unknown"

    def _split_large_paragraph(self, para: str, max_words: int = 600) -> List[str]:
        """
        Helper method to split paragraphs exceeding max_words into sub-paragraphs 
        at the nearest sentence boundary.
        """
        words = para.split()
        if len(words) <= max_words:
            return [para]
            
        # Split at period + space + capital letter
        sentences = re.split(r'(?<=\.)\s+(?=[A-Z])', para)
        
        sub_paras = []
        current_sub = []
        current_len = 0
        
        for sent in sentences:
            sent_len = len(sent.split())
            if current_len + sent_len > max_words and current_sub:
                sub_paras.append(" ".join(current_sub))
                current_sub = [sent]
                current_len = sent_len
            else:
                current_sub.append(sent)
                current_len += sent_len
                
        if current_sub:
            sub_paras.append(" ".join(current_sub))
            
        return sub_paras

    def chunk(self, parsed_paper: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Full pipeline: split -> clean -> filter -> merge small -> split large 
        -> assign sections -> build chunk dicts.
        """
        full_text = parsed_paper.get("text", "")
        metadata = parsed_paper.get("metadata", {})
        paper_title = metadata.get("title") or metadata.get("Title") or "Unknown Title"
        
        file_path = parsed_paper.get("file_path", "")
        paper_id = Path(file_path).stem if file_path else "unknown_id"
        
        # We instantiate PDFParser locally to extract the sections 
        # (Assuming it's available in the same module structure)
        from app.ingestion.pdf_parser import PDFParser
        parser = PDFParser()
        section_boundaries = parser.extract_sections(full_text)
        
        # 1. Split
        raw_paras = self.split_into_paragraphs(full_text)
        
        # 2. Clean
        cleaned_paras = [self.clean_paragraph(p) for p in raw_paras]
        
        # 3. Filter: valid paragraphs must have at least 50 words
        filtered_paras = [p for p in cleaned_paras if len(p.split()) >= 50]
        
        # 4. Merge small: consecutive paragraphs under 100 words
        merged_paras = []
        is_merged_flags = []
        i = 0
        while i < len(filtered_paras):
            p1 = filtered_paras[i]
            if i + 1 < len(filtered_paras):
                p2 = filtered_paras[i+1]
                if len(p1.split()) < 100 and len(p2.split()) < 100:
                    merged_paras.append(p1 + " " + p2)
                    is_merged_flags.append(True)
                    i += 2
                    continue
            merged_paras.append(p1)
            is_merged_flags.append(False)
            i += 1
            
        # 5. Split large: > 600 words
        final_paras = []
        final_merged_flags = []
        final_split_flags = []
        
        for p, is_m in zip(merged_paras, is_merged_flags):
            if len(p.split()) > 600:
                subs = self._split_large_paragraph(p, max_words=600)
                for sub in subs:
                    final_paras.append(sub)
                    final_merged_flags.append(is_m)
                    final_split_flags.append(True)
            else:
                final_paras.append(p)
                final_merged_flags.append(is_m)
                final_split_flags.append(False)
                
        # 6. Assign sections & Build dicts
        chunks = []
        for idx, (p, is_m, is_s) in enumerate(zip(final_paras, final_merged_flags, final_split_flags)):
            sec = self.assign_section(p, section_boundaries)
            prefix = f"[Section: {sec.title()}]\n" if sec != "unknown" else "[Section: Unknown]\n"
            
            chunk_dict = {
                "chunk_id": f"{paper_id}_p{idx}",
                "text": prefix + p,
                "paper_id": paper_id,
                "paper_title": paper_title,
                "section": sec,
                "word_count": len(p.split()),
                "paragraph_index": idx,
                "total_chunks": 0,  # placeholder
                "is_merged": is_m,
                "is_split": is_s
            }
            chunks.append(chunk_dict)
            
        # Update total chunks
        total_c = len(chunks)
        for c in chunks:
            c["total_chunks"] = total_c

        chunks = self.merge_metadata(chunks, metadata)

        return chunks

    def merge_metadata(self, chunks: List[Dict[str, Any]], arxiv_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Adds authors, date, url, abstract, categories to each chunk.
        """
        for c in chunks:
            c["authors"] = arxiv_metadata.get("authors", [])
            c["published_date"] = arxiv_metadata.get("published_date", "")
            c["url"] = arxiv_metadata.get("url", "")
            c["abstract"] = arxiv_metadata.get("abstract", "")
            c["categories"] = arxiv_metadata.get("categories", [])
        return chunks

    def get_chunk_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates and returns statistics about the chunking process.
        """
        if not chunks:
            stats = {
                "total_chunks": 0,
                "avg_word_count": 0,
                "min_word_count": 0,
                "max_word_count": 0,
                "merged_count": 0,
                "split_count": 0,
                "sections_found": []
            }
            print(f"Chunking Stats: {stats}")
            return stats
            
        word_counts = [c["word_count"] for c in chunks]
        merged_count = sum(1 for c in chunks if c["is_merged"])
        split_count = sum(1 for c in chunks if c["is_split"])
        sections = list(set(c["section"] for c in chunks if c["section"] != "unknown"))
        
        stats = {
            "total_chunks": len(chunks),
            "avg_word_count": round(sum(word_counts) / len(word_counts), 1),
            "min_word_count": min(word_counts),
            "max_word_count": max(word_counts),
            "merged_count": merged_count,
            "split_count": split_count,
            "sections_found": sections
        }
        
        print(f"Chunking Stats: {stats}")
        return stats
