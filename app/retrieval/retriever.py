import json
import re
from typing import List, Dict, Any

# We assume embedder is passed in to avoid strong coupling/re-initialization
from app.ingestion.embedder import PaperEmbedder


class PaperRetriever:
    """
    Handles querying the vector database to retrieve relevant academic paper chunks.
    It builds properly formatted contexts for LLM generation.
    """
    
    def __init__(self, embedder: PaperEmbedder):
        self.embedder = embedder
        self.collection = self.embedder.collection

    def retrieve(self, query: str, k: int = 8, filter_paper_id: str = None) -> List[Dict[str, Any]]:
        """
        Embeds the query and queries ChromaDB for the top-k similar chunks.
        Optionally filters results to a specific paper via paper_id.
        """
        query_embedding = self.embedder.get_embedding(query)
        
        # Build ChromaDB where-clause for metadata filtering
        where_clause = None
        if filter_paper_id:
            where_clause = {"paper_id": filter_paper_id}
            
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        # Check if results exist
        if not results["ids"] or not results["ids"][0]:
            return []
            
        chunks = []
        # results["ids"] is a list of lists (one list per query). Since we send 1 query, we access [0].
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            # ChromaDB cosine distance = 1 - cosine_similarity
            similarity = 1.0 - distance
            
            chunk = {
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "distance": distance,
                "similarity": similarity
            }
            
            # Merge and deserialize metadata back to native Python types
            metadata = results["metadatas"][0][i]
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                        try:
                            chunk[key] = json.loads(value)
                        except json.JSONDecodeError:
                            chunk[key] = value
                    else:
                        chunk[key] = value
                        
            chunks.append(chunk)
            
        return chunks

    def retrieve_for_comparison(self, paper_id_1: str, paper_id_2: str, topic: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieves relevant chunks from two specific papers based on a shared topic.
        Returns a dictionary separating the chunks.
        """
        # Retrieve top 5-8 chunks per paper to compare
        p1_chunks = self.retrieve(query=topic, k=8, filter_paper_id=paper_id_1)
        p2_chunks = self.retrieve(query=topic, k=8, filter_paper_id=paper_id_2)
        
        return {
            "paper1_chunks": p1_chunks,
            "paper2_chunks": p2_chunks
        }

    def build_context(self, chunks: List[Dict[str, Any]], max_tokens: int = 3000) -> str:
        """
        Formats chunks into a single context string.
        Truncates the inclusion of chunks to stay under the token limit constraint.
        Uses a heuristic of 1 token ≈ 4 characters to avoid heavy tokenizer dependencies.
        """
        max_chars = max_tokens * 4
        context_parts = []
        current_chars = 0
        
        for chunk in chunks:
            title = chunk.get("paper_title", "Unknown Paper")
            section = chunk.get("section", "Unknown").title()
            text = chunk.get("text", "").strip()
            
            # Strip the redundant [Section: X] prefix added by PaperChunker 
            # to cleanly apply the new requested format
            clean_text = re.sub(r'^\[Section: .*?\]\n', '', text, flags=re.IGNORECASE).strip()
            
            formatted = f"[Paper: {title} | Section: {section}]\n{clean_text}\n---"
            
            # Check length limits before adding
            if current_chars + len(formatted) > max_chars:
                # If we haven't added anything yet, aggressively truncate the text 
                # so we have at least partial context rather than breaking immediately.
                if current_chars == 0:
                    allowed_len = max_chars - len(f"[Paper: {title} | Section: {section}]\n\n---")
                    if allowed_len > 0:
                        formatted = f"[Paper: {title} | Section: {section}]\n{clean_text[:allowed_len]}...\n---"
                        context_parts.append(formatted)
                break
                
            context_parts.append(formatted)
            current_chars += len(formatted)
            
        return "\n\n".join(context_parts)

    def get_paper_list(self) -> List[Dict[str, Any]]:
        """
        Returns a list of all unique papers indexed in the database with their metadata.
        Aggregates by inspecting the chunk metadata in ChromaDB.
        """
        try:
            all_data = self.collection.get(include=["metadatas"])
        except Exception:
            return []
            
        if not all_data or not all_data["metadatas"]:
            return []
            
        unique_papers = {}
        for meta in all_data["metadatas"]:
            if not meta or "paper_id" not in meta:
                continue
                
            p_id = meta["paper_id"]
            if p_id not in unique_papers:
                # Extract and deserialize paper-level metadata
                paper_info = {}
                for key, value in meta.items():
                    if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                        try:
                            paper_info[key] = json.loads(value)
                        except json.JSONDecodeError:
                            paper_info[key] = value
                    else:
                        paper_info[key] = value
                
                # Consolidate standard paper fields 
                # (filtering out chunk-level keys like 'word_count', 'section')
                filtered_info = {
                    "paper_id": p_id,
                    "paper_title": paper_info.get("paper_title", "Unknown Title"),
                    "authors": paper_info.get("authors", []),
                    "abstract": paper_info.get("abstract", ""),
                    "published_date": paper_info.get("published_date", ""),
                    "url": paper_info.get("url", ""),
                    "categories": paper_info.get("categories", [])
                }
                unique_papers[p_id] = filtered_info
                
        return list(unique_papers.values())
