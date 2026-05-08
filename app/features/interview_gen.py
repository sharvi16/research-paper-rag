from app.retrieval.retriever import PaperRetriever
from app.generation.generator import RAGGenerator

class InterviewGenerator:
    """
    Feature class that synthesizes multi-tier interview questions
    (conceptual, implementation, critical thinking) based on specific papers or broad topics.
    """
    
    def generate(self, paper_id: str, retriever: PaperRetriever, generator: RAGGenerator) -> dict:
        """
        Generates interview questions rigidly rooted in a single academic paper.
        """
        # Fetch metadata for the specified paper
        papers = retriever.get_paper_list()
        paper_meta = next((p for p in papers if p.get("paper_id") == paper_id), None)
        
        if not paper_meta:
            return {"error": f"Paper with ID '{paper_id}' not found in the database."}
            
        # Retrieve chunks spanning the paper to build the "key sections" context
        # Querying by title guarantees we pull heavily from the core concepts of this specific paper
        chunks = retriever.retrieve(
            query=paper_meta.get("paper_title", "Machine Learning"), 
            k=15, 
            filter_paper_id=paper_id
        )
        
        key_sections = retriever.build_context(chunks, max_tokens=4000)
        
        # Generate via LLM
        return generator.generate_interview_questions(
            paper_metadata=paper_meta, 
            key_sections=key_sections
        )
        
    def generate_from_topic(self, topic: str, retriever: PaperRetriever, generator: RAGGenerator) -> dict:
        """
        Retrieves highly relevant chunks across multiple papers regarding a topic,
        and generates an interview sheet testing knowledge on that topic.
        """
        chunks = retriever.retrieve(query=topic, k=15)
        
        if not chunks:
            return {"error": f"No context found in the database for topic: '{topic}'."}
            
        # Synthesize a "Mega-Paper" metadata object to trick the prompt into spanning multiple papers
        unique_titles = list(set(c.get("paper_title", "Unknown") for c in chunks))
        title_str = "Various Papers: " + ", ".join(unique_titles)
        
        mega_meta = {
            "title": title_str,
            "abstract": f"A collection of highly relevant research excerpts regarding '{topic}' aggregated from multiple sources."
        }
        
        key_sections = retriever.build_context(chunks, max_tokens=4000)
        
        # Generate via LLM
        return generator.generate_interview_questions(
            paper_metadata=mega_meta, 
            key_sections=key_sections
        )
