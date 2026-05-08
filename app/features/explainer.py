from app.retrieval.retriever import PaperRetriever
from app.generation.generator import RAGGenerator

class BeginnerExplainer:
    """
    Feature class that orchestrates answering questions at a high school reading level,
    heavily leaning on analogies, simple language, and acronym explanations.
    """
    
    def explain(self, question: str, retriever: PaperRetriever, generator: RAGGenerator) -> dict:
        """
        Retrieves relevant context and delegates to the generator's beginner mode.
        """
        # Retrieve chunks relevant to the question
        chunks = retriever.retrieve(query=question, k=6)
        
        # Build context (enforcing token limits)
        context = retriever.build_context(chunks, max_tokens=3000)
        
        # Extract unique paper titles for context injection
        unique_titles = list(set(c.get("paper_title", "Unknown Paper") for c in chunks))
        paper_titles = ", ".join(unique_titles)
        
        # Generate answer using the beginner prompt template
        response = generator.answer_question(
            question=question, 
            context=context, 
            paper_titles=paper_titles,
            mode="beginner"
        )
        
        response["sources"] = [
            {"title": c.get("paper_title", "Unknown"), "section": c.get("section", "Unknown"), "similarity": c.get("similarity", 0.0)} 
            for c in chunks
        ]
        
        return response
