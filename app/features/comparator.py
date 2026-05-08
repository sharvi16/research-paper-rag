from typing import List, Tuple, Dict, Any
from app.retrieval.retriever import PaperRetriever
from app.generation.generator import RAGGenerator

class PaperComparator:
    """
    Feature class that compares two ML papers based on a specific aspect 
    and groups comparable papers based on shared taxonomy.
    """
    
    def compare(self, paper_id_1: str, paper_id_2: str, aspect: str, 
                retriever: PaperRetriever, generator: RAGGenerator) -> Dict[str, str]:
        """
        Retrieves contexts from both papers regarding the specified aspect,
        generates a markdown comparison table, and evaluates a summary/winner.
        """
        # Retrieve chunks isolated to each paper
        data = retriever.retrieve_for_comparison(paper_id_1, paper_id_2, aspect)
        p1_chunks = data["paper1_chunks"]
        p2_chunks = data["paper2_chunks"]
        
        # Extract exact titles to ensure accurate prompt mapping
        p1_title = p1_chunks[0].get("paper_title", f"Paper {paper_id_1}") if p1_chunks else f"Paper {paper_id_1}"
        p2_title = p2_chunks[0].get("paper_title", f"Paper {paper_id_2}") if p2_chunks else f"Paper {paper_id_2}"
        
        # Build contexts (using slightly lower max_tokens since we have two papers)
        p1_context = retriever.build_context(p1_chunks, max_tokens=1500)
        p2_context = retriever.build_context(p2_chunks, max_tokens=1500)
        
        paper1_data = {"title": p1_title, "context": p1_context}
        paper2_data = {"title": p2_title, "context": p2_context}
        
        # 1. Generate the Comparison Table
        table_str = generator.compare_papers(paper1_data, paper2_data, aspect)
        
        # 2. Evaluate Summary and Winner
        # Since the standard prompt doesn't explicitly enforce a "winner" field,
        # we make a secondary lightweight call to the standard QA endpoint.
        combined_context = f"Paper 1 ({p1_title}):\n{p1_context}\n\nPaper 2 ({p2_title}):\n{p2_context}"
        summary_query = f"Based on the context, compare these two approaches for '{aspect}'. Provide a 2-sentence summary and explicitly name which of the two papers is the better 'winner' for this aspect and why."
        
        summary_response = generator.answer_question(
            question=summary_query,
            context=combined_context,
            paper_titles=f"{p1_title}, {p2_title}",
            mode="standard"
        )
        
        summary_text = summary_response["answer"]
        
        # Heuristic extraction of the winner based on the text
        winner = "Tie / Depends on Use Case"
        ans_lower = summary_text.lower()
        
        # Simple boolean check: if one title is highly favored or mentioned
        if p1_title.lower() in ans_lower and p2_title.lower() not in ans_lower:
            winner = p1_title
        elif p2_title.lower() in ans_lower and p1_title.lower() not in ans_lower:
            winner = p2_title
        elif "winner" in ans_lower:
            # If the LLM explicitly stated a winner, we leave the exact extraction up to the user to read in the summary
            pass
            
        return {
            "comparison_table": table_str,
            "summary": summary_text,
            "winner_for_aspect": winner
        }

    def get_comparable_papers(self, retriever: PaperRetriever) -> List[Tuple[Dict, Dict]]:
        """
        Lists all paper pairs that share at least one common category tag from ArXiv.
        """
        papers = retriever.get_paper_list()
        pairs = []
        
        # O(N^2) comparison is fine for local academic collections
        for i in range(len(papers)):
            for j in range(i + 1, len(papers)):
                p1 = papers[i]
                p2 = papers[j]
                
                cat1 = set(p1.get("categories", []))
                cat2 = set(p2.get("categories", []))
                
                # If they share a category (e.g. 'cs.CL', 'cs.CV')
                if cat1.intersection(cat2):
                    pairs.append((p1, p2))
                    
        return pairs
