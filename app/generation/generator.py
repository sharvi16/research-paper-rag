import os
import re
import time
import logging
from typing import Dict, List, Any

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
import os

load_dotenv()

from app.generation.prompts import (
    QA_PROMPT,
    BEGINNER_PROMPT,
    COMPARISON_PROMPT,
    INTERVIEW_PROMPT,
    SUMMARY_PROMPT
)

class RAGGenerator:
    """
    Handles interacting with the LLM to generate responses, summaries, and structured outputs
    based on the retrieved context.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.model_name = "llama-3.1-8b-instant"
        groq_key = os.getenv("GROQ_API_KEY")

        # ChatGroq requires GROQ_API_KEY environment variable to be set.
        # LangChain handles the env var automatically if it's set.
        try:
            self.llm = ChatGroq(
                api_key=groq_key,
                model=self.model_name,
                temperature=0.2, # Low temp to reduce hallucinations in RAG
                max_retries=2
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize ChatGroq. Check GROQ_API_KEY: {e}")
            raise

    def _call_with_retry(self, template: str, kwargs: dict):
        """
        Helper method to invoke the LLM via LangChain LCEL with explicit fallback retry logic 
        for network or API errors. Max 2 retries.
        """
        chain = PromptTemplate.from_template(template) | self.llm
        max_retries = 2
        
        for attempt in range(max_retries + 1):
            try:
                response = chain.invoke(kwargs)
                return response
            except Exception as e:
                self.logger.warning(f"Groq API error on attempt {attempt + 1}/{max_retries + 1}: {e}")
                if attempt == max_retries:
                    raise e
                time.sleep(1.5 ** attempt) # Exponential backoff

    def answer_question(self, question: str, context: str, paper_titles: str, mode: str = "standard") -> Dict[str, Any]:
        """
        Answers a user question using the provided context.
        Supports standard and beginner modes.
        """
        if mode == "beginner":
            template = BEGINNER_PROMPT
            kwargs = {"context": context, "question": question}
        else:
            template = QA_PROMPT
            kwargs = {"context": context, "question": question, "paper_titles": paper_titles}
            
        response = self._call_with_retry(template, kwargs)
        
        # Attempt to extract token usage (LangChain ChatGroq returns this in response_metadata)
        tokens_used = 0
        if hasattr(response, "response_metadata") and "token_usage" in response.response_metadata:
            tokens_used = response.response_metadata["token_usage"].get("total_tokens", 0)
            
        return {
            "answer": response.content.strip(),
            "tokens_used": tokens_used,
            "model": self.model_name
        }

    def compare_papers(self, paper1_data: dict, paper2_data: dict, aspect: str = "methodology") -> str:
        """
        Compares two papers based on a specific aspect and returns a markdown table.
        """
        kwargs = {
            "paper1_title": paper1_data.get("title", "Paper 1"),
            "paper1_context": paper1_data.get("context", ""),
            "paper2_title": paper2_data.get("title", "Paper 2"),
            "paper2_context": paper2_data.get("context", ""),
            "aspect": aspect
        }
        
        response = self._call_with_retry(COMPARISON_PROMPT, kwargs)
        return response.content.strip()

    def generate_interview_questions(self, paper_metadata: dict, key_sections: str) -> Dict[str, List[Dict[str, str]]]:
        """
        Generates and parses interview questions for a paper.
        Returns parsed questions grouped into conceptual, implementation, and critical.
        """
        kwargs = {
            "title": paper_metadata.get("title", ""),
            "abstract": paper_metadata.get("abstract", ""),
            "key_sections": key_sections
        }

        response = self._call_with_retry(INTERVIEW_PROMPT, kwargs)
        text = response.content
        self.logger.info(f"Raw interview response:\n{text}")  # helpful for debugging

        result = {
            "conceptual": [],
            "implementation": [],
            "critical": []
        }

        # Map section headers to result keys
        section_map = {
            "conceptual": "conceptual",
            "implementation": "implementation",
            "critical thinking": "critical",
            "critical": "critical"
        }

        current_category = None
        current_question = None

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            line_lower = line.lower()

            # Detect section headers like "### Conceptual Questions"
            for keyword, category in section_map.items():
                if keyword in line_lower and ("question" in line_lower or "###" in line or keyword == line_lower):
                    current_category = category
                    current_question = None
                    break

            if current_category is None:
                continue

            # Match Q lines: "Q1: text" or "Q1. text"
            q_match = re.match(r'^[Qq]\d+[:.]\s*(.+)', line)
            if q_match:
                current_question = q_match.group(1).strip()
                continue

            # Match A lines: "A1: text" or "A1. text"
            a_match = re.match(r'^[Aa]\d+[:.]\s*(.+)', line)
            if a_match and current_question:
                result[current_category].append({
                    "question": current_question,
                    "model_answer": a_match.group(1).strip()
                })
                current_question = None
                continue

        # --- Fallback: if all 3 categories are still empty, try loose regex parsing ---
        if not any(result.values()):
            self.logger.warning("Structured parsing failed, attempting fallback regex parse.")
            current_category = None

            for line in text.split("\n"):
                line = line.strip()
                if not line:
                    continue
                line_lower = line.lower()

                for keyword, category in section_map.items():
                    if keyword in line_lower:
                        current_category = category
                        break

                if current_category:
                    # Try to find "Question ... Answer ..." pattern anywhere in line
                    parts = re.split(r'model answer[:\s]+', line, flags=re.IGNORECASE)
                    if len(parts) == 2:
                        q_text = re.sub(r'^[\d\.\-\*Qq]+\s*', '', parts[0]).strip()
                        a_text = parts[1].strip()
                        if q_text and a_text:
                            result[current_category].append({
                                "question": q_text,
                                "model_answer": a_text
                            })

        return result

    def summarize_paper(self, paper_metadata: dict, sections: dict) -> Dict[str, Any]:
        """
        Summarizes the paper based on its abstract, introduction, and conclusion.
        Returns the parsed summary paragraph and bulleted contributions.
        """
        kwargs = {
            "abstract": paper_metadata.get("abstract", "") or sections.get("abstract", ""),
            "introduction": sections.get("introduction", ""),
            "conclusion": sections.get("conclusion", "")
        }
        
        response = self._call_with_retry(SUMMARY_PROMPT, kwargs)
        text = response.content
        
        result = {
            "summary": "",
            "contributions": []
        }
        
        # Split output safely into summary vs key contributions
        parts = re.split(r'Key Contributions:', text, flags=re.IGNORECASE)
        if len(parts) == 2:
            summary_part = parts[0].replace("Summary:", "").strip()
            # Clean up potential leading/trailing newlines or bold markers if LLM uses markdown
            result["summary"] = re.sub(r'^\*\*Summary:\*\*\s*', '', summary_part, flags=re.IGNORECASE).strip()
            
            contrib_part = parts[1].strip()
            for line in contrib_part.split("\n"):
                line = line.strip()
                # Match standard markdown list markers
                if line.startswith("-") or line.startswith("*"):
                    # Remove the marker
                    cleaned = re.sub(r'^[-*]\s*', '', line).strip()
                    if cleaned:
                        result["contributions"].append(cleaned)
        else:
            # Fallback if LLM output fails exact formatting
            result["summary"] = text.strip()
            
        return result
