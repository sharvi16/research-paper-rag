import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Request, HTTPException, Query, UploadFile, File
from pydantic import BaseModel

# Top-level imports replacing inline imports
from app.features.explainer import BeginnerExplainer
from app.features.comparator import PaperComparator
from app.features.interview_gen import InterviewGenerator
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.chunker import PaperChunker

router = APIRouter()

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    question: str
    mode: str = "standard"  # "standard" or "beginner"
    paper_id: Optional[str] = None

class CompareRequest(BaseModel):
    paper_id_1: str
    paper_id_2: str
    aspect: str

class InterviewRequest(BaseModel):
    paper_id: str


# --- Helper ---
def get_services(request: Request):
    """Extracts globally initialized services from the FastAPI request state."""
    embedder = getattr(request.state, "embedder", None)
    retriever = getattr(request.state, "retriever", None)
    generator = getattr(request.state, "generator", None)
    
    if not embedder or not retriever or not generator:
        raise HTTPException(status_code=503, detail="ML Services are not fully initialized yet.")
        
    return embedder, retriever, generator


# --- Endpoints ---

@router.post("/query")
async def query_papers(request: Request, body: QueryRequest):
    try:
        _, retriever, generator = get_services(request)
        
        if body.mode == "beginner":
            explainer = BeginnerExplainer()
            res = explainer.explain(body.question, retriever, generator)
            # Explainer returns dict with answer, sources, mode, tokens
            res["mode"] = "beginner"
            return res
        else:
            # Standard mode
            chunks = retriever.retrieve(query=body.question, k=8, filter_paper_id=body.paper_id)
            
            if not chunks:
                return {
                    "answer": "No relevant context found in the database.",
                    "sources": [],
                    "mode": "standard"
                }
                
            context = retriever.build_context(chunks)
            unique_titles = list(set(c.get("paper_title", "Unknown") for c in chunks))
            paper_titles = ", ".join(unique_titles)
            
            res = generator.answer_question(
                question=body.question, 
                context=context, 
                paper_titles=paper_titles, 
                mode="standard"
            )
            
            sources = [
                {
                    "title": c.get("paper_title", "Unknown"), 
                    "section": c.get("section", "Unknown"), 
                    "similarity": c.get("similarity", 0.0),
                    "text": c.get("text", "")
                } 
                for c in chunks
            ]
            
            return {
                "answer": res["answer"],
                "sources": sources,
                "mode": "standard"
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compare")
async def compare_papers(request: Request, body: CompareRequest):
    try:
        _, retriever, generator = get_services(request)
        
        comparator = PaperComparator()
        result = comparator.compare(
            paper_id_1=body.paper_id_1,
            paper_id_2=body.paper_id_2,
            aspect=body.aspect,
            retriever=retriever,
            generator=generator
        )
        return {
            "comparison_table": result["comparison_table"],
            "summary": result["summary"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/interview")
async def generate_interview(request: Request, body: InterviewRequest):
    try:
        _, retriever, generator = get_services(request)
        
        interview_gen = InterviewGenerator()
        result = interview_gen.generate(
            paper_id=body.paper_id,
            retriever=retriever,
            generator=generator
        )
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/papers")
async def get_papers(request: Request):
    try:
        _, retriever, _ = get_services(request)
        papers = retriever.get_paper_list()
        
        # Format according to spec: {id, title, authors, date, abstract}
        return [{
            "id": p.get("paper_id"),
            "title": p.get("paper_title"),
            "authors": p.get("authors", []),
            "date": p.get("published_date", ""),
            "abstract": p.get("abstract", "")
        } for p in papers]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/paper/{paper_id}/summary")
async def get_paper_summary(paper_id: str, request: Request):
    try:
        _, retriever, generator = get_services(request)

        papers = retriever.get_paper_list()
        paper_meta = next((p for p in papers if p.get("paper_id") == paper_id), None)

        if not paper_meta:
            raise HTTPException(status_code=404, detail=f"Paper '{paper_id}' not found.")

        # Retrieve ALL chunks for this paper (not just top-k by similarity)
        chunks = retriever.retrieve(
            query=paper_meta.get("paper_title", "summary introduction conclusion abstract"),
            k=30,
            filter_paper_id=paper_id
        )

        sections = {"abstract": "", "introduction": "", "conclusion": ""}

        for c in chunks:
            sec = c.get("section", "").lower().strip()
            text = c.get("text", "")

            # Exact match first
            if sec in sections:
                sections[sec] += text + "\n\n"
                continue

            # Fuzzy match — section field may have extra words or differ slightly
            if "abstract" in sec:
                sections["abstract"] += text + "\n\n"
            elif "intro" in sec:
                sections["introduction"] += text + "\n\n"
            elif "conclu" in sec:
                sections["conclusion"] += text + "\n\n"

        # --- Fallback: if sections are still empty, re-parse the PDF directly ---
        if not any(sections.values()):
            try:
                # Find the PDF file on disk
                papers_dir = Path("data/papers")
                pdf_candidates = list(papers_dir.glob(f"{paper_id}*.pdf"))

                if pdf_candidates:
                    parser = PDFParser()
                    parsed = parser.parse(str(pdf_candidates[0]))
                    extracted = parser.extract_sections(parsed["text"])

                    sections["abstract"] = extracted.get("abstract", "")
                    sections["introduction"] = extracted.get("introduction", "")
                    sections["conclusion"] = extracted.get("conclusion", "")
            except Exception as parse_err:
                # Non-fatal — log and continue with whatever we have
                import logging
                logging.getLogger(__name__).warning(
                    f"PDF re-parse fallback failed for {paper_id}: {parse_err}"
                )

        # Last resort: use abstract from metadata if still empty
        if not sections["abstract"] and paper_meta.get("abstract"):
            sections["abstract"] = paper_meta["abstract"]

        result = generator.summarize_paper(paper_meta, sections)
        return result

    except HTTPException:   
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 

@router.get("/search")
async def search_papers(q: str, k: int = 5, request: Request = None):
    try:
        _, retriever, _ = get_services(request)
        chunks = retriever.retrieve(query=q, k=k)
        return chunks
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check(request: Request):
    try:
        embedder, _, generator = get_services(request)
    except HTTPException:
        return {"status": "starting up"}
        
    try:
        stats = embedder.get_collection_stats()
        
        return {
            "status": "healthy",
            "total_papers": stats.get("unique_papers", 0),
            "total_chunks": stats.get("total_documents", 0),
            "model": generator.model_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest")
async def ingest_paper(request: Request, file: UploadFile = File(...)):
    try:
        embedder, _, _ = get_services(request)
        
        out_dir = Path("data/papers")
        out_dir.mkdir(parents=True, exist_ok=True)
        file_path = out_dir / file.filename
        paper_id = file_path.stem
        
        # Handle duplicate uploads by checking if paper_id already exists in ChromaDB
        try:
            existing = embedder.collection.get(where={"paper_id": paper_id}, limit=1)
            if existing and existing.get("ids"):
                title = paper_id
                if existing.get("metadatas") and existing["metadatas"][0]:
                    title = existing["metadatas"][0].get("paper_title", paper_id)
                return {
                    "paper_id": paper_id,
                    "title": title,
                    "chunks_added": 0,
                    "status": "already_indexed"
                }
        except Exception:
            # Ignore if collection doesn't exist yet or query fails
            pass
        
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        parser = PDFParser()
        parsed = parser.parse(str(file_path))
        
        # Infer title and ID if missing
        if not parsed.get("metadata"):
            parsed["metadata"] = {}
        if not parsed["metadata"].get("title"):
            parsed["metadata"]["title"] = paper_id
            
        parsed["file_path"] = str(file_path)
            
        chunker = PaperChunker()
        chunks = chunker.chunk(parsed)
        
        embedder.index_papers(chunks)
        
        return {
            "paper_id": paper_id,
            "title": parsed["metadata"]["title"],
            "chunks_added": len(chunks),
            "status": "success"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
