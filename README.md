# 📄 Research Paper RAG

A powerful, end-to-end Retrieval-Augmented Generation (RAG) system built specifically to ingest, process, and deeply interact with academic papers. 

## Features
- **Intelligent Ingestion:** Automatically downloads papers from ArXiv, parses complex academic PDF layouts, extracts metadata/citations, and chunks them using semantic section-aware boundaries.
- **Contextual QA:** Query your document base using either standard academic formatting or a jargon-free "Beginner Mode."
- **Paper Comparison:** Side-by-side comparison of two specific papers, evaluating approaches, strengths, and weaknesses into a Markdown table.
- **Interview Prep Generator:** Synthesizes conceptual, implementation, and critical thinking questions from a specific paper or broad topic.
- **Interactive UI:** A full-featured Streamlit frontend with dynamic source tracking, similarity progress bars, and high-density summary modals.

## Architecture

```text
[ INGESTION PIPELINE ]
PDF Document → PDFParser (PyMuPDF) → PaperChunker (Semantic Split)
                                                 ↓
                                      PaperEmbedder (all-mpnet-base-v2)
                                                 ↓
                                         ChromaDB (Vector Store)

[ RETRIEVAL & GENERATION PIPELINE ]
User Query → Embedder → PaperRetriever (Cosine Similarity)
                                                 ↓
                                       RAGGenerator (ChatGroq)
                                                 ↓
                                            Response (UI)
```

## Setup Instructions

1. **Install Dependencies**
   Ensure you have Python 3.9+ installed.
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

3. **Download Papers**
   Use the provided script to automatically fetch recent benchmark papers from ArXiv:
   ```bash
   python scripts/download_papers.py
   ```

4. **Ingest Papers**
   Process the downloaded PDFs into the ChromaDB vector index. This script parses, chunks, embeds, and tracks metadata:
   ```bash
   python scripts/ingest_all.py
   ```

5. **Run the Application**
   You need two terminal windows to run both the FastAPI backend and Streamlit frontend.

   **Terminal 1 (Backend):**
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

   **Terminal 2 (Frontend):**
   ```bash
   streamlit run app/streamlit_app.py
   ```

## Example Queries

### Ask Papers
- **Standard Mode:** "How does the attention mechanism in the Transformer architecture solve the bottleneck problem of RNNs?"
- **Beginner Mode:** "Explain diffusion models to me like I'm a high schooler."
  *Expected Output:* An answer grounded in citations, followed by a "Simple version:" TL;DR and simple analogies.

### Compare Papers
- **Inputs:** Select "Attention Is All You Need" and "BERT: Pre-training of Deep Bidirectional Transformers".
- **Aspect:** "Contextual representation strategy"
  *Expected Output:* A Markdown table comparing approaches, strengths, and weaknesses, followed by a definitive summary declaring the "winner" for that aspect.

### Interview Prep
- **Input:** Select "InstructGPT" or type a broad topic like "RLHF".
  *Expected Output:* 10 questions categorized into Conceptual, Implementation, and Critical Thinking, complete with hidden Model Answers.

## Tech Stack

| Component | Library/Tool | Purpose |
| :--- | :--- | :--- |
| **Backend API** | FastAPI, Uvicorn | High-performance async REST API bridging ML services and UI. |
| **Frontend** | Streamlit | Rapid, interactive web application UI. |
| **PDF Parsing** | PyMuPDF (fitz), PyPDF2 | Extracts raw text, section boundaries, and citations from academic PDFs. |
| **Embeddings** | Sentence-Transformers | Uses `all-mpnet-base-v2` to vectorize text chunks for semantic search. |
| **Vector DB** | ChromaDB | Persists and queries embeddings using cosine distance. |
| **LLM Orchestration**| LangChain, ChatGroq | Manages prompt templates, retries, and structured output generation using `llama3-8b-8192`. |
| **Data Gathering** | ArXiv API | Automatically searches and downloads ML papers and metadata. |
