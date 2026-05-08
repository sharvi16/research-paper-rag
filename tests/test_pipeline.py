import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.api.routes import get_services
from app.ingestion.pdf_parser import PDFParser
from app.ingestion.chunker import PaperChunker
from app.ingestion.embedder import PaperEmbedder
from app.retrieval.retriever import PaperRetriever
from app.features.comparator import PaperComparator

# ─────────────────────────────────────────
# 1. Test PDF Parsing
# ─────────────────────────────────────────
@patch("app.ingestion.pdf_parser.fitz")
def test_pdf_parsing(mock_fitz):
    # Mock PyMuPDF behavior
    mock_doc = MagicMock()
    mock_doc.page_count = 2
    
    mock_page_1 = MagicMock()
    mock_page_1.get_text.return_value = "Abstract\nThis is a mock abstract for the ML paper."
    
    mock_page_2 = MagicMock()
    mock_page_2.get_text.return_value = "1. Introduction\nHere we introduce the core concepts."
    
    mock_doc.__iter__.return_value = [mock_page_1, mock_page_2]
    mock_fitz.open.return_value = mock_doc
    
    parser = PDFParser()
    res = parser.parse("dummy.pdf")
    
    assert res["num_pages"] == 2
    assert "Abstract" in res["text"]
    assert "1. Introduction" in res["text"]
    assert len(res["pages"]) == 2

# ─────────────────────────────────────────
# 2. Test Chunking
# ─────────────────────────────────────────
def test_chunking():
    chunker = PaperChunker()
    
    # A single paragraph needs >= 50 words to be kept.
    # We will provide a large paragraph to test splitting, and two small ones to test merging.
    long_text = "word " * 650
    short_text_1 = "This is a short paragraph but needs to be merged. " * 3 # ~24 words
    short_text_2 = "This is another short paragraph to be merged into one. " * 3 # ~24 words
    
    parsed = {
        "text": f"Introduction\n\n{long_text}\n\n{short_text_1}\n\n{short_text_2}",
        "metadata": {"title": "Test Title", "paper_id": "test_id"},
        "file_path": "test.pdf"
    }
    
    chunks = chunker.chunk(parsed)
    assert len(chunks) > 0
    # Long text should be split, short texts merged
    
    # Just verify output schema
    assert "text" in chunks[0]
    assert "paper_title" in chunks[0]
    assert chunks[0]["paper_title"] == "Test Title"

# ─────────────────────────────────────────
# 3. Test Embedding
# ─────────────────────────────────────────
@patch("app.ingestion.embedder.SentenceTransformer")
@patch("app.ingestion.embedder.chromadb.PersistentClient")
def test_embedding(mock_chromadb_client, mock_st):
    # Mock SentenceTransformer embedding output
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 768]
    mock_st.return_value = mock_model
    
    # Mock ChromaDB
    mock_client = MagicMock()
    mock_chromadb_client.return_value = mock_client
    
    embedder = PaperEmbedder()
    emb = embedder.get_embedding("Test semantic text")
    
    assert len(emb) == 768
    assert emb[0] == 0.1

# ─────────────────────────────────────────
# 4. Test Retrieval
# ─────────────────────────────────────────
def test_retrieval():
    mock_embedder = MagicMock()
    mock_embedder.get_embedding.return_value = [0.1] * 768
    
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids": [["chunk_1"]],
        "documents": [["Retrieved text"]],
        "metadatas": [[{"paper_title": "Paper A", "section": "Methodology"}]],
        "distances": [[0.5]]
    }
    mock_embedder.collection = mock_collection
    
    retriever = PaperRetriever(mock_embedder)
    chunks = retriever.retrieve("query", k=1)
    
    assert len(chunks) == 1
    assert chunks[0]["chunk_id"] == "chunk_1"
    assert chunks[0]["text"] == "Retrieved text"
    assert chunks[0]["paper_title"] == "Paper A"
    # Cosine similarity is typically 1.0 - distance in Chroma
    assert chunks[0]["similarity"] == 1.0 - 0.5

# ─────────────────────────────────────────
# 5. Test API: /query
# ─────────────────────────────────────────
def test_api_query():
    mock_embedder = MagicMock()
    mock_retriever = MagicMock()
    mock_generator = MagicMock()
    
    mock_retriever.retrieve.return_value = [{"chunk_id": "1", "text": "Sample context", "paper_title": "Paper A"}]
    mock_retriever.build_context.return_value = "Sample context"
    mock_generator.answer_question.return_value = {"answer": "Mock LLM Answer", "tokens_used": 150}
    
    # Patch get_services globally for the router
    with patch("app.api.routes.get_services", return_value=(mock_embedder, mock_retriever, mock_generator)):
        client = TestClient(app)
        res = client.post("/query", json={"question": "What is attention?", "mode": "standard"})
        
        assert res.status_code == 200
        data = res.json()
        assert data["answer"] == "Mock LLM Answer"
        assert data["mode"] == "standard"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["title"] == "Paper A"

# ─────────────────────────────────────────
# 6. Test API: /compare
# ─────────────────────────────────────────
def test_api_compare():
    mock_embedder = MagicMock()
    mock_retriever = MagicMock()
    mock_generator = MagicMock()
    
    # The comparator feature runs internally, so we mock its public method
    with patch("app.features.comparator.PaperComparator.compare") as mock_compare:
        mock_compare.return_value = {
            "comparison_table": "| Paper | Approach |\n|---|---|", 
            "summary": "Paper A wins.",
            "winner_for_aspect": "Paper A"
        }
        
        with patch("app.api.routes.get_services", return_value=(mock_embedder, mock_retriever, mock_generator)):
            client = TestClient(app)
            res = client.post("/compare", json={"paper_id_1": "1", "paper_id_2": "2", "aspect": "Architecture"})
            
            assert res.status_code == 200
            data = res.json()
            assert data["comparison_table"] == "| Paper | Approach |\n|---|---|"
            assert data["summary"] == "Paper A wins."

# ─────────────────────────────────────────
# 7. Test API: /interview
# ─────────────────────────────────────────
def test_api_interview():
    mock_embedder = MagicMock()
    mock_retriever = MagicMock()
    mock_generator = MagicMock()
    
    with patch("app.features.interview_gen.InterviewGenerator.generate") as mock_gen:
        mock_gen.return_value = {
            "conceptual": [{"question": "Q1", "model_answer": "A1"}],
            "implementation": [],
            "critical": []
        }
        
        with patch("app.api.routes.get_services", return_value=(mock_embedder, mock_retriever, mock_generator)):
            client = TestClient(app)
            res = client.post("/interview", json={"paper_id": "1"})
            
            assert res.status_code == 200
            data = res.json()
            assert len(data["conceptual"]) == 1
            assert data["conceptual"][0]["question"] == "Q1"

# ─────────────────────────────────────────
# 8. Test Feature: PaperComparator Logic
# ─────────────────────────────────────────
def test_feature_comparator():
    mock_retriever = MagicMock()
    mock_generator = MagicMock()
    
    # Mock retrieval behavior
    mock_retriever.retrieve_for_comparison.return_value = {
        "paper1_chunks": [{"text": "P1 chunk", "paper_title": "Paper One"}],
        "paper2_chunks": [{"text": "P2 chunk", "paper_title": "Paper Two"}]
    }
    mock_retriever.build_context.side_effect = ["Context P1", "Context P2"]
    
    # Mock generator behavior
    mock_generator.compare_papers.return_value = "| P1 | P2 |"
    # Mock the secondary LLM call that fetches the summary
    mock_generator.answer_question.return_value = {"answer": "Paper One is far superior for this task."}
    
    comparator = PaperComparator()
    result = comparator.compare("id_1", "id_2", "methodology", mock_retriever, mock_generator)
    
    assert result["comparison_table"] == "| P1 | P2 |"
    assert result["summary"] == "Paper One is far superior for this task."
    assert result["winner_for_aspect"] == "Paper One"
