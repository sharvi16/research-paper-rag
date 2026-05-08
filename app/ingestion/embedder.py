import json
from pathlib import Path
from typing import List, Dict, Any

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class PaperEmbedder:
    """
    Handles generating dense vector embeddings for academic paper chunks
    and indexing them into a persistent ChromaDB store.
    """

    def __init__(self, persist_directory: str = "data/chroma_store/", collection_name: str = "ml_papers"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Load the requested embedding model ('all-mpnet-base-v2' is highly rated for academic/general text)
        print("Loading SentenceTransformer model 'all-mpnet-base-v2'...")
        self.model = SentenceTransformer('all-mpnet-base-v2')
        
        # Initialize Persistent ChromaDB Client
        db_path = Path(self.persist_directory).resolve()
        db_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(db_path))
        
        # Create or get collection ensuring cosine similarity is used
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def get_embedding(self, text: str) -> List[float]:
        """
        Generates a single embedding array for the given text.
        """
        return self.model.encode(text).tolist()

    def _sanitize_metadata(self, metadata: dict) -> dict:
        """
        ChromaDB restricts metadata values to str, int, float, or bool.
        This securely serializes complex objects like lists of authors or categories.
        """
        sanitized = {}
        for k, v in metadata.items():
            if v is None:
                sanitized[k] = ""
            elif isinstance(v, (list, dict)):
                sanitized[k] = json.dumps(v)
            elif isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            else:
                sanitized[k] = str(v)
        return sanitized

    def index_papers(self, chunks: List[Dict[str, Any]], collection_name: str = None):
        """
        Indexes chunks into ChromaDB in batches of 100, skipping existing chunks.
        """
        # Allow overriding the collection temporarily
        if collection_name and collection_name != self.collection_name:
            self.collection_name = collection_name
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
        if not chunks:
            return
            
        # Get existing IDs to prevent duplicates
        existing_data = self.collection.get(include=[])
        existing_ids = set(existing_data["ids"])
        
        chunks_to_index = [c for c in chunks if c["chunk_id"] not in existing_ids]
        
        if not chunks_to_index:
            print("All chunks are already indexed.")
            return
            
        print(f"Indexing {len(chunks_to_index)} new chunks into '{self.collection_name}'...")
        
        batch_size = 100
        for i in tqdm(range(0, len(chunks_to_index), batch_size), desc="Indexing batches"):
            batch = chunks_to_index[i:i + batch_size]
            
            ids = [c["chunk_id"] for c in batch]
            texts = [c["text"] for c in batch]
            
            # Encode text batch synchronously
            embeddings = self.model.encode(texts).tolist()
            
            # Filter chunk_id and text out of metadata since they are standard fields
            metadatas = [
                self._sanitize_metadata({k: v for k, v in c.items() if k not in ["chunk_id", "text"]})
                for c in batch
            ]
            
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Calculates and returns statistics about the indexed documents.
        """
        count = self.collection.count()
        
        # To determine unique papers, we fetch metadata and extract paper_ids
        all_data = self.collection.get(include=["metadatas"])
        unique_papers = set()
        
        if all_data and all_data["metadatas"]:
            for meta in all_data["metadatas"]:
                if meta and "paper_id" in meta:
                    unique_papers.add(meta["paper_id"])
                    
        stats = {
            "collection_name": self.collection_name,
            "total_documents": count,
            "unique_papers": len(unique_papers)
        }
        
        return stats

    @classmethod
    def from_existing(cls, persist_directory: str = "data/chroma_store/", collection_name: str = "ml_papers"):
        """
        Alternative constructor to load an already-built index.
        Functionally the same as init because the chromadb client is persistent,
        but it makes semantic usage clearer.
        """
        print(f"Loading existing database from {persist_directory}...")
        return cls(persist_directory=persist_directory, collection_name=collection_name)
