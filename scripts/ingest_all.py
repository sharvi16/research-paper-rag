import os
import time
import json
import sys
from pathlib import Path
from tqdm import tqdm
import pandas as pd

# Ensure we can import from the app module
sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.ingestion.pdf_parser import PDFParser
from app.ingestion.chunker import PaperChunker
from app.ingestion.embedder import PaperEmbedder

def main():
    print("🚀 Starting batch ingestion pipeline...")
    start_time = time.time()
    
    data_dir = Path("data/papers")
    if not data_dir.exists():
        print(f"Directory {data_dir} does not exist. Please run download_papers.py first.")
        return
        
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {data_dir}.")
        return
        
    print(f"Found {len(pdf_files)} PDF files. Initializing ML models (this may take a moment)...")
    
    parser = PDFParser()
    chunker = PaperChunker()
    embedder = PaperEmbedder.from_existing()
    
    results = []
    
    for pdf_path in tqdm(pdf_files, desc="Ingesting papers"):
        paper_id = pdf_path.stem
        result = {
            "Paper ID": paper_id,
            "Status": "Success",
            "Chunks": 0,
            "Error": ""
        }
        
        try:
            # Check if already indexed to prevent duplication
            existing = embedder.collection.get(where={"paper_id": paper_id}, limit=1)
            if existing and existing.get("ids"):
                result["Status"] = "Skipped (Already Indexed)"
                results.append(result)
                continue
                
            # Parse PDF
            parsed = parser.parse(str(pdf_path))
            
            # Load companion metadata JSON if it was downloaded via arxiv script
            json_path = pdf_path.with_suffix(".json")
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    # Merge metadata
                    parsed["metadata"].update(meta)
                    
            if not parsed.get("metadata", {}).get("title"):
                if not parsed.get("metadata"):
                    parsed["metadata"] = {}
                parsed["metadata"]["title"] = paper_id
                
            parsed["file_path"] = str(pdf_path)
            
            # Chunk the text semantically
            chunks = chunker.chunk(parsed)
            
            # Embed and Index into ChromaDB
            if chunks:
                embedder.index_papers(chunks)
                
            result["Chunks"] = len(chunks)
            
        except Exception as e:
            result["Status"] = "Failed"
            result["Error"] = str(e)
            
        results.append(result)
        
    total_time = time.time() - start_time
    
    # ─────────────────────────────────────────
    # Summary Table Generation
    # ─────────────────────────────────────────
    print(f"\n✅ Ingestion complete in {total_time:.2f} seconds.")
    print("\n" + "="*50)
    print("📊 BATCH SUMMARY")
    print("="*50)
    
    df = pd.DataFrame(results)
    
    if not df.empty:
        # Group summary counts
        print(df["Status"].value_counts().to_string())
        print(f"\nTotal chunks newly indexed: {df['Chunks'].sum()}")
        print("\n🔍 Details:")
        # Display details, truncating long error messages if they exist
        pd.set_option("display.max_colwidth", 40)
        print(df.to_string(index=False))
        
if __name__ == "__main__":
    main()
