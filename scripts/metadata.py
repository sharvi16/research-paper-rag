import json
import re
from pathlib import Path
import fitz  # PyMuPDF


def extract_metadata_from_pdf(pdf_path: Path) -> dict:
    """
    Extracts metadata directly from a PDF file.
    No API calls, no internet needed.
    """
    doc = fitz.open(str(pdf_path))

    # --- 1. Try built-in PDF metadata first ---
    pdf_meta = doc.metadata
    title = pdf_meta.get("title", "").strip()
    author = pdf_meta.get("author", "").strip()

    # --- 2. Fall back to reading first page text ---
    first_page_text = doc[0].get_text() if len(doc) > 0 else ""
    second_page_text = doc[1].get_text() if len(doc) > 1 else ""
    full_first_two_pages = first_page_text + "\n" + second_page_text

    lines = [l.strip() for l in first_page_text.splitlines() if l.strip()]

    # If PDF metadata title is empty or garbage, grab first non-empty line
    if not title or len(title) < 5:
        title = lines[0] if lines else pdf_path.stem

    # If author is empty, try to find author-like lines
    # (lines before "Abstract" that aren't the title)
    authors = []
    if author:
        authors = [a.strip() for a in re.split(r",|;|&| and ", author) if a.strip()]
    else:
        # Heuristic: lines 1-6 before "abstract" keyword are often authors
        abstract_idx = next(
            (i for i, l in enumerate(lines) if "abstract" in l.lower()), 6
        )
        candidate_lines = lines[1:abstract_idx]
        for line in candidate_lines:
            # Author lines are usually short and don't contain these words
            if not any(w in line.lower() for w in ["university", "department",
                       "institute", "lab", "school", "http", "@", "arxiv"]):
                if len(line) < 80:
                    authors.append(line)

    # --- 3. Extract abstract ---
    abstract = ""
    abstract_match = re.search(
        r"abstract[:\s\n]+(.*?)(?=\n\n|\nintroduction|\n1\.|\n1\s)",
        full_first_two_pages,
        re.IGNORECASE | re.DOTALL
    )
    if abstract_match:
        abstract = abstract_match.group(1).replace("\n", " ").strip()
        abstract = re.sub(r"\s+", " ", abstract)

    # --- 4. Guess arxiv ID from filename ---
    arxiv_id = pdf_path.stem  # filename without .pdf

    # --- 5. Total pages ---
    num_pages = len(doc)
    doc.close()

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors if authors else ["Unknown"],
        "abstract": abstract if abstract else "Abstract not extracted.",
        "published_date": "",
        "categories": [],
        "url": f"https://arxiv.org/abs/{arxiv_id}" if re.match(r"\d{4}\.\d+", arxiv_id) else "",
        "num_pages": num_pages,
        "source": "extracted_from_pdf"
    }


def generate_all_metadata(papers_dir: str = "data/papers/"):
    out_dir = Path(papers_dir)
    pdfs = list(out_dir.glob("*.pdf"))

    if not pdfs:
        print(f"❌ No PDFs found in {out_dir.resolve()}")
        return

    print(f"Found {len(pdfs)} PDFs in {out_dir.resolve()}\n")

    success, skipped, failed = 0, 0, 0

    for pdf_path in pdfs:
        json_path = pdf_path.with_suffix(".json")

        if json_path.exists():
            print(f"  ⏭  Skipping {pdf_path.name} (metadata already exists)")
            skipped += 1
            continue

        try:
            metadata = extract_metadata_from_pdf(pdf_path)

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            print(f"  ✅ {pdf_path.name}")
            print(f"     Title   : {metadata['title'][:70]}")
            print(f"     Authors : {', '.join(metadata['authors'][:3])}")
            print(f"     Pages   : {metadata['num_pages']}")
            print(f"     Abstract: {'Yes' if metadata['abstract'] != 'Abstract not extracted.' else 'No'}\n")
            success += 1

        except Exception as e:
            print(f"  ❌ Failed on {pdf_path.name}: {e}")
            failed += 1

    print("=" * 50)
    print(f"Done: {success} generated | {skipped} skipped | {failed} failed")
    print(f"JSON files saved alongside PDFs in {out_dir.resolve()}")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    papers_dir = base_dir / "data" / "papers"
    generate_all_metadata(str(papers_dir))