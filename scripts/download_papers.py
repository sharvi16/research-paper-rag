import os
import time
import json
import logging
from pathlib import Path
import arxiv
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def search_and_download(query, max_papers=10, output_dir="data/papers/"):
    """
    Searches ArXiv for the given query, downloads PDFs, and saves metadata.
    Includes rate limiting to avoid HTTP 429 errors.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Conservative client settings to avoid rate limiting
    client = arxiv.Client(
        page_size=5,
        delay_seconds=10,
        num_retries=3
    )

    search = arxiv.Search(
        query=query,
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.Relevance
    )

    # Fetch results with retry on 429
    results = []
    for attempt in range(2):
        try:
            results = list(client.results(search))
            break
        except arxiv.UnexpectedEmptyPageError:
            logging.warning(f"Empty page for query: '{query}', skipping.")
            return
        except Exception as e:
            if "429" in str(e):
                logging.warning(f"Rate limited (429) on attempt {attempt + 1}. Waiting 60 seconds...")
                time.sleep(60)
                if attempt == 1:
                    logging.error(f"Skipping query '{query}' after retry.")
                    return
            else:
                logging.error(f"Error searching for query '{query}': {e}")
                return

    if not results:
        logging.warning(f"No results found for query: '{query}'")
        return

    print(f"\nFound {len(results)} papers for: '{query}'")

    for result in tqdm(results, desc="Downloading", unit="paper"):
        arxiv_id = result.get_short_id()
        pdf_path = out_dir / f"{arxiv_id}.pdf"
        meta_path = out_dir / f"{arxiv_id}.json"

        # Skip if already fully downloaded
        if pdf_path.exists() and meta_path.exists():
            tqdm.write(f"  Skipping {arxiv_id} (already exists)")
            continue

        # Save metadata
        metadata = {
            "arxiv_id": arxiv_id,
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "abstract": result.summary,
            "published_date": result.published.isoformat() if result.published else None,
            "categories": result.categories,
            "url": result.entry_id
        }

        # Download PDF
        if not pdf_path.exists():
            try:
                result.download_pdf(dirpath=str(out_dir), filename=f"{arxiv_id}.pdf")
                tqdm.write(f"  Downloaded: {result.title[:60]}...")
            except Exception as e:
                logging.error(f"  Failed to download {arxiv_id}: {e}")
                continue

        # Save metadata JSON
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # 3 second delay between each PDF download
        time.sleep(3)


if __name__ == "__main__":
    queries = [
        "transformer attention mechanism",
        "large language models fine tuning",
        "diffusion models image generation",
        "reinforcement learning human feedback",
        "retrieval augmented generation"
    ]

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data" / "papers"

    for i, q in enumerate(queries):
        print(f"\n{'='*50}")
        print(f"Query {i+1}/{len(queries)}: '{q}'")
        print(f"{'='*50}")

        search_and_download(q, max_papers=10, output_dir=str(data_dir))

        # 10 second delay between queries (skip after last one)
        if i < len(queries) - 1:
            print(f"\nWaiting 10s before next query...")
            time.sleep(10)

    print("\n Done! Check data/papers/ for downloaded files.")