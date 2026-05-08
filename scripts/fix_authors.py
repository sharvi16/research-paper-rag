"""
One-time script to clean up author fields in existing metadata JSON files.
Run this once from ml-paper-rag/ directory: python scripts/fix_authors.py
"""
import json
import re
from pathlib import Path


JUNK_WORDS = [
    "university", "department", "institute", "laboratory", "lab",
    "school", "college", "research", "center", "centre", "inc",
    "google", "openai", "meta", "microsoft", "deepmind", "berkeley",
    "stanford", "mit", "cmu", "nyu", "harvard", "oxford", "cambridge",
    "http", "arxiv", "equal contribution", "correspondence",
    "work done", "this work", "preprint", "anonymous"
]

def clean_author_name(name: str) -> str:
    """Clean a single author name string."""
    # Remove symbols like ∗, †, ‡, ⋆, ♠, ♣, numbers used as footnote markers
    name = re.sub(r'[∗†‡⋆♠♣♦♥\*\d]+', '', name)
    # Remove email addresses
    name = re.sub(r'\S+@\S+', '', name)
    # Remove content in parentheses or brackets
    name = re.sub(r'[\(\[\{].*?[\)\]\}]', '', name)
    # Normalize whitespace
    name = ' '.join(name.split())
    return name.strip()


def is_valid_author(name: str) -> bool:
    """Check if a string looks like a real person's name."""

    if not name or len(name) < 3:
        return False

    # Too long
    if len(name) > 40:
        return False

    name_lower = name.lower()

    # Junk institution/location words
    if any(junk in name_lower for junk in JUNK_WORDS):
        return False

    # Reject common affiliation/location phrases
    invalid_exact = {
        "mountain view",
        "new york",
        "california",
        "vector space",
        "google inc",
        "openai",
        "meta ai"
    }

    if name_lower in invalid_exact:
        return False

    # Must contain alphabetic chars only
    if not re.match(r'^[A-Za-z\-\.\s]+$', name):
        return False

    # Usually author names have 2-4 words
    words = name.split()

    if len(words) < 2 or len(words) > 4:
        return False

    # Every word should start uppercase
    for w in words:
        if not w[0].isupper():
            return False

    return True


def fix_authors_in_json(json_path: Path) -> bool:
    """Fix author field in a single JSON file. Returns True if changed."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    original_authors = data.get("authors", [])
    
    # Flatten — sometimes authors come as one big string
    all_names = []
    for entry in original_authors:
        # Split by common delimiters
        parts = re.split(r',|;|\band\b', entry)
        all_names.extend(parts)

    cleaned = []
    for name in all_names:
        name = clean_author_name(name)
        if is_valid_author(name):
            cleaned.append(name)

    # If cleaning wiped everything out, keep original but stripped
    if not cleaned:
        cleaned = ["Unknown"]

    if cleaned == original_authors:
        return False

    data["authors"] = cleaned
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return True


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    papers_dir = base_dir / "data" / "papers"
    jsons = list(papers_dir.glob("*.json"))

    if not jsons:
        print("No JSON files found.")
    else:
        fixed, skipped = 0, 0
        for json_path in jsons:
            changed = fix_authors_in_json(json_path)
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            status = "Fixed" if changed else "⏭  OK   "
            print(f"  {status} {json_path.name}")
            print(f"          Authors: {', '.join(data['authors'][:4])}")
            if fixed_count := (fixed + 1 if changed else fixed):
                fixed = fixed_count - (0 if changed else 1)
            skipped += 0 if changed else 1

        print(f"\n{'='*50}")
        print(f"Done: {fixed} fixed | {skipped} already clean")
        print("\nRemember to re-ingest papers so ChromaDB picks up")
        print("   the cleaned authors: python scripts/ingest_all.py")