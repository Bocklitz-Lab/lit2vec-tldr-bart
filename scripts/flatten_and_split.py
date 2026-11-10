"""Flatten the annotated records and split them into train / validation / test JSONL files.

Input: one JSON file with a list of records. Each record has `corpus_id`, `abstract`,
`summary` and a `reduced_metadata` object (doi, title, authors, venue, year,
fields_of_study, publication_date, url and license with `license`, `publisher`, `oa_pdf`).

Output: train.jsonl, validation.jsonl and test.jsonl. The split is random with a fixed seed
and uses the ratios 90% / 5% / 5%. For the 19,992 records of the released dataset this gives
17,992 / 999 / 1,001 records.

Note: `fields_of_study` keeps one entry per source, so a field can appear more than once, and
`license_url` is the open-access PDF link (`oa_pdf`), not a link to the license text.
"""
import json
import random
import argparse
from pathlib import Path
from typing import List

# Set seed for reproducibility
random.seed(42)

def flatten_record(record: dict) -> dict:
    """Flatten a single record."""
    metadata = record.get("reduced_metadata", {})
    license_info = metadata.get("license", {})
    authors = [a["name"] for a in metadata.get("authors", [])]
    fields_of_study = [f["category"] for f in metadata.get("fields_of_study", [])]

    return {
        "corpus_id": record.get("corpus_id"),
        "doi": metadata.get("doi"),
        "title": metadata.get("title"),
        "authors": authors,
        "venue": metadata.get("venue"),
        "year": metadata.get("year"),
        "fields_of_study": fields_of_study,
        "publication_date": metadata.get("publication_date"),
        "abstract": record.get("abstract"),
        "summary": record.get("summary"),
        "license_type": license_info.get("license"),
        "license_publisher": license_info.get("publisher"),
        "license_url": license_info.get("oa_pdf"),
        "source_url": metadata.get("url")
    }

def split_dataset(data: List[dict], train_ratio=0.9, val_ratio=0.05):
    """Split data into train/val/test."""
    random.shuffle(data)
    total = len(data)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)
    return data[:train_end], data[train_end:val_end], data[val_end:]

def write_jsonl(data: List[dict], output_path: Path):
    """Write list of dicts to a JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for record in data:
            json.dump(record, f, ensure_ascii=False)
            f.write("\n")

def main(args):
    # Load full input JSON file
    with open(args.input, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} raw records")

    # Flatten each record
    flat_data = [flatten_record(rec) for rec in raw_data]

    # Split
    train_data, val_data, test_data = split_dataset(flat_data)

    # Ensure output dir exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write splits
    write_jsonl(train_data, output_dir / "train.jsonl")
    write_jsonl(val_data, output_dir / "validation.jsonl")
    write_jsonl(test_data, output_dir / "test.jsonl")

    print(f"Written {len(train_data)} train, {len(val_data)} validation, {len(test_data)} test records to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flatten and split JSON dataset")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save JSONL files")
    args = parser.parse_args()
    main(args)
