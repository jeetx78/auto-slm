import os
import json
import re
from pathlib import Path

CHUNK_SIZE = 500   # tokens approx (rough words proxy)
OVERLAP = 50

def read_text_file(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(text: str):
    words = text.split(" ")
    chunks = []

    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - OVERLAP

    return chunks

def ingest_directory(data_path: str, output_path: str):
    data_path = Path(data_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    file_count = 0

    for file in data_path.rglob("*"):
        if file.suffix.lower() not in [".txt", ".md"]:
            continue

        raw = read_text_file(file)
        cleaned = clean_text(raw)
        chunks = chunk_text(cleaned)

        for chunk in chunks:
            all_chunks.append({
                "source": str(file),
                "text": chunk
            })

        file_count += 1

    out_file = output_path / "dataset.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for row in all_chunks:
            f.write(json.dumps(row) + "\n")

    return {
        "files": file_count,
        "chunks": len(all_chunks),
        "output": str(out_file)
    }
