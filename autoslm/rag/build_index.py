import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH = os.path.join(BASE_DIR, "docs.txt")
RAG_DIR = os.path.join(BASE_DIR, "rag_docs")
INDEX_PATH = os.path.join(RAG_DIR, "index.faiss")
META_PATH = os.path.join(RAG_DIR, "meta.pkl")

os.makedirs(RAG_DIR, exist_ok=True)

def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def main():
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ RAG built with {len(chunks)} chunks")
def build_index_for_project(project_id: str, source_path: str):
    out_dir = f"autoslm/rag/rag_docs/{project_id}"
    os.makedirs(out_dir, exist_ok=True)

    texts = load_and_chunk(source_path)
    embeddings = embedder.encode(texts, normalize_embeddings=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, f"{out_dir}/index.faiss")

    with open(f"{out_dir}/meta.pkl", "wb") as f:
        pickle.dump(texts, f)

if __name__ == "__main__":
    main()
