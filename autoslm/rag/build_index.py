import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "rag_docs", "input")
INDEX_PATH = os.path.join(BASE_DIR, "rag_docs", "index.faiss")
META_PATH = os.path.join(BASE_DIR, "rag_docs", "meta.pkl")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def load_texts():
    texts = []
    for file in os.listdir(DOCS_DIR):
        if file.endswith(".txt"):
            with open(os.path.join(DOCS_DIR, file), "r", encoding="utf-8") as f:
                texts.append(f.read())
    return texts

def main():
    texts = load_texts()
    if not texts:
        raise RuntimeError("No documents found")

    embeddings = embedder.encode(texts)

    if os.path.exists(INDEX_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
    else:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        meta = []

    index.add(embeddings)
    meta.extend(texts)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print(f"✅ Added {len(texts)} documents")

if __name__ == "__main__":
    main()
