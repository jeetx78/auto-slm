import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_PATH = os.path.join(BASE_DIR, "rag_docs", "docs.txt")
INDEX_PATH = os.path.join(BASE_DIR, "rag_docs", "index.faiss")
META_PATH = os.path.join(BASE_DIR, "rag_docs", "meta.pkl")

def main():
    with open(DOCS_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    # Simple chunking
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedder.encode(chunks, convert_to_numpy=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print("✅ RAG index built")

if __name__ == "__main__":
    main()
