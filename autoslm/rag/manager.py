import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

class RAGManager:
    def __init__(self, dim=384):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.rag_dir = os.path.join(self.base_dir, "rag_docs")

        os.makedirs(self.rag_dir, exist_ok=True)

        self.index_path = os.path.join(self.rag_dir, "index.faiss")
        self.meta_path = os.path.join(self.rag_dir, "meta.pkl")

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.dim = dim

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.chunks = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(dim)
            self.chunks = []

    def add_text(self, text: str):
        emb = self.embedder.encode([text], normalize_embeddings=True)
        self.index.add(np.array(emb))
        self.chunks.append(text)

        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def search(self, query: str, k=3):
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        _, idxs = self.index.search(q_emb, k)
        return [self.chunks[i] for i in idxs[0]]
