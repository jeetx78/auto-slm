import os, faiss, pickle
from sentence_transformers import SentenceTransformer
from typing import List

class RAGManager:
    def __init__(self, base_dir="autoslm/rag/rag_docs"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

        self.index_path = os.path.join(base_dir, "index.faiss")
        self.meta_path = os.path.join(base_dir, "meta.pkl")

        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.meta_path, "rb") as f:
                self.chunks = pickle.load(f)
        else:
            self.index = faiss.IndexFlatIP(384)
            self.chunks = []

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def add_documents(self, project_id: str, texts: List[str]):
        docs = []
        for text in texts:
            for chunk in text.split("\n\n"):
                if len(chunk.strip()) > 50:
                    docs.append(f"[{project_id}] {chunk.strip()}")

        embeddings = self.embedder.encode(
            docs,
            normalize_embeddings=True
        )

        self.index.add(embeddings)
        self.chunks.extend(docs)
        self._save()

    def search(self, project_id: str, query: str, k=3):
        q_emb = self.embedder.encode(
            [query],
            normalize_embeddings=True
        )

        _, idxs = self.index.search(q_emb, k)
        results = []

        for i in idxs[0]:
            if self.chunks[i].startswith(f"[{project_id}]"):
                results.append(self.chunks[i])

        return results[:k]
