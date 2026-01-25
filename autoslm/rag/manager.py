import faiss
import numpy as np
from collections import defaultdict
from autoslm.rag.embedder import Embedder

class RAGManager:
    def __init__(self):
        self.embedder = Embedder()
        self.indexes = {}
        self.texts = defaultdict(list)

    def _get_index(self, project_id: str):
        if project_id not in self.indexes:
            self.indexes[project_id] = faiss.IndexFlatIP(384)
        return self.indexes[project_id]

    def add_documents(self, project_id: str, docs: list[str]):
        index = self._get_index(project_id)
        all_chunks = []
        for doc in docs:
            chunks=chunk_text(doc)
            all_chunks.append(doc)
        embeddings = self.embedder.embed(all_chunks)

        index.add(np.array(embeddings))
        self.texts[project_id].extend(docs)

    def search(self, project_id: str, query: str, k: int = 3):
        if project_id not in self.indexes:
            return []

        q_emb = np.array(self.embedder.embed([query]))
        index = self.indexes[project_id]

        _, idxs = index.search(q_emb, k)
        return [
            self.texts[project_id][i]
            for i in idxs[0]
            if i < len(self.texts[project_id])
        ]
