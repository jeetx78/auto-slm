import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTS_DIR = os.path.join(BASE_DIR, "projects")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

class RAGManager:
    def __init__(self):
        self.cache = {}

    def load_project(self, project_id: str):
        if project_id in self.cache:
            return self.cache[project_id]

        project_dir = os.path.join(PROJECTS_DIR, project_id)

        index_path = os.path.join(project_dir, "index.faiss")
        meta_path = os.path.join(project_dir, "meta.pkl")

        if not os.path.exists(index_path):
            raise ValueError(f"Project not found: {project_id}")

        index = faiss.read_index(index_path)

        with open(meta_path, "rb") as f:
            chunks = pickle.load(f)

        self.cache[project_id] = (index, chunks)
        return index, chunks

    def retrieve(self, project_id: str, query: str, k=3):
        index, chunks = self.load_project(project_id)

        q_emb = embedder.encode([query], normalize_embeddings=True)
        _, I = index.search(q_emb, k)

        return [chunks[i] for i in I[0]]
