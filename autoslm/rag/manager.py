import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTS_DIR = os.path.join(BASE_DIR, "projects")


class RAGManager:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.cache = {}  # project_id -> (index, chunks)

    def _project_dir(self, project_id: str):
        return os.path.join(PROJECTS_DIR, project_id)

    def load(self, project_id: str):
        if project_id in self.cache:
            return

        pdir = self._project_dir(project_id)
        index_path = os.path.join(pdir, "index.faiss")
        meta_path = os.path.join(pdir, "meta.pkl")

        if not os.path.exists(index_path):
            raise RuntimeError(f"No RAG index for project {project_id}")

        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            chunks = pickle.load(f)

        self.cache[project_id] = (index, chunks)

    def query(self, project_id: str, query: str, k: int = 3) -> str:
        self.load(project_id)

        index, chunks = self.cache[project_id]
        emb = self.embedder.encode([query], normalize_embeddings=True)
        _, I = index.search(emb, k)

        return "\n".join(chunks[i] for i in I[0])
