import os
import faiss
import pickle
import json
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECTS_DIR = os.path.join(BASE_DIR, "projects")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def build_rag(project_id: str, documents: list[str]):
    project_dir = os.path.join(PROJECTS_DIR, project_id)
    os.makedirs(project_dir, exist_ok=True)

    embeddings = embedder.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(project_dir, "index.faiss"))

    with open(os.path.join(project_dir, "meta.pkl"), "wb") as f:
        pickle.dump(documents, f)

    with open(os.path.join(project_dir, "sources.json"), "w") as f:
        json.dump(
            [{"chunk_id": i} for i in range(len(documents))],
            f,
            indent=2,
        )

    print(f"✅ RAG built for project: {project_id}")
