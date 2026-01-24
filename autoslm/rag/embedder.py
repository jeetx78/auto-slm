from sentence_transformers import SentenceTransformer
import torch

class Embedder:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2",
            device=device,
        )

    def embed(self, texts: list[str]):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
        )
