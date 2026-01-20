import torch
import faiss
import pickle
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import os

print("🔥 SERVE.PY LOADED FROM:", __file__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.join(BASE_DIR, "rag", "rag_docs")
INDEX_PATH = os.path.join(RAG_DIR, "index.faiss")
META_PATH = os.path.join(RAG_DIR, "meta.pkl")


# --- Hard validation (fail fast, readable) ---
if not os.path.exists(INDEX_PATH):
    raise RuntimeError(f"FAISS index not found at: {INDEX_PATH}")

if not os.path.exists(META_PATH):
    raise RuntimeError(f"RAG metadata not found at: {META_PATH}")
# =====================
# CONFIG
# =====================
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "artifacts/adapter"

# =====================
# APP
# =====================
app = FastAPI(title="Auto-SLM + RAG")

class InferRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128

# =====================
# DEVICE
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# =====================
# LOAD RAG
# =====================
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    chunks = pickle.load(f)

# =====================
# TOKENIZER + MODEL
# =====================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=DTYPE,
    device_map=DEVICE,
    trust_remote_code=True,
)

base_model.config.use_cache = False

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# =====================
# HEALTH
# =====================
@app.get("/health")
def health():
    return {"status": "ok"}

# =====================
# INFER WITH RAG
# =====================
@app.post("/infer")
def infer(req: InferRequest):
    try:
        # ---- RETRIEVE ----
        q_emb = embedder.encode([req.prompt])
        _, I = index.search(q_emb, k=3)
        context = "\n".join([chunks[i] for i in I[0]])

        # ---- PROMPT ----
        prompt = f"""
Use the context below to answer the question.
Only use the provided context.

Context:
{context}

Question:
{req.prompt}

Answer:
"""

        inputs = tokenizer(prompt, return_tensors="pt")
        device = list(model.parameters())[0].device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                temperature=0.2,
                do_sample=False,
                use_cache=False,
            )

        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"success": True, "response": text}

    except Exception as e:
        return {"success": False, "error": str(e)}
