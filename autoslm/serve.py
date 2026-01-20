import os
import torch
import faiss
import pickle

from fastapi import FastAPI
from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import PeftModel
from sentence_transformers import SentenceTransformer

# =====================================================
# PATHS (ABSOLUTE, SAFE)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAG_DIR = os.path.join(BASE_DIR, "rag", "rag_docs")
INDEX_PATH = os.path.join(RAG_DIR, "index.faiss")
META_PATH = os.path.join(RAG_DIR, "meta.pkl")

ADAPTER_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "artifacts", "adapter")
)

# =====================================================
# FAIL FAST
# =====================================================
for path, name in [
    (INDEX_PATH, "FAISS index"),
    (META_PATH, "RAG metadata"),
    (ADAPTER_PATH, "LoRA adapter"),
]:
    if not os.path.exists(path):
        raise RuntimeError(f"❌ Missing {name}: {path}")

print("✅ All required files found")

# =====================================================
# CONFIG
# =====================================================
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"🚀 Device: {DEVICE}")
print(f"🧠 Dtype: {DTYPE}")

# =====================================================
# FASTAPI
# =====================================================
app = FastAPI(title="Auto-SLM (LoRA + RAG)")

class InferRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64

# =====================================================
# LOAD RAG (FAST)
# =====================================================
print("📥 Loading RAG...")
embedder = SentenceTransformer(
    "all-MiniLM-L6-v2",
    device=DEVICE,
)

index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    chunks: list[str] = pickle.load(f)

print(f"✅ RAG ready ({len(chunks)} chunks)")

# =====================================================
# TOKENIZER
# =====================================================
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token

# =====================================================
# LOAD MODEL (4-BIT, SAFE FOR PHI-3)
# =====================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

print("📦 Loading base model (4-bit)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# 🔴 REQUIRED FOR PHI-3 STABILITY
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

print("🧩 Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print("✅ Model loaded")

# =====================================================
# ROUTES
# =====================================================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/infer")
def infer(req: InferRequest):
    try:
        # ---------------------------
        # RETRIEVE (FAST)
        # ---------------------------
        q_emb = embedder.encode(
            req.prompt,
            normalize_embeddings=True,
        )

        _, idxs = index.search(q_emb.reshape(1, -1), k=3)
        context = "\n".join(chunks[i] for i in idxs[0])

        # ---------------------------
        # PROMPT
        # ---------------------------
        prompt = (
            "Answer using ONLY the context.\n"
            "If the answer is missing, say: Not found in documents.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{req.prompt}\n\n"
            "Answer:"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # ---------------------------
        # GENERATION (DETERMINISTIC)
        # ---------------------------
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False,  # 🚨 IMPORTANT
            )

        text = tokenizer.decode(
            output[0],
            skip_special_tokens=True,
        )

        return {
            "success": True,
            "response": text.strip(),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "type": type(e).__name__,
        }
