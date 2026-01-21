import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

from autoslm.rag.manager import RAGManager

# =====================================================
# BASIC SETUP
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "artifacts", "adapter")
)

if not os.path.exists(ADAPTER_PATH):
    raise RuntimeError(f"❌ LoRA adapter not found: {ADAPTER_PATH}")

print("✅ Adapter found")

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"🚀 Device: {DEVICE}")
print(f"🧠 Dtype: {DTYPE}")

# =====================================================
# FASTAPI
# =====================================================
app = FastAPI(title="Auto-SLM (Multi-Project RAG + LoRA)")

class InferRequest(BaseModel):
    project_id: str
    prompt: str
    max_new_tokens: int = 64

# =====================================================
# LOAD RAG MANAGER (MULTI-PROJECT)
# =====================================================
print("📥 Initializing RAG Manager...")
rag = RAGManager()
print("✅ RAG Manager ready")

# =====================================================
# TOKENIZER
# =====================================================
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token

# =====================================================
# LOAD MODEL (4-BIT + LORA)
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

print("✅ Model fully loaded")

# =====================================================
# ROUTES
# =====================================================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/infer")
def infer(req: InferRequest):
    try:
        # -------------------------------------------------
        # 1️⃣ RETRIEVE CONTEXT (PROJECT-SCOPED)
        # -------------------------------------------------
        contexts = rag.retrieve(
            project_id=req.project_id,
            query=req.prompt,
            k=3,
        )

        context_text = "\n".join(contexts)

        # -------------------------------------------------
        # 2️⃣ BUILD PROMPT
        # -------------------------------------------------
        prompt = f"""You are Auto-SLM.

Answer ONLY using the context below.
If the answer is not present, say:
"Not found in documents."

Context:
{context_text}

Question:
{req.prompt}

Answer:
"""

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # -------------------------------------------------
        # 3️⃣ GENERATE (DETERMINISTIC, SAFE)
        # -------------------------------------------------
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False,  # 🚨 CRITICAL FOR PHI-3
            )

        answer = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        return {
            "success": True,
            "project_id": req.project_id,
            "answer": answer.strip(),
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "type": type(e).__name__,
        }
