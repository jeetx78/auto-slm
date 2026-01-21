import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from autoslm.rag.manager import RAGManager

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "artifacts", "adapter"))
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"

if not os.path.exists(ADAPTER_PATH):
    raise RuntimeError("❌ LoRA adapter missing")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# =========================
# FASTAPI
# =========================
app = FastAPI(title="Auto-SLM (Multi-Project RAG + LoRA)")

class InferRequest(BaseModel):
    project_id: str
    prompt: str
    max_new_tokens: int = 64

# =========================
# RAG MANAGER
# =========================
rag = RAGManager()

# =========================
# MODEL
# =========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
)

base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/infer")
def infer(req: InferRequest):
    try:
        context = rag.query(req.project_id, req.prompt)

        prompt = f"""
Answer using ONLY the context.
If missing, say: Not found in documents.

Context:
{context}

Question:
{req.prompt}

Answer:
"""

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                use_cache=False,
            )

        text = tokenizer.decode(output[0], skip_special_tokens=True)

        return {"success": True, "response": text.strip()}

    except Exception as e:
        return {"success": False, "error": str(e)}
