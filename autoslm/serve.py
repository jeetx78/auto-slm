import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# =====================
# CONFIG
# =====================
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "artifacts/adapter"

# =====================
# FASTAPI APP
# =====================
app = FastAPI(title="Auto-SLM")

# =====================
# REQUEST SCHEMA
# =====================
class InferRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128

# =====================
# DEVICE + DTYPE
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print("=================================")
print("🚀 Auto-SLM starting")
print("Device:", DEVICE)
print("Dtype:", DTYPE)
print("=================================")

# =====================
# TOKENIZER
# =====================
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# =====================
# BASE MODEL
# =====================
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=DTYPE,
    device_map=DEVICE,
    trust_remote_code=True
)

# 🔴 CRITICAL FIX FOR PHI-3
base_model.config.use_cache = False

# =====================
# LOAD LoRA ADAPTER
# =====================
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH
)

model.eval()

# =====================
# HEALTH CHECK
# =====================
@app.get("/health")
def health():
    return {"status": "ok"}

# =====================
# INFERENCE (BULLETPROOF)
# =====================
@app.post("/infer")
def infer(req: InferRequest):
    try:
        # Tokenize
        inputs = tokenizer(
            req.prompt,
            return_tensors="pt"
        )

        # Safe device move
        device = list(model.parameters())[0].device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate (CACHE OFF)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                temperature=0.7,
                do_sample=True,
                use_cache=False,   # 🔴 ABSOLUTELY REQUIRED
            )

        text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return {
            "success": True,
            "response": text
        }

    except Exception as e:
        # ALWAYS JSON
        return {
            "success": False,
            "error": str(e),
            "type": type(e).__name__
        }
