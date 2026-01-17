import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# =========================
# CONFIG
# =========================
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "artifacts/adapter"

# =========================
# FastAPI app
# =========================
app = FastAPI(title="Auto-SLM Inference Server")

# =========================
# Load tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token

# =========================
# Load base model (GPU + FP16)
# =========================
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)

# Disable cache for safety
base_model.config.use_cache = False

# =========================
# Load LoRA adapter
# =========================
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

model.eval()

# =========================
# Request schema
# =========================
class InferRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9


# =========================
# Inference endpoint
# =========================
@app.post("/infer")
def infer(req: InferRequest):
    inputs = tokenizer(
        req.prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True,
    )

    return {
        "prompt": req.prompt,
        "response": generated,
    }


# =========================
# Health check
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}
