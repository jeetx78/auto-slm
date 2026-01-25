import threading
import torch
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel

from autoslm.rag.manager import RAGManager

# =========================
# APP
# =========================
app = FastAPI(title="Auto-SLM (RAG + Streaming)")
rag = RAGManager()
from fastapi.staticfiles import StaticFiles

app.mount("/ui", StaticFiles(directory="static", html=True), name="static")

# =========================
# MODEL CONFIG
# =========================
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "artifacts/adapter"

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token

bnb = BitsAndBytesConfig(load_in_4bit=True)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
)

# REQUIRED for Phi-3 stability
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# =========================
# SCHEMAS
# =========================
class AddDocs(BaseModel):
    project_id: str
    text: str

class Infer(BaseModel):
    project_id: str
    prompt: str
    max_new_tokens: int = 128

# =========================
# ROUTES
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}

# -------- ADD DOCUMENT (PASTE TEXT) --------
@app.post("/add")
def add_docs(req: AddDocs):
    rag.add_documents(req.project_id, [req.text])
    return {"success": True}

# -------- STREAMING INFERENCE --------
@app.post("/infer/stream")
def infer_stream(req: Infer):
    context_chunks = rag.search(req.project_id, req.prompt)
    context = "\n".join(context_chunks)

    prompt = f"""
Answer ONLY using the context.
If the answer is not present, say "Not found".

Context:
{context}

Question:
{req.prompt}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        do_sample=False,
        streamer=streamer,
        use_cache=False,
        eos_token_id=tokenizer.eos_token_id,
    )

    thread = threading.Thread(
        target=model.generate,
        kwargs=generation_kwargs,
    )
    thread.start()

    return StreamingResponse(
        (token for token in streamer),
        media_type="text/plain",
    )
