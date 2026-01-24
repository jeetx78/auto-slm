import os
import torch
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

from autoslm.supabase_client import supabase
from autoslm.rag.manager import RAGManager

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

# ======================================================
# APP
# ======================================================
app = FastAPI(title="Auto-SLM (Multi-Project RAG)")

rag = RAGManager()

# ======================================================
# MODEL CONFIG
# ======================================================
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = os.path.abspath("artifacts/adapter")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================================================
# LOAD TOKENIZER
# ======================================================
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token

# ======================================================
# LOAD MODEL (4-BIT SAFE)
# ======================================================
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

# 🔴 REQUIRED for Phi-3 stability
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# ======================================================
# SCHEMAS
# ======================================================
class InferRequest(BaseModel):
    project_id: str
    prompt: str
    max_new_tokens: int = 64

class PasteTextRequest(BaseModel):
    project_id: str
    text: str

# ======================================================
# HEALTH
# ======================================================
@app.get("/health")
def health():
    return {"status": "ok"}

# ======================================================
# ADD TEXT (CHATGPT STYLE PASTE)
# ======================================================
@app.post("/projects/{project_id}/text")
def add_text(project_id: str, req: PasteTextRequest):
    # store raw text
    supabase.table("documents").insert({
        "project_id": project_id,
        "filename": "pasted_text",
        "content": req.text
    }).execute()

    # update RAG
    rag.add_documents(project_id, [req.text])

    return {"success": True}

# ======================================================
# UPLOAD FILES
# ======================================================
@app.post("/projects/{project_id}/documents")
async def upload_docs(project_id: str, files: list[UploadFile]):
    texts = []

    for f in files:
        raw = (await f.read()).decode("utf-8", errors="ignore")

        supabase.table("documents").insert({
            "project_id": project_id,
            "filename": f.filename,
            "content": raw,
        }).execute()

        texts.append(raw)

    rag.add_documents(project_id, texts)

    return {"success": True, "files": len(files)}

# ======================================================
# INFER
# ======================================================
@app.post("/infer")
def infer(req: InferRequest):
    try:
        chunks = rag.search(req.project_id, req.prompt, k=3)

        if not chunks:
            return {
                "success": True,
                "response": "Not found in documents."
            }

        context = "\n".join(chunks)

        prompt = f"""
Answer ONLY using the context below.
If not found, say: Not found in documents.

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
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                use_cache=False,
            )

        return {
            "success": True,
            "response": tokenizer.decode(
                output[0],
                skip_special_tokens=True
            )
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "type": type(e).__name__,
        }
