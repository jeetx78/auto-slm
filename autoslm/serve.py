import os, torch
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from autoslm.supabase_client import supabase
from autoslm.rag.manager import RAGManager

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

app = FastAPI(title="Auto-SLM")

rag = RAGManager()

BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"
ADAPTER_PATH = "artifacts/adapter"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------- MODEL --------
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb = BitsAndBytesConfig(load_in_4bit=True)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True
)
base_model.config.use_cache = False

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# -------- SCHEMAS --------
class InferRequest(BaseModel):
    project_id: str
    prompt: str
    max_new_tokens: int = 64

# -------- HEALTH --------
@app.get("/health")
def health():
    return {"status": "ok"}

# -------- UPLOAD DOCS --------
@app.post("/projects/{project_id}/documents")
async def upload_docs(project_id: str, files: list[UploadFile]):
    texts = []

    for f in files:
        content = (await f.read()).decode("utf-8")
        texts.append(content)

        supabase.table("documents").insert({
            "project_id": project_id,
            "filename": f.filename,
            "content": content
        }).execute()

    rag.add_documents(project_id, texts)
    return {"success": True, "files": len(files)}

# -------- INFER --------
@app.post("/infer")
def infer(req: InferRequest):
    context = "\n".join(
        rag.search(req.project_id, req.prompt)
    )

    prompt = f"""
Answer ONLY using the context.
If not found, say "Not found".

Context:
{context}

Question:
{req.prompt}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=False,
            use_cache=False,
        )

    return {
        "success": True,
        "response": tokenizer.decode(out[0], skip_special_tokens=True)
    }

