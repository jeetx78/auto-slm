import torch
import threading
from fastapi import (
    FastAPI,
    Depends,
    Header,
    HTTPException,
    UploadFile,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from autoslm.supabase_client import supabase
from autoslm.rag.manager import RAGManager

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from peft import PeftModel

# =====================================================
# APP
# =====================================================
app = FastAPI(title="Auto-SLM (Streaming + RAG + LoRA)")
rag = RAGManager()  # ❌ removed load_from_db()

# =====================================================
# MODEL
# =====================================================
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

# REQUIRED FOR PHI-3
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# =====================================================
# AUTH
# =====================================================
def get_user(authorization: str = Header(...)):
    token = authorization.replace("Bearer ", "")
    user = supabase.auth.get_user(token)
    if not user:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user.user

def verify_project(project_id: str, user):
    res = supabase.table("projects").select("id").eq(
        "id", project_id
    ).eq(
        "owner_id", user.id
    ).execute()

    if not res.data:
        raise HTTPException(status_code=403, detail="Forbidden")

# =====================================================
# SCHEMAS
# =====================================================
class AddDocs(BaseModel):
    project_id: str
    text: str

class Infer(BaseModel):
    project_id: str
    prompt: str
    max_new_tokens: int = 128

# =====================================================
# ROUTES
# =====================================================
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- ADD DOCUMENT (PASTE TEXT) ----------
@app.post("/add")
def add_docs(req: AddDocs, user=Depends(get_user)):
    verify_project(req.project_id, user)

    supabase.table("documents").insert({
        "project_id": req.project_id,
        "owner_id": user.id,
        "content": req.text,
    }).execute()

    rag.add_documents(req.project_id, [req.text])
    return {"success": True}

# ---------- UPLOAD FILES ----------
@app.post("/projects/{project_id}/documents")
async def upload_docs(
    project_id: str,
    files: list[UploadFile],
    user=Depends(get_user),
):
    verify_project(project_id, user)

    texts = []
    for f in files:
        content = (await f.read()).decode("utf-8")
        texts.append(content)

        supabase.table("documents").insert({
            "project_id": project_id,
            "owner_id": user.id,
            "content": content,
        }).execute()

    rag.add_documents(project_id, texts)
    return {"success": True, "files": len(files)}

# ---------- STREAMING INFERENCE ----------
@app.post("/infer/stream")
def infer_stream(req: Infer, user=Depends(get_user)):
    verify_project(req.project_id, user)

    retrieved = rag.search(req.project_id, req.prompt)

    if retrieved:
        context = "\n".join(retrieved)
        prompt = f"""
Answer ONLY using the context.
If missing, say "Not found".

Context:
{context}

Question:
{req.prompt}

Answer:
"""
    else:
        prompt = req.prompt

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

    thread = threading.Thread(
        target=model.generate,
        kwargs=dict(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            do_sample=False,
            streamer=streamer,
            use_cache=False,
            eos_token_id=tokenizer.eos_token_id,
        ),
    )
    thread.start()

    return StreamingResponse(
        (token for token in streamer),
        media_type="text/plain",
    )
