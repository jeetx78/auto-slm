from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import time

app = FastAPI()

# ---------- SCHEMAS ----------
class AddDocs(BaseModel):
    project_id: str
    text: str

class Infer(BaseModel):
    project_id: str
    prompt: str
    max_new_tokens: int = 128

# ---------- ROUTES ----------
@app.get("/")
def root():
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/add")
def add_docs(req: AddDocs):
    return {
        "success": True,
        "project_id": req.project_id,
        "length": len(req.text),
    }

@app.post("/infer/stream")
def infer_stream(req: Infer):
    def gen():
        for word in req.prompt.split():
            yield word + " "
            time.sleep(0.1)

    return StreamingResponse(gen(), media_type="text/plain")
