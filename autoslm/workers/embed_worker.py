from autoslm.supabase_client import supabase
from autoslm.rag.manager import RAGManager
import time

rag = RAGManager()

while True:
    res = supabase.table("documents") \
        .select("id, project_id, content") \
        .eq("embedded", False) \
        .limit(10) \
        .execute()

    if not res.data:
        time.sleep(5)
        continue

    for doc in res.data:
        rag.add_documents(doc["project_id"], [doc["content"]])

        supabase.table("documents") \
            .update({"embedded": True}) \
            .eq("id", doc["id"]) \
            .execute()
