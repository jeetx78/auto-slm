import json
from collections import Counter

def infer_task_from_dataset(dataset_path: str):
    """
    Infer task type from dataset.jsonl
    """

    texts = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])

    if not texts:
        return "unknown"

    # Heuristics
    avg_len = sum(len(t.split()) for t in texts) / len(texts)
    question_like = sum("?" in t for t in texts)
    colon_like = sum(":" in t for t in texts)

    # Decision rules (simple, effective)
    if question_like / len(texts) > 0.3:
        return "qa"

    if colon_like / len(texts) > 0.4:
        return "extraction"

    if avg_len > 300:
        return "summarization"

    return "assistant"
