def chunk_text(text: str, size=400, overlap=50):
    words = text.split()
    chunks = []

    step = size - overlap
    for i in range(0, len(words), step):
        chunks.append(" ".join(words[i:i+size]))

    return chunks
