import os
import tempfile
from pathlib import Path

import typesense
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from gpt4all import GPT4All
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# ------------------ Typesense Config ------------------
TYPESENSE_HOST = "localhost"
TYPESENSE_PORT = "8108"
TYPESENSE_PROTOCOL = "http"
TYPESENSE_API_KEY = "xyz"
COLLECTION_NAME = "documents"

client = typesense.Client({
    "nodes": [{
        "host": TYPESENSE_HOST,
        "port": TYPESENSE_PORT,
        "protocol": TYPESENSE_PROTOCOL
    }],
    "api_key": TYPESENSE_API_KEY,
    "connection_timeout_seconds": 2
})

# ------------------ Embedding Model ------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# ------------------ Local LLM (PURE GPT4ALL) ------------------
llm = GPT4All(
    model_name="orca-mini-3b-gguf2-q4_0.gguf",
    model_path=os.path.expanduser("~/.cache/gpt4all"),
    allow_download=True,
    verbose=True
)


# ------------------ Memory ------------------
conversation_store = {}

def get_memory(user_id: str):
    if user_id not in conversation_store:
        conversation_store[user_id] = []
    return conversation_store[user_id]

# ------------------ Typesense Collection ------------------
def create_collection():
    schema = {
        "name": COLLECTION_NAME,
        "fields": [
            {"name": "document_id", "type": "string"},
            {"name": "chunk_index", "type": "int32"},
            {"name": "content", "type": "string"},
            {"name": "embedding", "type": "float[]", "num_dim": 384}
        ]
    }
    try:
        client.collections.create(schema)
    except Exception:
        pass

# ------------------ PDF Text Extraction ------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# ------------------ Chunking ------------------
def chunk_text(text: str, source: str, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    idx = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append({
            "document_id": source,
            "chunk_index": idx,
            "content": text[start:end]
        })
        start = end - overlap
        idx += 1

    return chunks

# ------------------ Indexing ------------------
def index_document(file_path: str):
    if Path(file_path).suffix.lower() != ".pdf":
        raise Exception("Only PDF supported")

    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text, Path(file_path).name)

    for chunk in chunks:
        chunk["embedding"] = embed_model.encode(chunk["content"]).tolist()

    client.collections[COLLECTION_NAME].documents.import_(chunks)
    return len(chunks)

# ------------------ Hybrid Search ------------------
def hybrid_search(query: str, k: int = 3):
    emb = embed_model.encode(query).tolist()

    vector = client.collections[COLLECTION_NAME].documents.search({
        "q": "*",
        "vector_query": f"embedding:([{' '.join(map(str, emb))}], k:{k})"
    })

    keyword = client.collections[COLLECTION_NAME].documents.search({
        "q": query,
        "query_by": "content",
        "per_page": k
    })

    return [h["document"] for h in vector["hits"] + keyword["hits"]]

# ------------------ RAG Answer ------------------
def generate_answer(user_id: str, question: str) -> str:
    docs = hybrid_search(question)
    context = "\n\n".join(d["content"] for d in docs)

    memory = get_memory(user_id)

    prompt = f"""
You are a helpful assistant.
Answer strictly using the context.

Context:
{context}

Conversation history:
{memory}

Question:
{question}

Answer:
"""

    with llm.chat_session():
        answer = llm.generate(prompt, max_tokens=300)

    memory.append({
        "user": question,
        "assistant": answer
    })

    return answer

# ------------------ API Models ------------------
class AskRequest(BaseModel):
    user_id: str
    question: str

# ------------------ FastAPI ------------------
@app.on_event("startup")
def startup():
    create_collection()

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        return {"error": "PDF only"}

    path = f"{tempfile.gettempdir()}/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())

    return {"chunks": index_document(path)}

@app.post("/ask")
def ask(req: AskRequest):
    return {"answer": generate_answer(req.user_id, req.question)}

@app.get("/history/{user_id}")
def history(user_id: str):
    return {"history": get_memory(user_id)}

# ------------------ Run Server ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "without_lang:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
