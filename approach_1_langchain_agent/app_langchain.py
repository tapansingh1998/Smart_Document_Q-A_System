import os
import uuid
import tempfile
from pathlib import Path

import typesense
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import GPT4All

from unstructured.partition.pdf import partition_pdf
from sentence_transformers import SentenceTransformer

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

# ------------------ GPT4All Local LLM ------------------
llm = GPT4All(
    model="ggml-gpt4all-j-v1.3-groovy",
    temp=0.2
)

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
    elements = partition_pdf(
        filename=pdf_path,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000
    )
    return "".join([el.text for el in elements])

# ------------------ Text Chunking ------------------
def chunk_text(text: str, source: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    return [{"document_id": source, "chunk_index": i, "content": c} for i, c in enumerate(chunks)]

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

# ------------------ Memory ------------------
memory_store = {}

def get_memory(user_id):
    if user_id not in memory_store:
        memory_store[user_id] = ConversationBufferMemory(memory_key="history")
    return memory_store[user_id]

# ------------------ RAG Tool ------------------
def rag_tool(question: str) -> str:
    docs = hybrid_search(question)
    return "\n\n".join(d["content"] for d in docs)

RAG_TOOL = Tool(
    name="RAG",
    func=rag_tool,
    description="Retrieve document context"
)

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
    agent = initialize_agent(
        tools=[RAG_TOOL],
        llm=llm,
        agent="zero-shot-react-description",
        memory=get_memory(req.user_id),
        verbose=False
    )
    return {"answer": agent.run(req.question)}

@app.get("/history/{user_id}")
def history(user_id: str):
    return {"history": get_memory(user_id).buffer}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
