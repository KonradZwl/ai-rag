"""Ingest a PDF into Pinecone without LangChain.

Extracts text from ``data/info.pdf``, splits it into fixed-size word chunks,
generates embeddings via the Ollama REST API, and upserts the vectors into a
Pinecone index.

Required environment variables (set in ``.env``):
    PINECONE_API_KEY – Pinecone API key.
    OLLAMA_API_URL   – Base URL of the Ollama instance (e.g. http://localhost:11434).
"""

import PyPDF2
import requests
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
load_dotenv()

ollama_api_url = os.getenv("OLLAMA_API_URL")
api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=api_key)

# Load in a PDF and split it into chunks of 75 words
pdf_path = "../data/info.pdf"
text = ""
with open(pdf_path, "rb") as f:
    reader = PyPDF2.PdfReader(f)
    for page in reader.pages:
        text += page.extract_text() + "\n"

def chunk_text(text, chunk_size=75):
    """Split *text* into chunks of *chunk_size* words."""
    words = text.split()
    chunks = [] 
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

chunks = chunk_text(text)

# Create embeddings for each chunk
embeddings = []
for chunk in chunks:
    response = requests.post(
        ollama_api_url + "/api/embeddings",
        json={"model": "nomic-embed-text:latest", "prompt": chunk}
    )
    embeddings.append(response.json()["embedding"])

# Create index and upsert vectors
index_name = "pdf-rag-test"
indexes = pc.list_indexes()
existing_names = [idx["name"] for idx in indexes]

if index_name not in existing_names:
    dim = 768
    pc.create_index(name=index_name, dimension=dim, metric="cosine")
else:
    print(f"Index '{index_name}' already exists, skipping creation.")

index = pc.Index(index_name)

items_to_upsert = [
    (f"{i}", embeddings[i], {"text": chunks[i]})
    for i in range(len(embeddings))
]

index.upsert(
    vectors=items_to_upsert
)
print("Vectors upserted!")