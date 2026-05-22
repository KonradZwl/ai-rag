"""Een PDF opnemen in Pinecone zonder LangChain.

Extraheert tekst uit ``data/info.pdf``, splitst deze in chunks van een vast
aantal woorden, genereert embeddings via de Ollama REST API en voegt de
vectoren toe aan een Pinecone-index.

Vereiste omgevingsvariabelen (in te stellen in ``.env``):
    PINECONE_API_KEY -- Pinecone API-sleutel.
    OLLAMA_API_URL   -- Basis-URL van de Ollama-instantie (bijv. http://localhost:11434).
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

# Laad een PDF in en splits deze in chunks van 75 woorden
pdf_path = "../data/info.pdf"
text = ""
with open(pdf_path, "rb") as f:
    reader = PyPDF2.PdfReader(f)
    for page in reader.pages:
        text += page.extract_text() + "\n"

def chunk_text(text, chunk_size=75):
    """Splits *text* op in chunks van *chunk_size* woorden."""
    words = text.split()
    chunks = [] 
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

chunks = chunk_text(text)

# Maak embeddings aan voor elke chunk
embeddings = []
for chunk in chunks:
    response = requests.post(
        ollama_api_url + "/api/embeddings",
        json={"model": "nomic-embed-text:latest", "prompt": chunk}
    )
    embeddings.append(response.json()["embedding"])

# Index aanmaken en vectoren upserten
index_name = "pdf-rag-test"
indexes = pc.list_indexes()
existing_names = [idx["name"] for idx in indexes]

if index_name not in existing_names:
    dim = 768
    pc.create_index(name=index_name, dimension=dim, metric="cosine")
else:
    print(f"Index '{index_name}' bestaat al, aanmaken overgeslagen.")

index = pc.Index(index_name)

items_to_upsert = [
    (f"{i}", embeddings[i], {"text": chunks[i]})
    for i in range(len(embeddings))
]

index.upsert(
    vectors=items_to_upsert
)
print("Vectoren toegevoegd!")