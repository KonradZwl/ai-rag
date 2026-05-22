"""De Pinecone-index bevragen en een antwoord genereren zonder LangChain.

Embed een gebruikersvraag via de Ollama REST API, bevraagt Pinecone voor de
top-k overeenkomende chunks en stuurt de opgehaalde context samen met de
vraag naar een lokaal Ollama LLM om een beknopt antwoord te genereren.

Vereiste omgevingsvariabelen (in te stellen in ``.env``):
    PINECONE_API_KEY -- Pinecone API-sleutel.
    PINECONE_HOST    -- Pinecone index host-URL.
    OLLAMA_API_URL   -- Basis-URL van de Ollama-instantie (bijv. http://localhost:11434).
"""

from pinecone import Pinecone
import requests
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
pinecone_host = os.getenv("PINECONE_HOST")
ollama_api_url = os.getenv("OLLAMA_API_URL")

question = input("Ask your question: ")
embedding_response = requests.post(
    ollama_api_url + "/api/embeddings",
    json={
        "model": "nomic-embed-text:latest",
        "prompt": question
    }
)
question_embedding = embedding_response.json()["embedding"]


pc = Pinecone(api_key=api_key)
index = pc.Index(host=pinecone_host)

results = index.query(
    vector=question_embedding,
    top_k=3,
    include_metadata=True
)

matches = results['matches']
top_context = "\n\n".join(match['metadata']['text'] for match in matches)

# Structureer de prompt met de context en de vraag
llm_prompt = f"""
You are an expert assistant. Use the context below to answer the question as concisely as possible.
If the answer is not contained in the context, say: "I don't know."

Context:
{top_context}

Question: {question}

Answer (concise):
"""

# Roep het LLM aan
llm_response = requests.post(
    ollama_api_url + "/api/generate",
    json={
        "model": "qwen2.5:0.5b",
        "prompt": llm_prompt,
        "stream": False
    }
)
answer = llm_response.json()["response"]
print("Answer:", answer)