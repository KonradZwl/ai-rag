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

# Structure the prompt using the context and question
llm_prompt = f"""
You are an expert assistant. Use the context below to answer the question as concisely as possible.
If the answer is not contained in the context, say: "I don't know."

Context:
{top_context}

Question: {question}

Answer (concise):
"""

# Call the LLM
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