import os
import json
from dotenv import load_dotenv

from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from pinecone import Pinecone as PineconeClient, ServerlessSpec

load_dotenv()
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

index_name = "faq-rag-test"
pc = PineconeClient(api_key=PINECONE_API_KEY)

if index_name not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

with open("../data/faq.json", "r", encoding="utf-8") as f:
    faq_data = json.load(f)

documents = []
for item in faq_data["faq"]:
    doc = Document(
        page_content=item["answer"],
        metadata={"question": item["question"]}
    )
    documents.append(doc)

embeddings = OllamaEmbeddings(model="all-minilm:l6-v2", base_url=OLLAMA_API_URL)

vectorstore = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    index_name=index_name
)