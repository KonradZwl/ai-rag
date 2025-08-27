import os
from dotenv import load_dotenv

from langchain_pinecone import Pinecone
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

index_name = "faq-rag-test"

embeddings = OllamaEmbeddings(model="all-minilm:l6-v2", base_url=OLLAMA_API_URL)
vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={
    "k": 5,
    })

llm = OllamaLLM(
    model="qwen2.5:7b-instruct",
    base_url=OLLAMA_API_URL,
    max_tokens=512,
    temperature=0.0,
    )

prompt_template = """
You are a precise assistant that answers questions using ONLY the context below.

Rules:
1. Answer ONLY the specific question that was asked. Do not include unrelated information.
2. Write the answer in clear, natural, complete sentences. Avoid robotic phrasing such as "Based on the information provided in the context".
3. If the context does not contain the answer, say: "I can't help you with that."

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={
        "prompt": PROMPT
        }
)

while True:
    question = input("\nAsk your question (or 'exit'): ")
    if question.lower() in ["exit", "quit"]:
        break

    # Call the QA chain as usual
    response = qa_chain.invoke({"query": question})
    print(f"\n💡 Answer: {response['result']}")
