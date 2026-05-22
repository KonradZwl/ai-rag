# AI-RAG: Retrieval-Augmented Generation with Pinecone and Ollama

This project demonstrates a Retrieval-Augmented Generation (RAG) pipeline using Python, LangChain, Pinecone, and Ollama.
It ingests FAQ data, stores question-answer pairs in a vector database, and answers user queries using a local LLM.

## Features

- Ingests FAQ data from JSON or PDF
- Splits and embeds documents using Ollama
- Stores embeddings in Pinecone vector database
- Retrieves relevant context for user questions
- Uses a local LLM (Ollama) to generate answers based on retrieved context
- Environment configuration via .env file

## Setup

**Configure environment variables**
   - Edit .env in the project root:
     ```
     PINECONE_API_KEY=your_pinecone_api_key
     PINECONE_HOST=your_pinecone_host_url
     PINECONE_LANG=your_pinecone_host_url_lang_version
     OLLAMA_API_URL=http://localhost:11434
     ```
**Start Ollama**
   - Install and run Ollama locally: [Ollama documentation](https://ollama.com/)
   - Pull required models (e.g., `ollama pull all-minilm:l6-v2`)

## Usage

The project includes two independent implementations of the RAG pipeline:

### `lang/` — LangChain-based pipeline

Uses LangChain abstractions for document handling, embedding, retrieval, and LLM orchestration.

1. **Ingest FAQ data** — run `lang/ingest.py` to load `data/faq.json` into Pinecone.
   ```bash
   cd lang
   python ingest.py
   ```
2. **Query** — run `lang/main.py` for an interactive Q&A loop.
   ```bash
   python main.py
   ```

| Component | Detail |
|---|---|
| Embedding model | `all-minilm:l6-v2` (384 dims) |
| LLM | `qwen2.5:7b-instruct` |
| Index name | `faq-rag-test` |

### `manual/` — Lightweight pipeline (no LangChain)

Calls the Ollama and Pinecone REST APIs directly — useful for understanding what LangChain abstracts away.

1. **Ingest a PDF** — run `manual/ingest.py` to extract text from `data/info.pdf`, chunk it, and upsert into Pinecone.
   ```bash
   cd manual
   python ingest.py
   ```
2. **Query** — run `manual/query.py` to ask a single question and get a concise answer.
   ```bash
   python query.py
   ```

| Component | Detail |
|---|---|
| Embedding model | `nomic-embed-text:latest` (768 dims) |
| LLM | `qwen2.5:0.5b` |
| Index name | `pdf-rag-test` |

## File Structure

```
.
├── data/
│   ├── faq.json          # FAQ dataset (used by lang/)
│   └── info.pdf          # PDF document (used by manual/)
├── lang/
│   ├── ingest.py         # LangChain-based FAQ ingestion
│   └── main.py           # LangChain-based interactive Q&A
├── manual/
│   ├── ingest.py         # Direct-API PDF ingestion
│   └── query.py          # Direct-API single-question query
├── requirements.txt
├── .env                  # Environment variables (not committed)
└── README.md
```

## Notes

- Make sure your Pinecone index dimension matches your embedding model output (`lang/` uses 384, `manual/` uses 768).
- For best results, store both questions and answers together in each document.
- You can adjust chunk size and retrieval parameters in the code.
