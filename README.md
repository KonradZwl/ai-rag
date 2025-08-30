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

### Ingest FAQ Data

- Edit and run `lang/ingest.py` to ingest FAQ data from `data/faq.json` into Pinecone.

### Query the System

- Run `lang/main.py` to start the interactive question-answering loop:
  ```bash
  python main.py
  ```
- Type your question and get an answer based on the retrieved context.

## File Structure

```
.
├── data/
│   └── faq.json
├── lang/
│   ├── ingest.py
│   └── main.py
├── requirements.txt
├── .env
└── README.md
```

## Notes

- Make sure your Pinecone index dimension matches your embedding model output.
- For best results, store both questions and answers together in each document.
- You can adjust chunk size and retrieval parameters in the code.
