# AI-RAG Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Technology Stack](#technology-stack)
4. [Getting Started](#getting-started)
5. [Project Structure](#project-structure)
6. [Usage Guide](#usage-guide)
7. [Implementation Details](#implementation-details)
8. [Configuration](#configuration)
9. [Dependencies](#dependencies)
10. [Development](#development)
11. [Troubleshooting](#troubleshooting)

---

## Project Overview

**AI-RAG** is a Retrieval-Augmented Generation (RAG) system that combines vector search with large language models to provide accurate, context-aware answers to user questions. The project demonstrates two implementation approaches:

1. **LangChain-based implementation** (`lang/` directory) - High-level abstractions using LangChain framework
2. **Manual implementation** (`manual/` directory) - Direct API calls for fine-grained control

The system ingests documents (JSON FAQs or PDF files), converts them into embeddings, stores them in Pinecone vector database, and retrieves relevant context to generate answers using Ollama-powered LLMs.

### What Problem Does This Solve?

Traditional chatbots can only answer questions based on their training data. This RAG system allows you to:
- Provide accurate answers based on your specific document corpus
- Update knowledge without retraining the model
- Ground AI responses in factual, retrievable information
- Reduce hallucinations by constraining answers to available context

---

## Features

### Core Capabilities
- **Multiple Document Formats**: Supports JSON-based FAQs and PDF documents
- **Vector Embeddings**: Converts text into semantic embeddings using Ollama models
- **Semantic Search**: Finds relevant information using Pinecone vector database
- **Context-Aware Answers**: Generates responses based on retrieved context using local LLMs
- **Two Implementation Approaches**: Choose between LangChain framework or manual API control
- **Interactive Query Interface**: Command-line interface for asking questions
- **Customizable Prompts**: Fine-tune how the LLM generates answers
- **Source Document Tracking**: Shows which documents were used to generate answers

### Technical Features
- Serverless vector storage with Pinecone
- Local LLM inference with Ollama (no cloud dependencies for inference)
- Environment-based configuration
- Configurable retrieval parameters (top-k, chunk size)
- Multiple embedding models support

---

## Technology Stack

### Core Technologies
- **Python 3.x**: Primary programming language
- **LangChain**: Framework for building LLM applications
  - `langchain-pinecone`: Pinecone integration
  - `langchain-ollama`: Ollama integration
  - `langchain-text-splitters`: Document chunking
- **Pinecone**: Cloud-based vector database for embeddings storage
- **Ollama**: Local LLM runtime for embeddings and inference

### Key Libraries
- **Document Processing**: PyPDF2, pypdf - PDF text extraction
- **Vector Operations**: faiss-cpu - Local vector operations
- **HTTP Client**: requests, httpx, aiohttp - API communication
- **Configuration**: python-dotenv - Environment variable management
- **Data Validation**: pydantic - Schema validation

### AI Models (via Ollama)
- **Embedding Model**: `all-minilm:l6-v2` (384 dimensions) or `nomic-embed-text:latest` (768 dimensions)
- **LLM Models**: `qwen2.5:7b-instruct`, `qwen2.5:0.5b` - Text generation

---

## Getting Started

### Prerequisites

1. **Python 3.8+**: Ensure Python is installed
   ````bash
   python --version
   ````

2. **Ollama**: Install and run locally
   - Download from [ollama.com](https://ollama.com/)
   - Install required models:
     ````bash
     ollama pull all-minilm:l6-v2
     ollama pull qwen2.5:7b-instruct
     # For manual implementation:
     ollama pull nomic-embed-text:latest
     ollama pull qwen2.5:0.5b
     ````

3. **Pinecone Account**: Sign up at [pinecone.io](https://www.pinecone.io/)
   - Create a new project
   - Note your API key and environment

### Installation

1. **Clone the repository**
   ````bash
   git clone https://github.com/KonradZwl/ai-rag.git
   cd ai-rag
   ````

2. **Install dependencies**
   ````bash
   pip install -r requirements.txt
   ````

3. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ````env
   # Pinecone Configuration
   PINECONE_API_KEY=your_pinecone_api_key_here
   PINECONE_HOST=your_pinecone_host_url_here
   
   # Ollama Configuration
   OLLAMA_API_URL=http://localhost:11434
   ````

   **How to get your Pinecone credentials:**
   - Log in to Pinecone console
   - Navigate to API Keys section
   - Copy your API key
   - The host URL is available after creating an index

4. **Verify Ollama is running**
   ````bash
   curl http://localhost:11434/api/tags
   ````

---

## Project Structure

````
ai-rag/
│
├── lang/                          # LangChain-based implementation
│   ├── ingest.py                  # Ingest FAQ data using LangChain
│   └── main.py                    # Query system using LangChain
│
├── manual/                        # Manual API-based implementation
│   ├── ingest.py                  # Ingest PDF data with direct API calls
│   └── query.py                   # Query system with direct API calls
│
├── data/                          # Data directory (gitignored)
│   ├── faq.json                   # FAQ data in JSON format (not in repo)
│   └── info.pdf                   # PDF document for ingestion (not in repo)
│
├── requirements.txt               # Python dependencies
├── .env                           # Environment configuration (gitignored)
├── .gitignore                     # Git ignore rules
├── README.md                      # Quick start guide
└── documentation.md               # This file

````

### Directory Purposes

- **`lang/`**: High-level implementation using LangChain abstractions. Recommended for most users.
- **`manual/`**: Low-level implementation with direct Pinecone and Ollama API calls. Useful for understanding internals or custom optimization.
- **`data/`**: Should contain your source documents (excluded from git for privacy/size)

### Key Files

| File | Purpose |
|------|---------|
| `lang/ingest.py` | Loads FAQ JSON, creates embeddings, stores in Pinecone using LangChain |
| `lang/main.py` | Interactive Q&A interface with LangChain RetrievalQA chain |
| `manual/ingest.py` | PDF processing with manual chunking and Pinecone upsert |
| `manual/query.py` | Direct vector search and LLM generation without LangChain |
| `requirements.txt` | All Python package dependencies |

---

## Usage Guide

### Approach 1: LangChain Implementation (Recommended)

#### Step 1: Prepare FAQ Data

Create `data/faq.json` with your FAQ content:
````json
{
  "faq": [
    {
      "question": "What is RAG?",
      "answer": "Retrieval-Augmented Generation is a technique that combines retrieval of relevant documents with LLM generation to produce accurate, grounded answers."
    },
    {
      "question": "How does vector search work?",
      "answer": "Vector search converts text into numerical vectors and finds similar vectors using distance metrics like cosine similarity."
    }
  ]
}
````

#### Step 2: Ingest FAQ Data

````bash
cd lang
python ingest.py
````

**What happens:**
1. Loads FAQ data from `../data/faq.json`
2. Creates documents combining questions and answers
3. Generates embeddings using Ollama's `all-minilm:l6-v2` model
4. Creates/uses Pinecone index `faq-rag-test` (384 dimensions, cosine metric)
5. Stores vectors in Pinecone

#### Step 3: Query the System

````bash
python main.py
````

**Example interaction:**
````
Ask your question (or 'exit'): What is RAG?

Retrieved context:
Q: What is RAG?
A: Retrieval-Augmented Generation is a technique that combines retrieval of relevant documents with LLM generation to produce accurate, grounded answers.

Answer: Retrieval-Augmented Generation is a technique that combines retrieval of relevant documents with LLM generation to produce accurate, grounded answers.
````

Type `exit` to quit.

### Approach 2: Manual Implementation

#### Step 1: Prepare PDF Document

Place your PDF file at `data/info.pdf`

#### Step 2: Ingest PDF

````bash
cd manual
python ingest.py
````

**What happens:**
1. Extracts text from PDF using PyPDF2
2. Chunks text into 75-word segments
3. Generates embeddings using `nomic-embed-text:latest` model (768 dimensions)
4. Creates Pinecone index `pdf-rag-test`
5. Upserts vectors with metadata

#### Step 3: Query the System

````bash
python query.py
````

**Example:**
````
Ask your question: How do I install Python packages?
Answer: Use pip install followed by the package name...
````

---

## Implementation Details

### LangChain Approach (`lang/`)

#### Embedding Process
````python
embeddings = OllamaEmbeddings(
    model="all-minilm:l6-v2", 
    base_url=OLLAMA_API_URL
)
````
- Uses 384-dimensional embeddings
- Model runs locally via Ollama

#### Document Structure
````python
Document(
    page_content=f"Q: {question}\nA: {answer}",
    metadata={"question": question}
)
````
- Combines Q&A pairs for better context retrieval
- Metadata enables filtering

#### Retrieval Configuration
````python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
````
- Retrieves top 5 most relevant documents
- Uses cosine similarity

#### Prompt Engineering
The system uses a carefully crafted prompt template:
````python
"""
You are a precise assistant that answers questions using ONLY the context below.

Rules:
1. Answer ONLY the specific question that was asked
2. Write in clear, natural, complete sentences
3. If the context doesn't contain the answer, say: "I can't help you with that."

Context: {context}
Question: {question}
Answer:
"""
````

#### RetrievalQA Chain
````python
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",  # Stuffs all context into prompt
    return_source_documents=True
)
````

### Manual Approach (`manual/`)

#### PDF Chunking
````python
def chunk_text(text, chunk_size=75):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks
````
- Splits by word count (75 words default)
- Maintains sentence boundaries when possible

#### Manual Embedding
````python
response = requests.post(
    ollama_api_url + "/api/embeddings",
    json={"model": "nomic-embed-text:latest", "prompt": chunk}
)
embedding = response.json()["embedding"]
````

#### Vector Upsert
````python
items_to_upsert = [
    (f"{i}", embeddings[i], {"text": chunks[i]})
    for i in range(len(embeddings))
]
index.upsert(vectors=items_to_upsert)
````

#### Query Process
1. Convert question to embedding
2. Search Pinecone for top-k matches
3. Extract metadata text
4. Construct prompt with context
5. Generate answer via Ollama

---

## Configuration

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `PINECONE_API_KEY` | Your Pinecone API key | `pcsk_xxxxx` |
| `PINECONE_HOST` | Pinecone index host URL | `https://your-index.pinecone.io` |
| `OLLAMA_API_URL` | Ollama API endpoint | `http://localhost:11434` |

### Pinecone Index Settings

**LangChain Implementation:**
- **Index name**: `faq-rag-test`
- **Dimension**: 384 (matches `all-minilm:l6-v2`)
- **Metric**: cosine
- **Spec**: Serverless (AWS us-east-1)

**Manual Implementation:**
- **Index name**: `pdf-rag-test`
- **Dimension**: 768 (matches `nomic-embed-text`)
- **Metric**: cosine

### Model Configuration

**LangChain:**
````python
llm = OllamaLLM(
    model="qwen2.5:7b-instruct",
    base_url=OLLAMA_API_URL,
    max_tokens=512,
    temperature=0.0  # Deterministic answers
)
````

**Manual:**
````python
{
    "model": "qwen2.5:0.5b",
    "prompt": llm_prompt,
    "stream": False
}
````

### Customization Options

1. **Change retrieval count**: Modify `"k": 5` in retriever setup
2. **Adjust chunk size**: Change `chunk_size=75` in manual chunking
3. **Switch embedding models**: Update model name (ensure dimensions match)
4. **Modify LLM**: Change model in `OllamaLLM` or generate call
5. **Customize prompt**: Edit `prompt_template` for different response styles

---

## Dependencies

### Critical Dependencies

#### LangChain Ecosystem
- **`langchain`** (0.3+): Core framework for LLM applications
- **`langchain-pinecone`**: Pinecone vector store integration
- **`langchain-ollama`**: Ollama LLM and embeddings integration
- **`langchain-text-splitters`**: Document chunking utilities
- **`langchain-core`**: Core abstractions

#### Vector Database
- **`pinecone`**: Official Pinecone Python SDK
- **`pinecone-plugin-assistant`**: Additional Pinecone features
- **`faiss-cpu`**: Optional local vector search (Facebook AI)

#### LLM Integration
- **`ollama`**: Python client for Ollama
- **`openai`**: OpenAI API client (for potential future use)

#### Document Processing
- **`PyPDF2`**: PDF text extraction
- **`pypdf`**: Alternative PDF library

#### HTTP & Networking
- **`requests`**: Simple HTTP library
- **`httpx`**: Modern async HTTP client
- **`aiohttp`**: Async HTTP client/server

#### Configuration & Utilities
- **`python-dotenv`**: Load environment variables from .env
- **`pydantic`**: Data validation using Python type hints
- **`tiktoken`**: Token counting for OpenAI models

#### Testing (Development)
- **`pytest`**: Testing framework
- **`pytest-asyncio`**: Async test support
- **`pytest-benchmark`**: Performance benchmarking

### Installation

Install all dependencies:
````bash
pip install -r requirements.txt
````

Install specific groups:
````bash
# Core dependencies only
pip install langchain langchain-pinecone langchain-ollama pinecone python-dotenv

# Document processing
pip install PyPDF2 pypdf

# Development tools
pip install pytest pytest-asyncio
````

---

## Development

### Setting Up Development Environment

1. **Create virtual environment**
   ````bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/Mac
   # or
   .venv\Scripts\activate  # Windows
   ````

2. **Install dependencies**
   ````bash
   pip install -r requirements.txt
   ````

3. **Configure environment**
   ````bash
   cp .env.example .env  # If example exists
   # Edit .env with your credentials
   ````

### Code Structure Best Practices

**For LangChain approach:**
- Keep embeddings and LLM initialization separate
- Use environment variables for all configuration
- Return source documents for transparency
- Implement error handling for API calls

**For manual approach:**
- Validate API responses before processing
- Handle pagination for large document sets
- Implement retry logic for network calls
- Cache embeddings when possible

### Adding New Features

**To add a new data source:**
1. Create a loader function in `lang/ingest.py` or `manual/ingest.py`
2. Convert to Document format (LangChain) or text chunks (manual)
3. Generate embeddings and upsert to Pinecone

**To customize prompts:**
1. Edit `prompt_template` in `lang/main.py`
2. Adjust rules and formatting as needed
3. Test with various question types

**To switch embedding models:**
1. Update model name in embeddings initialization
2. Check output dimensions
3. Update Pinecone index dimension if needed
4. Re-ingest all documents

### Testing

While there are no automated tests in the repository currently, you can manually test:

**Test ingestion:**
````bash
cd lang
python ingest.py
# Check Pinecone console for vector count
````

**Test retrieval:**
````bash
python main.py
# Ask questions and verify answers match source documents
````

**Test different models:**
````bash
ollama list  # See available models
# Update model in code
````

---

## Troubleshooting

### Common Issues

#### 1. Ollama Connection Error
**Symptom:** `Connection refused to localhost:11434`

**Solutions:**
- Ensure Ollama is running: `ollama serve`
- Check if port 11434 is accessible
- Verify OLLAMA_API_URL in .env

#### 2. Pinecone Authentication Error
**Symptom:** `401 Unauthorized` or `Invalid API key`

**Solutions:**
- Verify PINECONE_API_KEY in .env
- Check API key in Pinecone console
- Ensure no extra whitespace in .env file

#### 3. Dimension Mismatch Error
**Symptom:** `Dimension mismatch: expected 384, got 768`

**Solutions:**
- Ensure embedding model matches index dimension
  - `all-minilm:l6-v2` → 384 dimensions
  - `nomic-embed-text:latest` → 768 dimensions
- Delete and recreate index with correct dimension
- Or switch embedding model to match existing index

#### 4. Model Not Found
**Symptom:** `model 'qwen2.5:7b-instruct' not found`

**Solutions:**
````bash
ollama pull qwen2.5:7b-instruct
ollama pull all-minilm:l6-v2
ollama list  # Verify models are installed
````

#### 5. File Not Found (data/faq.json)
**Symptom:** `FileNotFoundError: ../data/faq.json`

**Solutions:**
- Create `data/` directory in project root
- Add your `faq.json` file with proper format
- Check file path is relative to script location

#### 6. Empty or Poor Quality Answers
**Symptom:** "I can't help you with that" or irrelevant answers

**Solutions:**
- Increase retrieval count: `"k": 10`
- Check if documents were ingested successfully
- Verify question relates to ingested content
- Review retrieved context in output
- Adjust chunk size for better context

#### 7. Slow Response Times
**Symptom:** Queries take very long to complete

**Solutions:**
- Use smaller LLM model (e.g., `qwen2.5:0.5b`)
- Reduce `max_tokens` in LLM config
- Decrease retrieval count (`k` value)
- Ensure Ollama has sufficient resources

#### 8. Out of Memory Errors
**Symptom:** Python crashes during embedding or LLM generation

**Solutions:**
- Use smaller models
- Reduce batch size when ingesting
- Process documents in chunks
- Increase system RAM or use swap

### Debugging Tips

**Enable verbose output:**
````python
import logging
logging.basicConfig(level=logging.DEBUG)
````

**Check Pinecone index stats:**
````python
from pinecone import Pinecone
pc = Pinecone(api_key=api_key)
index = pc.Index("faq-rag-test")
print(index.describe_index_stats())
````

**Test Ollama directly:**
````bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:7b-instruct",
  "prompt": "Hello, how are you?",
  "stream": false
}'
````

**Verify embeddings:**
````python
from langchain_ollama import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="all-minilm:l6-v2")
result = embeddings.embed_query("test query")
print(f"Embedding dimension: {len(result)}")
````

### Getting Help

- **Ollama Issues**: [github.com/ollama/ollama/issues](https://github.com/ollama/ollama/issues)
- **LangChain Issues**: [github.com/langchain-ai/langchain/issues](https://github.com/langchain-ai/langchain/issues)
- **Pinecone Support**: [docs.pinecone.io](https://docs.pinecone.io)
- **Project Issues**: [github.com/KonradZwl/ai-rag/issues](https://github.com/KonradZwl/ai-rag/issues)

### Performance Optimization

1. **Batch processing**: Ingest multiple documents at once
2. **Caching**: Cache frequently used embeddings
3. **Index optimization**: Use appropriate Pinecone pod/serverless tier
4. **Model selection**: Balance quality vs speed with model choice
5. **Retrieval tuning**: Find optimal `k` value for your use case

---

## License

This project is open source. Please check the repository for license information.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

*Documentation last updated: February 2026*
