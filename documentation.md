# AI-RAG: Comprehensive Documentation

## Table of Contents

1. [Project Overview](#project-overview)
2. [Getting Started](#getting-started)
3. [Usage Guide](#usage-guide)
4. [Project Structure](#project-structure)
5. [Development](#development)
6. [Dependencies](#dependencies)
7. [Configuration](#configuration)
8. [Examples](#examples)
9. [Troubleshooting](#troubleshooting)
10. [Performance Optimization](#performance-optimization)
11. [Security Considerations](#security-considerations)
12. [Debug Mode](#debug-mode)

---

## Project Overview

### What is AI-RAG?

AI-RAG (Retrieval-Augmented Generation) is a Python-based implementation of a RAG pipeline that combines the power of vector databases, embeddings, and large language models to provide accurate, context-aware answers to user questions. The system retrieves relevant information from a knowledge base and uses it to generate informed responses.

### Key Features

- **Dual Implementation Paths**: 
  - High-level LangChain framework approach (`lang/` directory)
  - Low-level manual implementation approach (`manual/` directory)
- **Multiple Data Source Support**:
  - FAQ JSON data with 384-dimensional embeddings
  - PDF document processing with 768-dimensional embeddings
- **Local LLM Integration**: Uses Ollama for privacy-conscious local inference
- **Vector Storage**: Pinecone cloud vector database for efficient similarity search
- **Flexible Embedding Models**: Support for multiple embedding models (all-minilm:l6-v2, nomic-embed-text)
- **Environment-based Configuration**: Easy setup through `.env` files
- **Interactive Query Interface**: Command-line interface for real-time Q&A

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Language** | Python 3.x | Core implementation |
| **Framework** | LangChain | High-level RAG orchestration |
| **Vector Database** | Pinecone | Serverless vector storage and retrieval |
| **LLM** | Ollama (qwen2.5, etc.) | Local language model inference |
| **Embeddings** | Ollama (all-minilm, nomic-embed-text) | Text vectorization |
| **PDF Processing** | PyPDF2 | PDF text extraction |
| **Environment Management** | python-dotenv | Configuration management |

### Architecture

The RAG pipeline follows this flow:

1. **Ingestion Phase**:
   - Documents/FAQs are loaded from source files
   - Text is chunked into manageable segments
   - Embeddings are generated for each chunk
   - Vectors are stored in Pinecone with metadata

2. **Query Phase**:
   - User question is embedded using the same model
   - Similar vectors are retrieved from Pinecone
   - Retrieved context is formatted into a prompt
   - LLM generates an answer based on the context

---

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8+**: [Download Python](https://www.python.org/downloads/)
- **Ollama**: [Install Ollama](https://ollama.com/)
- **Pinecone Account**: [Sign up for Pinecone](https://www.pinecone.io/)
- **Git**: For cloning the repository

### Installation

#### Step 1: Clone the Repository

```bash
git clone https://github.com/KonradZwl/ai-rag.git
cd ai-rag
```

#### Step 2: Set Up Python Environment

Create and activate a virtual environment (recommended):

```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

#### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including LangChain, Pinecone, Ollama client, and more.

#### Step 4: Install Ollama Models

Pull the required embedding and LLM models:

```bash
# For FAQ data (384 dimensions)
ollama pull all-minilm:l6-v2

# For PDF data (768 dimensions)
ollama pull nomic-embed-text:latest

# For text generation
ollama pull qwen2.5:7b-instruct

# Alternative smaller model for manual implementation
ollama pull qwen2.5:0.5b
```

Verify Ollama is running:

```bash
curl http://localhost:11434/api/version
```

#### Step 5: Set Up Pinecone

1. Create a Pinecone account at [pinecone.io](https://www.pinecone.io/)
2. Create an API key from the Pinecone console
3. Note your Pinecone environment region (e.g., `us-east-1`)

### Initial Configuration

#### Create Environment File

Create a `.env` file in the project root:

```bash
touch .env
```

Add the following configuration (see [Configuration](#configuration) section for details):

```env
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_HOST=your_pinecone_host_url
PINECONE_LANG=your_pinecone_host_url_lang_version
OLLAMA_API_URL=http://localhost:11434
```

#### Create Data Directory

If you're working with custom data:

```bash
mkdir -p data
```

---

## Usage Guide

The project provides two distinct implementation approaches. Choose the one that best fits your needs.

### LangChain Implementation (Recommended for Beginners)

The LangChain approach (`lang/` directory) provides a high-level, abstracted interface.

#### Ingesting FAQ Data

1. Prepare your FAQ data in JSON format (see [Examples](#examples))
2. Place the file at `data/faq.json`
3. Run the ingestion script:

```bash
cd lang
python ingest.py
```

This will:
- Load FAQ data from `data/faq.json`
- Create a Pinecone index named `faq-rag-test` (384 dimensions)
- Generate embeddings using `all-minilm:l6-v2`
- Upload vectors to Pinecone

#### Querying the System

Start the interactive Q&A interface:

```bash
cd lang
python main.py
```

Example interaction:

```
Ask your question (or 'exit'): What is machine learning?

Retrieved context:
Q: What is machine learning?
A: Machine learning is a subset of artificial intelligence...

Answer: Machine learning is a subset of artificial intelligence that enables 
systems to learn and improve from experience without being explicitly programmed.
```

Type `exit` or `quit` to terminate the session.

### Manual Implementation (For Learning/Customization)

The manual approach (`manual/` directory) provides low-level control over the RAG pipeline.

#### Ingesting PDF Documents

1. Place your PDF file at `data/info.pdf`
2. Run the ingestion script:

```bash
cd manual
python ingest.py
```

This will:
- Extract text from the PDF
- Chunk text into 75-word segments
- Generate embeddings using `nomic-embed-text:latest`
- Create a Pinecone index named `pdf-rag-test` (768 dimensions)
- Upload vectors with metadata

#### Querying PDF Data

Run the query script:

```bash
cd manual
python query.py
```

Example:

```
Ask your question: What topics are covered in the document?
Answer: The document covers topics including...
```

### Key Differences Between Implementations

| Feature | LangChain (`lang/`) | Manual (`manual/`) |
|---------|--------------------|--------------------|
| **Abstraction Level** | High-level, less code | Low-level, more control |
| **Embedding Model** | all-minilm:l6-v2 (384d) | nomic-embed-text (768d) |
| **Data Source** | FAQ JSON | PDF documents |
| **LLM Model** | qwen2.5:7b-instruct | qwen2.5:0.5b |
| **Retrieval** | LangChain retriever | Direct Pinecone queries |
| **Best For** | Production, rapid development | Learning, customization |

---

## Project Structure

```
ai-rag/
├── .github/
│   └── workflows/          # GitHub Actions workflows
│       ├── documentation-update.md
│       └── repository-activity-summary.md
├── lang/                   # LangChain-based implementation
│   ├── ingest.py          # FAQ data ingestion (384d embeddings)
│   └── main.py            # Interactive query interface
├── manual/                 # Manual implementation
│   ├── ingest.py          # PDF ingestion (768d embeddings)
│   └── query.py           # Query script for PDF data
├── data/                   # Data directory (create manually)
│   ├── faq.json           # FAQ data in JSON format
│   └── info.pdf           # PDF documents for processing
├── .env                    # Environment variables (create from .env.example)
├── .gitignore             # Git ignore patterns
├── requirements.txt       # Python dependencies
├── README.md              # Quick start guide
└── documentation.md       # This comprehensive documentation
```

### File Descriptions

#### `lang/ingest.py`
- Loads FAQ data from JSON file
- Creates Pinecone index with 384 dimensions
- Uses LangChain's document loaders and text splitters
- Generates embeddings with `all-minilm:l6-v2`
- Stores vectors in Pinecone using LangChain's PineconeVectorStore

#### `lang/main.py`
- Initializes LangChain's RetrievalQA chain
- Configures custom prompt template for precise answers
- Retrieves top 5 relevant context chunks
- Uses `qwen2.5:7b-instruct` for answer generation
- Displays retrieved context and final answer

#### `manual/ingest.py`
- Extracts text from PDF files using PyPDF2
- Implements custom text chunking (75-word segments)
- Makes direct HTTP requests to Ollama API
- Uses `nomic-embed-text:latest` (768 dimensions)
- Directly interfaces with Pinecone SDK for vector storage

#### `manual/query.py`
- Embeds user question using Ollama API
- Performs similarity search in Pinecone
- Retrieves top 3 matching chunks with metadata
- Constructs prompt with retrieved context
- Uses `qwen2.5:0.5b` for lightweight inference

#### `requirements.txt`
- Contains all Python package dependencies
- Includes LangChain ecosystem packages
- Pinecone client libraries
- Ollama Python client
- PDF processing libraries

---

## Development

### Setting Up Development Environment

1. **Fork and Clone**: Fork the repository and clone your fork
2. **Create Branch**: Create a feature branch for your work
3. **Install Dependencies**: Follow the installation steps in [Getting Started](#getting-started)
4. **Configure Environment**: Set up your `.env` file with test credentials

### Code Patterns and Best Practices

#### Environment Variables

Always load environment variables at the top of your script:

```python
import os
from dotenv import load_dotenv

load_dotenv()
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
```

#### Document Processing

When creating documents for LangChain:

```python
from langchain.docstore.document import Document

doc = Document(
    page_content="Your text content here",
    metadata={"source": "filename", "category": "faq"}
)
```

#### Error Handling

Implement proper error handling for API calls:

```python
try:
    response = requests.post(ollama_api_url + "/api/embeddings", json=payload)
    response.raise_for_status()
    embedding = response.json()["embedding"]
except requests.exceptions.RequestException as e:
    print(f"Error calling Ollama API: {e}")
    # Handle error appropriately
```

### Customization Guide

#### Changing Embedding Models

To use a different embedding model:

1. **Pull the model**: `ollama pull <model-name>`
2. **Update the code**:
   ```python
   # LangChain
   embeddings = OllamaEmbeddings(model="your-model-name", base_url=OLLAMA_API_URL)
   
   # Manual
   response = requests.post(
       ollama_api_url + "/api/embeddings",
       json={"model": "your-model-name", "prompt": text}
   )
   ```
3. **Adjust Pinecone dimension**: Match the new model's output dimensions

#### Modifying Chunk Size

For the manual implementation:

```python
def chunk_text(text, chunk_size=100):  # Adjust chunk_size
    words = text.split()
    chunks = [] 
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks
```

For LangChain:

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Adjust chunk size
    chunk_overlap=50     # Adjust overlap
)
```

#### Adjusting Retrieval Parameters

Modify the number of retrieved documents:

```python
# LangChain
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # Top 10 results

# Manual
results = index.query(vector=question_embedding, top_k=5)  # Top 5 results
```

#### Customizing LLM Prompts

Edit the prompt template in `lang/main.py`:

```python
prompt_template = """
You are a [describe role]. 
Use the context below to [describe task].

Rules:
1. [Your rule 1]
2. [Your rule 2]

Context:
{context}

Question:
{question}

Answer:
"""
```

### Testing

Currently, the project does not include automated tests. For manual testing:

1. **Test Ingestion**: Verify vectors are uploaded to Pinecone
2. **Test Retrieval**: Check that relevant context is retrieved
3. **Test Generation**: Ensure answers are accurate and relevant
4. **Test Edge Cases**: Empty queries, no matching context, etc.

---

## Dependencies

### Core Dependencies

| Package | Version Range | Purpose |
|---------|--------------|---------|
| **langchain** | Latest | Core RAG framework and orchestration |
| **langchain-community** | Latest | Community integrations for LangChain |
| **langchain-core** | Latest | Core LangChain abstractions |
| **langchain-ollama** | Latest | Ollama integration for LangChain |
| **langchain-pinecone** | Latest | Pinecone vector store integration |
| **langchain-text-splitters** | Latest | Text chunking utilities |
| **pinecone** | Latest | Pinecone SDK for vector operations |
| **pinecone-plugin-assistant** | Latest | Additional Pinecone features |
| **ollama** | Latest | Python client for Ollama API |
| **python-dotenv** | Latest | Environment variable management |
| **PyPDF2** | Latest | PDF text extraction |
| **requests** | Latest | HTTP client for API calls |

### Supporting Libraries

| Package | Purpose |
|---------|---------|
| **aiohttp** | Async HTTP client for performance |
| **httpx** | Modern HTTP client with async support |
| **pydantic** | Data validation and settings management |
| **pydantic-settings** | Settings management with Pydantic |
| **numpy** | Numerical operations for embeddings |
| **tqdm** | Progress bars for long operations |
| **tenacity** | Retry logic for API calls |
| **SQLAlchemy** | Database abstraction (if needed) |

### Development Dependencies

| Package | Purpose |
|---------|---------|
| **pytest** | Testing framework |
| **pytest-asyncio** | Async test support |
| **pytest-benchmark** | Performance testing |
| **pytest-socket** | Network call control in tests |

### Optional Dependencies

| Package | Purpose |
|---------|---------|
| **faiss-cpu** | Alternative vector search (local) |
| **tiktoken** | Token counting for OpenAI models |
| **langsmith** | LangChain debugging and monitoring |

### Installation Notes

- All dependencies can be installed via: `pip install -r requirements.txt`
- For GPU acceleration: Consider `faiss-gpu` instead of `faiss-cpu`
- Python 3.8+ is required for full compatibility
- Some packages may have system-level dependencies (e.g., build tools)

---

## Configuration

### Environment Variables

Create a `.env` file in the project root with the following variables:

| Variable | Description | Example Value | Required |
|----------|-------------|---------------|----------|
| `PINECONE_API_KEY` | Your Pinecone API key | `abc123...` | Yes |
| `PINECONE_HOST` | Pinecone index host URL (for manual impl.) | `https://pdf-rag-test-abc123.svc.aped-4627-b74a.pinecone.io` | For manual only |
| `PINECONE_LANG` | Pinecone host for LangChain implementation | `https://faq-rag-test-xyz456.svc.aped-4627-b74a.pinecone.io` | For lang only |
| `OLLAMA_API_URL` | Ollama API endpoint | `http://localhost:11434` | Yes |

### Pinecone Configuration

#### Index Settings for FAQ Data (LangChain)

```python
index_name = "faq-rag-test"
dimension = 384              # all-minilm:l6-v2 output size
metric = "cosine"            # Similarity metric
cloud = "aws"                # Cloud provider
region = "us-east-1"         # AWS region
```

#### Index Settings for PDF Data (Manual)

```python
index_name = "pdf-rag-test"
dimension = 768              # nomic-embed-text output size
metric = "cosine"            # Similarity metric
```

#### Creating Indexes Programmatically

The ingestion scripts automatically create indexes if they don't exist:

```python
# LangChain approach
if index_name not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Manual approach
if index_name not in existing_names:
    pc.create_index(name=index_name, dimension=768, metric="cosine")
```

### Ollama Model Configuration

#### Embedding Models

| Model | Dimensions | Use Case | Size |
|-------|-----------|----------|------|
| `all-minilm:l6-v2` | 384 | General purpose, fast | ~23MB |
| `nomic-embed-text:latest` | 768 | Higher quality, slower | ~274MB |
| `mxbai-embed-large` | 1024 | Best quality | ~669MB |

#### LLM Models

| Model | Parameters | Use Case | Size |
|-------|-----------|----------|------|
| `qwen2.5:0.5b` | 0.5B | Fast, lightweight queries | ~352MB |
| `qwen2.5:7b-instruct` | 7B | Balanced quality and speed | ~4.7GB |
| `llama3:8b` | 8B | High quality responses | ~4.7GB |
| `mistral:7b` | 7B | Alternative to Qwen | ~4.1GB |

#### Model Configuration Options

When initializing Ollama models in code:

```python
llm = OllamaLLM(
    model="qwen2.5:7b-instruct",
    base_url=OLLAMA_API_URL,
    max_tokens=512,              # Maximum response length
    temperature=0.0,             # 0 = deterministic, 1 = creative
    top_p=1.0,                   # Nucleus sampling parameter
    top_k=50,                    # Top-k sampling parameter
)
```

---

## Examples

### Example `.env` File

```env
# Pinecone Configuration
PINECONE_API_KEY=pcsk_abc123_AbCdEfGhIjKlMnOpQrStUvWxYz1234567890

# Pinecone Index Hosts
PINECONE_HOST=https://pdf-rag-test-abc123.svc.aped-4627-b74a.pinecone.io
PINECONE_LANG=https://faq-rag-test-xyz456.svc.aped-4627-b74a.pinecone.io

# Ollama Configuration
OLLAMA_API_URL=http://localhost:11434
```

### Example `faq.json` File

Create this file at `data/faq.json`:

```json
{
  "faq": [
    {
      "question": "What is machine learning?",
      "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data and make predictions or decisions."
    },
    {
      "question": "What is the difference between AI and machine learning?",
      "answer": "Artificial Intelligence (AI) is the broader concept of machines being able to carry out tasks in a way that we would consider 'smart'. Machine Learning (ML) is a subset of AI that focuses on the ability of machines to receive data and learn for themselves."
    },
    {
      "question": "What are neural networks?",
      "answer": "Neural networks are computing systems inspired by the biological neural networks in animal brains. They consist of layers of interconnected nodes (neurons) that process and transmit information, allowing the system to learn complex patterns."
    },
    {
      "question": "What is deep learning?",
      "answer": "Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks). These networks can automatically learn hierarchical representations of data, making them particularly effective for tasks like image recognition and natural language processing."
    },
    {
      "question": "What is natural language processing?",
      "answer": "Natural Language Processing (NLP) is a branch of AI that focuses on enabling computers to understand, interpret, and generate human language. It combines computational linguistics with machine learning and deep learning to process text and speech."
    }
  ]
}
```

### Example Custom Prompt

```python
prompt_template = """
You are an expert AI assistant specializing in technical documentation.
Your task is to provide clear, accurate answers based solely on the provided context.

Instructions:
1. Only use information from the context below
2. If the answer isn't in the context, say "I don't have enough information to answer that"
3. Be concise but complete
4. Use technical terminology when appropriate
5. Format code or technical terms with proper syntax

Context:
{context}

Question: {question}

Detailed Answer:
"""
```

### Example Query Session

```bash
$ cd lang
$ python main.py

Ask your question (or 'exit'): What is machine learning?

Retrieved context:
Q: What is machine learning?
A: Machine learning is a subset of artificial intelligence that enables 
systems to learn and improve from experience without being explicitly 
programmed...

Answer: Machine learning is a subset of artificial intelligence that enables 
systems to learn and improve from experience without being explicitly programmed. 
It uses algorithms to identify patterns in data and make predictions or decisions.

Ask your question (or 'exit'): exit
```

---

## Troubleshooting

### 1. Ollama Connection Error

**Problem**: `requests.exceptions.ConnectionError: Failed to establish a connection to Ollama`

**Solutions**:
- Verify Ollama is running: `curl http://localhost:11434/api/version`
- Start Ollama: `ollama serve`
- Check the port in your `.env` file matches Ollama's port
- Ensure no firewall is blocking localhost connections

### 2. Model Not Found

**Problem**: `Error: model 'all-minilm:l6-v2' not found`

**Solutions**:
- Pull the required model: `ollama pull all-minilm:l6-v2`
- List installed models: `ollama list`
- Check model name spelling in your code
- Wait for the model to finish downloading if recently pulled

### 3. Pinecone Authentication Error

**Problem**: `UnauthorizedException: Invalid API key`

**Solutions**:
- Verify your API key in the Pinecone console
- Check `.env` file exists and contains `PINECONE_API_KEY`
- Ensure no extra spaces or quotes around the key
- Try regenerating the API key in Pinecone console
- Verify `.env` is in the correct directory

### 4. Dimension Mismatch Error

**Problem**: `ValueError: dimension mismatch: 384 != 768`

**Solutions**:
- Delete the existing Pinecone index with wrong dimensions
- Ensure embedding model matches index dimensions:
  - `all-minilm:l6-v2` = 384 dimensions
  - `nomic-embed-text:latest` = 768 dimensions
- Check you're using the correct ingestion script for your data type
- Create a new index with the correct dimensions

### 5. Empty or No Results

**Problem**: Queries return no results or empty context

**Solutions**:
- Verify data was ingested: Check Pinecone console for vector count
- Re-run the ingestion script
- Check index name matches between ingest and query scripts
- Verify the data file exists at the expected path
- Increase `top_k` or `k` parameter for more results
- Check embedding model is the same for ingestion and queries

### 6. Slow Response Times

**Problem**: Queries take too long to return answers

**Solutions**:
- Use smaller LLM model (e.g., `qwen2.5:0.5b` instead of `7b`)
- Reduce `max_tokens` parameter
- Decrease `top_k` to retrieve fewer documents
- Use smaller embedding model (`all-minilm` instead of `nomic-embed-text`)
- Check Ollama is using GPU if available
- Optimize chunk size during ingestion

### 7. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'langchain'`

**Solutions**:
- Activate your virtual environment
- Install dependencies: `pip install -r requirements.txt`
- Verify Python version is 3.8+: `python --version`
- Update pip: `pip install --upgrade pip`
- Try reinstalling in a fresh virtual environment

### 8. PDF Extraction Issues

**Problem**: PDF text extraction fails or returns gibberish

**Solutions**:
- Verify PDF is not encrypted or password-protected
- Check PDF is text-based, not scanned images
- Try alternative PDF library: `pip install pdfplumber`
- Use OCR for scanned PDFs: `pip install pytesseract`
- Inspect extracted text before embedding
- Consider using `pypdf` instead of `PyPDF2`

### Additional Tips

- **Check Logs**: Run with verbose logging enabled
- **Test Components**: Test embedding, retrieval, and generation separately
- **Verify Connectivity**: Ensure internet access for Pinecone API
- **Resource Limits**: Check disk space and memory availability
- **Documentation**: Consult official docs for Pinecone, Ollama, and LangChain

---

## Performance Optimization

### Speed vs. Accuracy Tradeoffs

#### 1. Embedding Model Selection

| Priority | Model | Dimensions | Speed | Quality |
|----------|-------|-----------|-------|---------|
| **Speed** | all-minilm:l6-v2 | 384 | Fast | Good |
| **Balanced** | nomic-embed-text | 768 | Medium | Better |
| **Quality** | mxbai-embed-large | 1024 | Slow | Best |

**Recommendation**: Use `all-minilm:l6-v2` for FAQ/short text, `nomic-embed-text` for documents.

#### 2. LLM Model Size

| Model | Parameters | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| qwen2.5:0.5b | 0.5B | Very Fast | Basic | Simple Q&A |
| qwen2.5:3b | 3B | Fast | Good | General use |
| qwen2.5:7b | 7B | Medium | Better | Detailed answers |
| llama3:8b | 8B | Slower | Best | Complex reasoning |

**Recommendation**: Start with 0.5B for testing, upgrade to 7B for production.

#### 3. Chunk Size Optimization

**Smaller Chunks (50-100 words)**:
- ✅ More precise retrieval
- ✅ Faster processing
- ❌ May miss context
- Best for: FAQs, definitions

**Larger Chunks (200-500 words)**:
- ✅ More context preserved
- ✅ Better for complex topics
- ❌ Slower retrieval
- ❌ Less precise matches
- Best for: Documents, articles

**Recommendation**: 75-150 words for general purpose.

#### 4. Retrieval Parameters

```python
# Fast but less context
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Balanced
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Comprehensive but slower
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
```

**Recommendation**: k=5 for most use cases, k=3 for speed, k=10 for completeness.

#### 5. Pinecone Index Type

- **Serverless** (current): Auto-scaling, pay-per-use, slower cold starts
- **Pod-based**: Always-on, consistent performance, fixed cost

**Recommendation**: Serverless for development, pods for high-traffic production.

### Performance Best Practices

1. **Cache Embeddings**: Store frequently used embeddings to avoid recomputation
2. **Batch Processing**: Embed multiple documents at once during ingestion
3. **Async Operations**: Use async/await for concurrent API calls
4. **Connection Pooling**: Reuse HTTP connections to Ollama and Pinecone
5. **Lazy Loading**: Load models only when needed
6. **Index Optimization**: Use appropriate similarity metrics (cosine for normalized vectors)

### Monitoring Performance

Track these metrics:
- **Embedding Time**: How long to generate embeddings
- **Retrieval Time**: Time to query Pinecone
- **Generation Time**: LLM response latency
- **End-to-End Latency**: Total time from question to answer

Example monitoring code:

```python
import time

start = time.time()
# Your RAG code here
end = time.time()
print(f"Query took {end - start:.2f} seconds")
```

---

## Security Considerations

### API Key Management

#### Best Practices

1. **Never commit secrets**: Keep `.env` files out of version control
2. **Use environment variables**: Load sensitive data from `.env` files
3. **Rotate keys regularly**: Update API keys every 90 days
4. **Limit key permissions**: Use read-only keys when possible
5. **Separate environments**: Different keys for dev/staging/production

#### Securing `.env` Files

```bash
# Ensure .env is in .gitignore
echo ".env" >> .gitignore

# Set proper file permissions (Unix/Linux)
chmod 600 .env

# Never share .env files via email or chat
```

#### Key Rotation Process

1. Generate new API key in Pinecone console
2. Update `.env` file with new key
3. Test application with new key
4. Delete old key from Pinecone console
5. Update any CI/CD secrets

### Data Privacy

#### Local vs. Cloud Processing

- **Local (Ollama)**: LLM processing stays on your machine
- **Cloud (Pinecone)**: Vector storage is in the cloud
- **Consideration**: Sensitive data is sent to Pinecone

#### Data Handling Recommendations

1. **Anonymize PII**: Remove personal information before ingestion
2. **Encrypt at Rest**: Use Pinecone's built-in encryption
3. **Access Control**: Limit who can access Pinecone indexes
4. **Data Retention**: Delete old data regularly
5. **Audit Logs**: Track access to sensitive data

#### GDPR Compliance

- Document what data is stored in Pinecone
- Implement data deletion procedures
- Provide data export capabilities
- Maintain user consent records

### Network Security

1. **Use HTTPS**: Ensure all API calls use secure connections
2. **Firewall Rules**: Restrict Pinecone access to known IPs
3. **VPN/Private Networks**: Use for sensitive deployments
4. **Rate Limiting**: Implement to prevent abuse

### Code Security

```python
# Good: Use environment variables
api_key = os.getenv("PINECONE_API_KEY")

# Bad: Hardcoded secrets
api_key = "sk-abc123..."  # Never do this!

# Good: Validate inputs
if not question or len(question) > 1000:
    raise ValueError("Invalid question")

# Good: Handle errors securely
except Exception as e:
    logger.error("Query failed")  # Don't expose internal details
```

### Dependency Security

1. **Regular Updates**: Keep dependencies up to date
2. **Vulnerability Scanning**: Use tools like `pip-audit`
3. **Pin Versions**: Use specific versions in `requirements.txt`
4. **Review Dependencies**: Understand what each package does

```bash
# Check for vulnerabilities
pip install pip-audit
pip-audit

# Update dependencies safely
pip install --upgrade langchain pinecone-client
```

### Ollama Security

1. **Local Only**: Don't expose Ollama API to the internet
2. **Firewall**: Block external access to port 11434
3. **Model Verification**: Only use official Ollama models
4. **Resource Limits**: Prevent DoS via resource exhaustion

---

## Debug Mode

### Enabling Logging

#### Python Logging

Add to the top of your script:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
```

#### LangChain Debug Mode

```python
from langchain.globals import set_debug

set_debug(True)  # Enable verbose LangChain logging
```

This will show:
- LLM inputs and outputs
- Retrieval results
- Chain execution steps
- API calls and responses

### Debugging Techniques

#### 1. Inspect Retrieved Context

```python
response = qa_chain.invoke({"query": question})
print("Retrieved Documents:")
for i, doc in enumerate(response["source_documents"]):
    print(f"\n--- Document {i+1} ---")
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
    print(f"Score: {doc.metadata.get('score', 'N/A')}")
```

#### 2. Test Embeddings

```python
# Generate test embedding
test_text = "sample text"
embedding = embeddings.embed_query(test_text)
print(f"Embedding dimensions: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")
```

#### 3. Verify Pinecone Connection

```python
from pinecone import Pinecone

pc = Pinecone(api_key=PINECONE_API_KEY)
print("Available indexes:", pc.list_indexes())

index = pc.Index(index_name)
stats = index.describe_index_stats()
print(f"Index stats: {stats}")
```

#### 4. Test Ollama Connection

```python
import requests

# Test API availability
response = requests.get(f"{OLLAMA_API_URL}/api/version")
print(f"Ollama version: {response.json()}")

# List available models
response = requests.get(f"{OLLAMA_API_URL}/api/tags")
print(f"Available models: {response.json()}")
```

#### 5. Trace API Calls

```python
import time

def timed_request(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    duration = time.time() - start
    print(f"{func.__name__} took {duration:.2f}s")
    return result

# Usage
embedding = timed_request(embeddings.embed_query, question)
```

### Common Debug Scenarios

#### Empty Results

```python
# Check if vectors exist in index
index = pc.Index(index_name)
stats = index.describe_index_stats()
if stats['total_vector_count'] == 0:
    print("Error: No vectors in index. Run ingestion script first.")
```

#### Incorrect Answers

```python
# Reduce temperature for more deterministic answers
llm = OllamaLLM(
    model="qwen2.5:7b-instruct",
    temperature=0.0  # More deterministic
)

# Increase k to retrieve more context
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
```

#### Slow Performance

```python
# Time each component
start = time.time()
query_embedding = embeddings.embed_query(question)
print(f"Embedding: {time.time() - start:.2f}s")

start = time.time()
docs = vectorstore.similarity_search(question, k=5)
print(f"Retrieval: {time.time() - start:.2f}s")

start = time.time()
answer = llm.invoke(prompt)
print(f"Generation: {time.time() - start:.2f}s")
```

### Diagnostic Commands

```bash
# Check Python environment
python --version
pip list | grep -E "langchain|pinecone|ollama"

# Check Ollama status
ollama list
curl http://localhost:11434/api/version

# Check disk space (for models)
df -h

# Monitor system resources
htop  # or top on Mac/Linux
```

### LangSmith Integration (Optional)

For advanced debugging, integrate LangSmith:

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-key"
os.environ["LANGCHAIN_PROJECT"] = "ai-rag-debug"
```

This provides:
- Visual trace of all LangChain operations
- Input/output logging
- Performance metrics
- Error tracking

---

## Additional Resources

### Official Documentation

- [LangChain Documentation](https://python.langchain.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Ollama Documentation](https://ollama.com/docs)
- [PyPDF2 Documentation](https://pypdf2.readthedocs.io/)

### Tutorials and Guides

- [RAG Fundamentals](https://python.langchain.com/docs/tutorials/rag/)
- [Vector Databases Explained](https://www.pinecone.io/learn/vector-database/)
- [Local LLMs with Ollama](https://ollama.com/library)

### Community

- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [Pinecone Community](https://community.pinecone.io/)
- [Ollama GitHub](https://github.com/ollama/ollama)

---

## License

This project is provided as-is for educational and development purposes. Please check the repository for license information.

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

*Documentation Version: 1.0*  
*Last Updated: February 18, 2026*  
*Generated for AI-RAG Project*
