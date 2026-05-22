# AI-RAG: Retrieval-Augmented Generation met Pinecone en Ollama

Dit project demonstreert een Retrieval-Augmented Generation (RAG) pipeline met Python, LangChain, Pinecone en Ollama.
Het neemt FAQ-gegevens op, slaat vraag-antwoordparen op in een vectordatabase en beantwoordt gebruikersvragen met een lokaal LLM.

## Functionaliteiten

- Neemt FAQ-gegevens op vanuit JSON of PDF
- Splitst en embed documenten met Ollama
- Slaat embeddings op in een Pinecone vectordatabase
- Haalt relevante context op voor gebruikersvragen
- Gebruikt een lokaal LLM (Ollama) om antwoorden te genereren op basis van opgehaalde context
- Omgevingsconfiguratie via .env-bestand

## Installatie

**Omgevingsvariabelen configureren**
   - Bewerk .env in de projectroot:
     ```
     PINECONE_API_KEY=jouw_pinecone_api_sleutel
     PINECONE_HOST=jouw_pinecone_host_url
     PINECONE_LANG=jouw_pinecone_host_url_lang_versie
     OLLAMA_API_URL=http://localhost:11434
     ```
**Ollama starten**
   - Installeer en start Ollama lokaal: [Ollama-documentatie](https://ollama.com/)
   - Pull de benodigde modellen (bijv. `ollama pull all-minilm:l6-v2`)

## Gebruik

Het project bevat twee onafhankelijke implementaties van de RAG-pipeline:

### `lang/` -- Pipeline op basis van LangChain

Gebruikt LangChain-abstracties voor documentverwerking, embedding, retrieval en LLM-orchestratie.

1. **FAQ-gegevens opnemen** -- voer `lang/ingest.py` uit om `data/faq.json` in Pinecone te laden.
   ```bash
   cd lang
   python ingest.py
   ```
2. **Vragen stellen** -- voer `lang/main.py` uit voor een interactieve vraag-en-antwoordlus.
   ```bash
   python main.py
   ```

| Component | Detail |
|---|---|
| Embeddingmodel | `all-minilm:l6-v2` (384 dimensies) |
| LLM | `qwen2.5:7b-instruct` |
| Indexnaam | `faq-rag-test` |

### `manual/` -- Lichtgewicht pipeline (zonder LangChain)

Roept de Ollama- en Pinecone REST API's direct aan -- handig om te begrijpen wat LangChain abstraheert.

1. **Een PDF opnemen** -- voer `manual/ingest.py` uit om tekst uit `data/info.pdf` te extraheren, op te splitsen en in Pinecone op te slaan.
   ```bash
   cd manual
   python ingest.py
   ```
2. **Vragen stellen** -- voer `manual/query.py` uit om een enkele vraag te stellen en een beknopt antwoord te krijgen.
   ```bash
   python query.py
   ```

| Component | Detail |
|---|---|
| Embeddingmodel | `nomic-embed-text:latest` (768 dimensies) |
| LLM | `qwen2.5:0.5b` |
| Indexnaam | `pdf-rag-test` |

## Bestandsstructuur

```
.
├── data/
│   ├── faq.json          # FAQ-dataset (gebruikt door lang/)
│   └── info.pdf          # PDF-document (gebruikt door manual/)
├── lang/
│   ├── ingest.py         # FAQ-opname via LangChain
│   └── main.py           # Interactieve Q&A via LangChain
├── manual/
│   ├── ingest.py         # PDF-opname via directe API-aanroepen
│   └── query.py          # Enkele-vraag query via directe API-aanroepen
├── requirements.txt
├── .env                  # Omgevingsvariabelen (niet gecommit)
└── README.md
```

## Opmerkingen

- Zorg ervoor dat de dimensie van je Pinecone-index overeenkomt met de output van je embeddingmodel (`lang/` gebruikt 384, `manual/` gebruikt 768).
- Sla voor de beste resultaten zowel vragen als antwoorden samen op in elk document.
- Je kunt de chunkgrootte en retrievalparameters aanpassen in de code.
