# WebCenter Content AI Search

A production-grade Retrieval-Augmented Generation (RAG) pipeline that enables natural language search over Oracle WebCenter Content (WCC/UCM) document repositories — running entirely on-premise with zero data leaving the network.

## Problem

Enterprise ECM systems like Oracle WebCenter Content store millions of documents. Finding the right information means navigating folder hierarchies, identifying the right file, then locating the right section. This is slow and requires tribal knowledge.

## Solution

This system lets authorized users ask plain-English questions and receive grounded, accurate answers retrieved directly from WCC documents — with no hallucination and full data residency compliance.

## Architecture

WCC Document

↓ Heading-aware chunking (section-based, with overlap)

↓ Local embeddings (MiniLM / nomic-embed-text)

↓ Vector store (ChromaDB, persisted)

↓ Semantic retrieval (top-k similarity search)

↓ Local LLM generation (Mistral 7B via Ollama)

↓ RAGAS evaluation (faithfulness, relevancy, precision)

Grounded Answer


## Key Features

- **100% on-premise** — No external API calls, no data leakage. Suitable for banking, government, and defense.
- **Grounded answers** — LLM responds only from retrieved context, refuses when answer isn't present.
- **Automated quality evaluation** — RAGAS scores every response on faithfulness, answer relevancy, and context precision.
- **WCC-aware chunking** — Splits documents by section headings to preserve context.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | LangChain |
| Embeddings | MiniLM-L6-v2 / nomic-embed-text |
| Vector Store | ChromaDB |
| LLM | Mistral 7B (local via Ollama) |
| Evaluation | RAGAS |

## Evaluation Results

| Metric | Score | Threshold |
|--------|-------|-----------|
| Faithfulness | 1.00 | > 0.85 |
| Answer Relevancy | 1.00 | > 0.80 |
| Context Precision | 1.00 | > 0.75 |

*Note: Perfect scores reflect a controlled test set. Production evaluation requires a golden dataset of 50+ diverse queries including multi-document and absent-answer edge cases.*

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Pull local models
ollama pull mistral
ollama pull nomic-embed-text

# Run
ollama serve  # in a separate terminal
python wcc_rag.py
```

## Roadmap

- [ ] WCC REST API integration for live document ingestion
- [ ] Hybrid search (BM25 + semantic) for Oracle terminology
- [ ] Reranking layer for improved context precision
- [ ] Golden dataset evaluation at scale
- [ ] Metadata-based security filtering (dSecurityGroup)

---

Built by [Rahul Dumpala](https://github.com/RahulDumpala) — Oracle PaaS Architect exploring enterprise AI solutions.