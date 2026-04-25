# AI Financial Analyst

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-412991?style=flat-square&logo=openai&logoColor=white)](https://openai.com)
[![FAISS](https://img.shields.io/badge/Vector_DB-FAISS-f7931e?style=flat-square)](https://faiss.ai)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-ff4b4b?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088ff?style=flat-square&logo=github-actions&logoColor=white)](https://github.com/features/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)

A Retrieval-Augmented Generation (RAG) system for financial analysis. It ingests earnings call transcripts, SEC filings, and real-time news, then answers natural-language questions with grounded, citation-backed responses and FinBERT sentiment analysis.

The system works entirely offline — without any API keys it falls back to template-based generation and bundled sample data.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Evaluation Metrics](#evaluation-metrics)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Configuration](#configuration)
- [Testing](#testing)
- [References](#references)

---

## Overview

AI Financial Analyst is built around three core capabilities:

**Semantic retrieval.** Financial documents are chunked, embedded into 384-dimensional vectors using `all-MiniLM-L6-v2`, and stored in a FAISS index. At query time, the user's question is embedded and the most relevant chunks are retrieved via cosine similarity.

**Grounded generation.** Retrieved chunks are injected into a structured prompt and sent to OpenAI GPT-3.5. The model is instructed to cite its sources, so every answer traces back to a specific document passage.

**Domain-aware sentiment.** Answers and retrieved passages are classified by FinBERT — a BERT model fine-tuned on financial corpora — which correctly interprets phrases like "beat estimates" or "soft guidance" that confuse general-purpose sentiment models.

A built-in evaluation framework measures faithfulness, answer relevance, and context recall against a 10-question golden QA set, making it straightforward to track improvements as the pipeline evolves.

---

## How It Works

### Pipeline Steps

1. **Ingest** — Financial documents (news articles, earnings transcripts, PDF filings) are parsed and split into overlapping text chunks to preserve context at chunk boundaries.

2. **Embed** — Each chunk is encoded into a 384-dimensional vector using `all-MiniLM-L6-v2` from [sentence-transformers](https://www.sbert.net/).

3. **Index** — Vectors are stored in a FAISS `IndexFlatIP` index with L2 normalization, enabling exact cosine-similarity search at sub-millisecond latency.

4. **Retrieve** — The user question is embedded and the top-K most similar chunks are fetched from FAISS.

5. **Generate** — Retrieved context is injected into a prompt and sent to OpenAI GPT-3.5. A local template-based fallback is used when the API is unavailable.

6. **Sentiment** — The answer and retrieved passages are run through FinBERT to produce a positive / neutral / negative label with a confidence score.

7. **Evaluate** — The evaluation framework scores faithfulness (token overlap between answer and context), answer relevance (cosine similarity between question and answer embeddings), and context recall (keyword overlap between expected answer and retrieved chunks).

### Design Decisions

**Why FAISS over a managed vector database?** For this project's scale (thousands of chunks), `IndexFlatIP` with normalized vectors delivers exact cosine similarity with sub-millisecond latency — no approximation, no network overhead, no cost. Migrating to Pinecone or Weaviate is straightforward if scale demands it; a `PINECONE_API_KEY` slot is already wired into the config.

**Why FinBERT over a general-purpose sentiment model?** General-purpose models perform poorly on financial language. FinBERT is fine-tuned on financial news and earnings calls, and correctly handles domain-specific phrasing that generic models misclassify.

---

## Architecture

```mermaid
flowchart TD
    subgraph ingestion ["Data Ingestion"]
        News["News (NewsAPI)"]
        PDF["PDF Reports (PyMuPDF)"]
        Transcripts["Earnings Transcripts"]
    end

    subgraph embedding ["Embed & Index"]
        Chunker["Chunker (overlapping windows)"]
        Embedder["all-MiniLM-L6-v2"]
        FAISS["FAISS Index (IndexFlatIP + L2 norm)"]
    end

    subgraph retrieval ["Retrieve & Generate"]
        Query["User Query"]
        Retrieve["Top-K Retrieval"]
        LLM["OpenAI GPT-3.5 / Local Fallback"]
        Answer["Answer + Citations"]
    end

    subgraph analysis ["Analysis"]
        FinBERT["FinBERT Sentiment"]
        Evaluator["RAGEvaluator"]
    end

    News --> Chunker
    PDF --> Chunker
    Transcripts --> Chunker
    Chunker --> Embedder --> FAISS

    Query --> Embedder
    Embedder --> Retrieve
    FAISS --> Retrieve
    Retrieve --> LLM --> Answer
    Answer --> FinBERT
    Answer --> Evaluator
```

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- An OpenAI API key (optional — the system falls back to local generation without one)
- A NewsAPI key (optional — enables real-time news ingestion)

### Installation

```bash
git clone https://github.com/krishi-shah/ai-financial-analyst.git
cd ai-financial-analyst
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your API keys
```

### Running the App

```bash
make run
# Opens the Streamlit dashboard at http://localhost:8501
```

### Makefile Reference

| Command | Description |
|---|---|
| `make install` | Install all Python dependencies |
| `make run` | Launch the Streamlit web app |
| `make test` | Run the full pytest test suite |
| `make evaluate` | Run RAG evaluation on the golden QA set |

---

## Evaluation Metrics

The evaluation framework in `evaluation/rag_evaluator.py` measures three dimensions of RAG quality over a 10-question golden QA set.

| Metric | What It Measures | How It Is Computed |
|---|---|---|
| Faithfulness | Is the answer grounded in retrieved context? | Token overlap between answer and retrieved chunks |
| Answer Relevance | Is the answer on-topic with the question? | Cosine similarity between question and answer embeddings |
| Context Recall | Did retrieval surface the right chunks? | Keyword overlap between expected answer and retrieved chunks |

Run `make evaluate` to produce a report:

```
============================================================
  RAG Evaluation Results
============================================================
  Questions evaluated :  10
  Avg Faithfulness    :  0.2369
  Avg Answer Relevance:  0.5272
  Avg Context Recall  :  0.9255
============================================================
```

Results are saved as timestamped JSON files in `evaluation/results/` for tracking improvements over time.

---

## Project Structure

```
ai-financial-analyst/
├── data_ingestion/
│   ├── news_scraper.py              # NewsAPI integration and article fetching
│   ├── earnings_call_parser.py      # Earnings transcript parsing
│   └── reports_parser.py            # PDF parsing via PyMuPDF
├── embeddings/
│   └── embedder.py                  # sentence-transformers text encoder
├── retrieval/
│   ├── rag_pipeline.py              # FAISS index and OpenAI generation pipeline
│   └── local_llm.py                 # Template-based offline fallback
├── sentiment/
│   └── sentiment_analyzer.py        # FinBERT sentiment classification
├── evaluation/
│   ├── rag_evaluator.py             # Faithfulness, relevance, and recall metrics
│   ├── golden_qa.json               # 10-pair golden QA set for benchmarking
│   └── results/                     # Timestamped evaluation output JSON files
├── ui/
│   └── streamlit_app.py             # Streamlit web dashboard
├── tests/
│   ├── test_embedder.py             # Shape, normalization, and similarity tests
│   ├── test_rag_pipeline.py         # Index construction and retrieval tests
│   └── test_sentiment.py            # Label validity and confidence range tests
├── .github/workflows/ci.yml         # GitHub Actions CI
├── config.py                        # Environment variable loader
├── .env.example                     # API key template
├── requirements.txt                 # Pinned Python dependencies
├── Makefile                         # Developer task shortcuts
└── README.md
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) | Encode text into 384-dimensional semantic vectors |
| Vector Search | FAISS (`faiss-cpu`, `IndexFlatIP`) | Exact cosine-similarity retrieval |
| LLM | OpenAI GPT-3.5 with local fallback | Citation-backed answer generation |
| Sentiment | FinBERT via HuggingFace `transformers` | Domain-specific financial sentiment classification |
| Data Ingestion | NewsAPI, PyMuPDF, BeautifulSoup | News, PDF, and transcript parsing |
| Evaluation | Custom `RAGEvaluator` | Faithfulness, relevance, and recall scoring |
| UI | Streamlit | Interactive query and results dashboard |
| CI/CD | GitHub Actions + pytest | Automated tests on every commit |
| Config | `python-dotenv` | Secure API key management via `.env` |

---

## Configuration

Copy `.env.example` to `.env` and fill in your keys:

```
OPENAI_API_KEY=sk-...        # Required for GPT-powered generation
NEWS_API_KEY=...              # Optional — enables real-time news ingestion
PINECONE_API_KEY=...          # Optional — managed vector DB, not used by default
YAHOO_FINANCE_API=...         # Optional — additional financial data
```

All keys are optional. Without any keys the system runs entirely offline using template-based generation and bundled sample data.

---

## Testing

```bash
# Run the full test suite
make test

# Run individual test files with verbose output
pytest tests/test_embedder.py -v
pytest tests/test_rag_pipeline.py -v
pytest tests/test_sentiment.py -v
```

Test coverage includes:

- **Embedder** — output shape, L2 normalization, semantic similarity ordering
- **RAG Pipeline** — FAISS index construction, top-K retrieval correctness, end-to-end query flow
- **Sentiment** — valid label assertion, confidence range `[0, 1]`, probability distribution sum

---

## References

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020) — [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
- Araci, *FinBERT: Financial Sentiment Analysis with Pre-Trained Language Models* (2019) — [arXiv:1908.10063](https://arxiv.org/abs/1908.10063)
- Johnson et al., *Billion-scale similarity search with GPUs — FAISS* (2019) — [arXiv:1702.08734](https://arxiv.org/abs/1702.08734)

---

## License

Distributed under the [MIT License](LICENSE).
