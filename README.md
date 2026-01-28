# Local PDF Question Answering System (CPU-Only RAG)

This repository demonstrates a **fully local, interview-ready Retrieval-Augmented Generation (RAG) system** built under strict system constraints. The focus is on **clarity, correctness, and explainability**, not cloud APIs or large models.

It shows a complete end-to-end RAG pipeline and the reasoning behind each design choice.

---

## 1. Problem Statement

Build a **PDF-based Question Answering system** that:

* Accepts PDF documents
* Indexes content locally
* Retrieves relevant chunks
* Generates answers grounded only in retrieved context
* Runs fully offline (no paid APIs)
* Is easy to explain to an interviewer

---

## 2. Architecture Overview

### System Flow

```
Client
  ↓
FastAPI
  ↓
PDF Ingestion → Chunking → Embeddings
  ↓
Typesense (Vector + Keyword Search)
  ↓
Context Builder
  ↓
Local LLM (GPT4All)
  ↓
Answer
```

This diagram shows the **end-to-end data flow** from request to response. Each stage has a single, well-defined responsibility.

---

## 3. Core Components

### FastAPI

* Exposes `/upload`, `/ask`, `/history`
* Handles request/response only
* Keeps business logic separate from API layer

---

### PDF Processing

* Uses `PyPDF2` for stable text extraction on Windows
* Avoids `pdfminer` / `unstructured` dependency issues
* Produces raw text from PDFs

---

### Text Chunking

* Manual sliding-window chunking
* Fixed chunk size with overlap
* Deterministic, predictable behavior
* No framework abstraction or hidden logic

---

### Embeddings

* Model: `all-MiniLM-L6-v2`
* CPU-friendly, fast
* 384-dimensional vectors
* Converts text chunks into embeddings

---

### Vector Store (Typesense)

* Stores text + embeddings
* Supports:

  * Vector search (semantic)
  * Keyword search (lexical)
* Local, lightweight, no JVM dependency

---

### Retrieval (Hybrid)

* Combines:

  * Vector similarity results
  * Keyword matching results
* Improves recall over single-method retrieval
* Mimics production-grade RAG retrieval

---

### LLM Layer

* Model: GPT4All (small, CPU-only)
* No API keys or cloud dependency
* Runs fully locally
* Used only for reasoning over retrieved context

---

### Memory

* Simple in-memory dictionary per user
* Stores conversation history
* No LangChain dependency

---

## 3. System Constraints

**Hardware:**

* 8 GB RAM
* CPU-only
* Windows machine

**Implications:**

* Large LLMs are not feasible
* CPU inference must be stable
* Pipeline correctness > model accuracy

The goal is architectural clarity, not benchmark performance.

---

## 4. LLM Selection

### Rejected

* Cloud APIs (paid, internet dependency)
* Large local models (RAM pressure, slow CPU inference)

### Final Choice: GPT4All

* Designed for local CPU inference
* Small memory footprint
* Works offline after download

**Model used:**

```
ggml-gpt4all-j-v1.3-groovy
```

---

## 5. Why LangChain Was Removed

LangChain was initially evaluated but removed due to:

* Heavy dependencies and version conflicts (Windows)
* Hidden logic behind abstractions
* Reduced interview explainability

**Benefit of removal:**

* Full control over data flow
* Clear, debuggable logic
* Stronger system design explanation

This project focuses on **core RAG understanding**, not framework usage.

---

## 6. PDF Parsing

* Initial attempt: `unstructured.partition.pdf` (failed due to incompatibilities)
* Final choice: `PyPDF2`

`PyPDF2` is lightweight, stable, and sufficient for text-based PDFs on Windows.

---

## 7. Chunking Strategy

* Manual sliding-window chunking
* Fixed chunk size and overlap

This ensures deterministic behavior and avoids framework “magic”.

---

## 8. Embeddings

* Library: SentenceTransformers

* Model: `all-MiniLM-L6-v2`

* 384-dimensional embeddings

* Fast on CPU

* Strong performance-to-cost ratio

---

## 9. Vector Store & Retrieval

### Why Typesense

* Supports vector + keyword search
* No JVM dependency
* Simple local setup

### Retrieval

* Hybrid search:

  * Semantic (vector similarity)
  * Lexical (keyword match)
* Results merged and reranked

This reflects real-world RAG retrieval strategies.

---

## 10. Conversational Memory

* Simple in-memory Python list
* Stored per user ID
* Appended per interaction

Chosen for transparency and zero extra dependencies.

---

## 11. Architecture Philosophy

**Optimized for:**

* Clarity over cleverness
* Stability over scale
* Local reproducibility
* Interview explainability

**Intentionally avoided:**

* Over-engineering
* Framework lock-in
* Hardware-incompatible models
* Black-box abstractions

---

## 12. Final Outcome

* Fully working local RAG system
* Clean separation of concerns
* Minimal but sufficient tooling
* Clear reasoning from **PDF → Answer**

This project demonstrates **system design understanding**, not just library usage.
