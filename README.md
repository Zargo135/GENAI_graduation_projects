# Generative AI Roadmap — Free Resources, E2E Projects, and Production Playbook

A practical, **free-first** roadmap to go from zero → shipping real GenAI apps. Includes hands-on projects, sample workflows (Arabic RAG, code assistant, and 4 more), and production tips.

---

## Table of Contents

- [Who This Is For](#who-this-is-for)
- [Learning Objectives](#learning-objectives)
- [12-Week Study Plan](#12-week-study-plan)
- [Core Skills & Free Resources](#core-skills--free-resources)
  - [0) Developer Hygiene](#0-developer-hygiene)
  - [1) Deep Learning Primer](#1-deep-learning-primer)
  - [2) Transformers & LLMs](#2-transformers--llms)
  - [3) Fine-Tuning & PEFT](#3-fine-tuning--peft)
  - [4) RAG](#4-rag)
  - [5) Agents](#5-agents)
  - [6) Evaluation & Observability](#6-evaluation--observability)
  - [7) Serving & Deployment](#7-serving--deployment)
  - [8) Prompting & System Design Patterns](#8-prompting--system-design-patterns)
- [End-to-End Projects (Do These)](#end-to-end-projects-do-these)
- [Six Sample Workflows (Copy-Paste Blueprints)](#six-sample-workflows-copy-paste-blueprints)
- [Ten Graduation‑Ready Project Ideas](#ten-graduation-ready-project-ideas)
- [Production Tips & Checklists](#production-tips--checklists)
- [Quick-Start Snippets](#quick-start-snippets)
- [Project Template](#project-template)
- [FAQ](#faq)
- [License](#license)

---

## Who This Is For

- Newcomers to GenAI who want to **ship** something useful fast.
- Engineers moving from classic ML/DL to **RAG, agents, and LLM serving**.
- Builders who want **free** resources and **production-grade** patterns.

## Learning Objectives

- Understand transformers/LLMs enough to **use, evaluate, and serve** them.
- Build **RAG** and **agent** systems with structured outputs and guardrails.
- Stand up **observability/evals** and ship with **cost/latency** discipline.

---

## 12-Week Study Plan

**Weeks 1–2:** Python + Git + Docker + basic data wrangling; fast-track DL intuition.

**Weeks 3–4:** Transformers & LLM fundamentals with hands-on mini-labs.

**Week 5:** RAG MVP + faithfulness evals.

**Week 6:** Workflow automation (e.g., email triage) + prompt testing.

**Week 7:** Tracing & observability across apps; cost dashboards.

**Week 8:** Agentic patterns (bounded tools, timeouts, guardrails).

**Week 9:** Multi-doc RAG (hybrid retrieval + rerankers) with citations.

**Week 10:** Serve with a high-throughput engine; measure p50/p95 latency.

**Week 11:** Quantize & benchmark; or push NVIDIA stack for peak perf.

**Week 12:** Hardening: CI gates on eval metrics + red-team suites.

---

## Core Skills & Free Resources

### 0) Developer Hygiene

- Python (virtualenv/uv, packaging), Git (branches/PRs), CLI basics.
- Docker fundamentals, docker-compose for local stacks.
- NumPy/Pandas for quick data handling.

### 1) Deep Learning Primer

- **fast.ai – Practical DL for Coders** (free). Hands-on intuition for modern DL.

### 2) Transformers & LLMs

- **Hugging Face – Transformers/LLM Course**: tokenizers, datasets, accelerate, inference, fine-tuning.
- Read **model cards** thoroughly (safety, license, context length, training data notes).

### 3) Fine-Tuning & PEFT

- LoRA/adapters with Hugging Face. Prefer PEFT for speed/cost; use RAG to inject fresh knowledge.

### 4) RAG

- Start local: **FAISS** index; swap to a vector DB later (e.g., **Weaviate**).
- Frameworks: **Haystack** pipelines or DIY: chunk → embed → index → retrieve → rerank → generate.
- Log everything: input, retrieved docs, prompts, latencies, token counts.

### 5) Agents

- Try two approaches to feel trade-offs: **AutoGen**, **CrewAI**.
- Or **DSPy** to “program” LLMs instead of hand-prompting.
- Always bound tools, retries, time, and budget.

### 6) Evaluation & Observability

- **Ragas** (RAG faithfulness, relevancy), **TruLens** (feedback functions), **promptfoo** (prompt tests & red-team in CI).
- **Langfuse** or similar for tracing, spans, costs, and dataset runs.

### 7) Serving & Deployment

- Local/dev: **Ollama**, **llama.cpp** (GGUF CPU-friendly).
- Production GPU: **vLLM** (continuous batching, KV cache), **TensorRT-LLM** (+ Triton) for peak NVIDIA perf.
- Quantization: **AWQ** often a strong quality/speed trade-off for 7B–13B.

### 8) Prompting & System Design Patterns

- Favor **tool/function calling** over long instructions.
- Constrain outputs with **JSON Schemas**; validate before acting.
- Keep context short; use rerankers before you add more chunks.
- Cache embeddings & generations; version prompts.

---

## End-to-End Projects (Do These)

### P1) Docs Q&A RAG MVP (weekend)

**Goal:** Ask questions over a folder of PDFs.

**Stack:** FAISS + minimal RAG pipeline (Haystack optional); local model via Ollama or server via vLLM.

**Ship Criteria:** p50 < 2s, faithfulness gate passes, citations included, clean traces.

---

### P2) Email Triage Copilot (1 week)

**Goal:** Classify/route and draft answers with action suggestions.

**Stack:** Prompt tests (promptfoo), simple tools (calendar/CRM mocks), JSON schema outputs.

**Ship Criteria:** ≥95% schema-valid JSON; red-team suite passes; p50 < 3s.

---

### P3) Multi-Doc RAG + Summarizer (1–2 weeks)

**Goal:** Summarize and answer across PDFs/HTML/CSV.

**Stack:** Hybrid retrieval (BM25 + dense), rerankers, citations; Ragas + Langfuse.

**Ship Criteria:** Relevancy/faithfulness above thresholds; clear source attributions.

---

### P4) Task-Doer Agent (1 week)

**Goal:** One agent using 2–3 tools (web, calendar, internal API).

**Stack:** AutoGen or CrewAI; tight timeouts/tool budgets; extensive traces/evals.

**Ship Criteria:** ≥90% success on scripted tasks; no unsafe actions.

---

### P5) High-Throughput Serving + Quantization (2–3 days)

**Goal:** Serve a 7B–13B instruct model at low cost.

**Stack:** vLLM + AWQ (or TensorRT-LLM for peak perf). Measure before/after.

**Ship Criteria:** ≥2× throughput, p95 < SLA, negligible quality loss.

---

## Six Sample Workflows (Copy-Paste Blueprints)

### W1) Arabic RAG for Customer Support

**Use-case:** Arabic/Arabizi/English user queries → precise answers from your help center + tickets.

**Architecture (text diagram):**

```
User → API (validate + rate-limit)
     → Query Preprocess (normalize Arabic, remove diacritics, unify Hamza)
     → Retriever (hybrid: BM25 + dense multilingual embeddings)
     → Reranker (cross-encoder)
     → Generator (instruct model; system prompt enforces Arabic answer + citations)
     → Postprocess (cite sources, sanitize links)
     → Logging (Langfuse) + Evals (Ragas)
```

**Recommended Stack:**

- Embeddings: `intfloat/multilingual-e5-large` or `bge-m3`
- Reranker: `bge-reranker-large`
- Vector: FAISS → Weaviate later
- Serving: vLLM (GPU) or Ollama (local)

**Metrics & Gates:**

- Faithfulness ≥ 0.75, Answer relevancy ≥ 0.8, Toxicity = 0
- Latency p50 < 2.5s, Cost/1000 queries ≤ budget

**Special Arabic Notes:**

- Normalize forms (أ/ا/إ), Tatweel removal, optional Buckwalter transliteration for debugging.
- Add synonyms/typos dictionary for brand/product names.

---

### W2) Code Assistant for Your Repos

**Use-case:** Ask “Where is the login rate-limit?” or “Write a safe wrapper for S3 uploads.”

**Architecture:**

```
User → API (auth + rate-limit) → Repo Ingest (git clone, commits)
→ Parser (tree-sitter; build symbol table) → Chunk by function/class
→ Hybrid Retrieval (BM25 + code embeddings) → Reranker
→ Generator (structured JSON: {changes, rationale, risk}) → Optional tool: create PR
→ Tracing + Evals (unit tests, promptfoo suites)
```

**Stack:**

- Code embeddings: `jinaai/jina-embeddings-v2-base-code` (or similar)
- Optional static analysis: Semgrep rules for red flags
- PR bot: GitHub App + checks that fail on schema-invalid JSON

**Gates:** schema-valid ≥ 98%, unit tests pass, no secret leaks.

---

### W3) Multilingual FAQ Bot (Text + Voice)

**Use-case:** Users ask in any language/voice; bot answers + cites.

**Add-ons:**

- ASR: Whisper (open-source)
- TTS: Coqui TTS or Piper

**Architecture:**

```
Audio/Text → ASR (if audio) → Normalization → Hybrid Retrieval
→ Generator (system prompt: short, friendly, cite sources)
→ Output (text + optional TTS) → Traces/Evals
```

**Notes:**

- Cache TTS for repeated answers; guard offensive queries; language-auto-detect.

---

### W4) Meeting Minutes & Action-Item Agent

**Use-case:** Upload meeting audio; get minutes, owners, deadlines, and a follow-up email.

**Architecture:**

```
Audio → ASR → Segment + Topic clustering → Summarizer
→ Action Extractor (JSON schema) → Calendar/Task API tools
→ Email Draft (structured) → Human confirm → Send
```

**Gates:** JSON schema validity ≥ 98%, hallucination checks vs transcript.

---

### W5) E‑commerce Semantic Search + Product Q&A

**Use-case:** “Black running shoes under \$80, wide feet” → ranked results + explanations.

**Architecture:**

```
User query → Query Understanding (parse facets) → Hybrid Search
→ Reranker (cross-encoder) → (Optional) Generator for short rationale
→ Metrics Logging (CTR, add-to-cart uplift) → A/B test
```

**Notes:**

- Keep generation optional; search quality should not depend on LLM availability.

---

### W6) Document Intake → Auto-Classification → Compliance Summary

**Use-case:** Intake PDFs (contracts/reports), auto-classify, detect PII, summarize risks.

**Architecture:**

```
Upload → OCR (if needed) → Chunk → Classifier (few-shot or small finetune)
→ PII Redaction (regex + NER) → RAG for policy mapping → Summary JSON
→ Reviewer UI → Export (PDF/JSON) → Traces/Evals
```

**Gates:** Zero PII leakage; faithfulness to policy docs ≥ threshold.

---

## Ten Graduation‑Ready Project Ideas

> Each idea includes **novelty**, **public datasets**, a suggested **stack**, **eval metrics**, and **deliverables**. All are scoped for a 3–4 month capstone.

### G1) Arabic Multi‑Dialect Support Copilot (RAG + Dialect ID)

**Why/Novelty:** Real customer support handles MSA + 20+ dialects + Arabizi. Build a dialect‑aware RAG that routes queries and normalizes variants. **Datasets:** MADAR (dialects), your help‑center docs; optional in‑house FAQs. **Stack:** Dialect ID (CAMeL models) → normalization → hybrid retrieval (BM25 + multilingual embeddings) → reranker → generator (Arabic‑centric prompt). **Eval:** Faithfulness, answer relevancy, exact citations; latency p50; dialect routing accuracy. **Deliverables:** API + small admin UI; report comparing with non‑dialect baseline.

### G2) Legal Clause Reviewer (Contracts RAG + Classifier)

**Why/Novelty:** Combine clause classification with RAG explanations to flag risky clauses. **Datasets:** CUAD (contracts & labels) + your template library. **Stack:** Clause classifier (PEFT) + RAG over playbooks/templates; JSON risk report. **Eval:** F1 per clause category; faithfulness of explanations; reviewer time saved. **Deliverables:** Web app with upload → review → export (PDF/JSON) and citations.

### G3) Meeting Minutes & Action‑Item Agent (ASR → Structuring)

**Why/Novelty:** High‑precision extraction of owners/deadlines and email handoff. **Datasets:** AMI Meeting Corpus (train/benchmark) + your recorded meetings (opt‑in). **Stack:** Whisper (ASR) → segment/clustering → summarizer → action extractor (JSON schema) → calendar/email tools. **Eval:** Schema validity, action extraction F1, faithfulness vs transcript. **Deliverables:** Upload UI + ICS export + audit log.

### G4) Hybrid Table‑Text Financial QA

**Why/Novelty:** Retrieval/reasoning over **tables + text** (real financial reports). **Datasets:** TAT‑QA (tabular+text QA). **Stack:** Dual retrieval (BM25 + dense) for tables and text → program-of-thought or DSPy for numerical reasoning → cite cells/paragraphs. **Eval:** EM/F1 on TAT‑QA; numerical reasoning accuracy; latency. **Deliverables:** API + demo UI visualizing the supporting rows/paragraphs.

### G5) Open‑Domain Table+Text QA over Wikipedia

**Why/Novelty:** At‑scale retrieval over tables and passages with robust evals. **Datasets:** OTT‑QA (open table+text QA). **Stack:** Index Wikipedia tables + passages; retriever‑reranker; answer with source highlighting. **Eval:** EM/F1 on OTT‑QA; retrieval recall\@k; citation accuracy. **Deliverables:** Leaderboard‑style report; ablations (reranker/no‑reranker, context size).

### G6) Document Layout → Structured JSON (DocAI)

**Why/Novelty:** General document parsing beyond research papers (invoices, forms, brochures). **Datasets:** DocLayNet (diverse pages) + PubLayNet (scientific) for transfer. **Stack:** Layout detection (DiT/YOLO‑based) → key‑value extraction → validation rules. **Eval:** mAP for layout; field‑level F1; robustness across domains. **Deliverables:** Demo that converts random PDFs to JSON/HTML with bounding‑box overlays.

### G7) E‑commerce Semantic Search + Product Q&A

**Why/Novelty:** Hybrid search with LLM rationales; works even without generation. **Datasets:** Amazon Reviews/metadata (McAuley Lab) as proxy; your catalog if available. **Stack:** Query parsing → hybrid retrieval → reranker → optional generator for rationale. **Eval:** NDCG\@k / MRR; CTR uplift (simulated); latency/cost. **Deliverables:** Search UI with filters and “why this result?” explanations.

### G8) Multilingual Speech Q&A (ASR/ST → RAG)

**Why/Novelty:** Voice queries in various languages → translated → answered with citations. **Datasets:** CoVoST 2 for ST; your domain docs for RAG. **Stack:** ASR/ST → normalization → retrieval → generator; output in user language. **Eval:** BLEU/WER on ST; faithfulness for answers; latency. **Deliverables:** Web demo (mic input) + transcripts + sources.

### G9) Web Task Agent in a Sandbox (Tool‑Use + Safety)

**Why/Novelty:** Reliable, **bounded** web agents that solve multi‑step tasks safely. **Datasets/Env:** WebArena (self‑hostable websites + benchmark tasks). **Stack:** Agent framework (AutoGen/CrewAI) + tool calling; validators; strict budgets. **Eval:** Task success rate on WebArena; timeouts; unsafe‑action rate. **Deliverables:** Reproducible benchmark runs + traces.

### G10) Egypt Open‑Data Q&A Portal (Arabic/English)

**Why/Novelty:** Public service portal over national open datasets with **cited answers**. **Datasets:** Egypt Data Portal (and similar public datasets/APIs). **Stack:** Data ingestion/ETL → semantic search (hybrid) → generator with citations; Arabic normalization. **Eval:** Answer faithfulness, citation accuracy, latency; user study. **Deliverables:** Simple portal + API + report on data coverage and gaps.

> **Graduation rubric (use this to impress supervisors):**
>
> 1. Clear problem + users, 2) Public datasets & baseline, 3) Strong evals (not demos), 4) Observability, 5) Ablations & error analysis, 6) Deployment + README with one‑click run.

---

## Production Tips & Checklists

**Latency/Throughput**

- Use continuous batching & KV cache; keep contexts short; prefer rerankers to many chunks.
- Quantize when compute-bound (AWQ is a good start for 7B–13B); benchmark before/after.

**Reliability & Safety**

- Enforce JSON schemas; validate and reject before side effects.
- Add rate limits, abuse filters, and jailbreak tests.

**Observability & Evals**

- Trace every step (inputs, retrieved docs, latencies, costs); version prompts.
- Gate deploys on **faithfulness**, **relevancy**, **toxicity**, and **schema-valid** metrics.

**Data Hygiene**

- Chunk semantically, not by raw pages; deduplicate; store doc IDs for citations.
- Build a small, high-quality golden dataset for evals early.

**Cost Discipline**

- Cache embeddings & generations; route to smaller models by default.

---

## Quick-Start Snippets

### Minimal FAISS Index (Python)

```python
from sentence_transformers import SentenceTransformer
from datasets import Dataset
import faiss, numpy as np

texts = ["Reset password steps", "Shipping policy", "Arabic: سياسة الاستبدال"]
model = SentenceTransformer("intfloat/multilingual-e5-large")
emb = model.encode(texts, normalize_embeddings=True)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(np.array(emb, dtype=np.float32))

q = model.encode(["ازاي اغير الباسورد؟"], normalize_embeddings=True)
dist, idx = index.search(np.array(q, dtype=np.float32), 5)
print([texts[i] for i in idx[0]])
```

### Ragas Eval (Faithfulness/Relevancy)

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
print(result)
```

### promptfoo Test (prompt.yml)

```yaml
providers:
  - id: openai:gpt-4o-mini   # or any OpenAI-compatible endpoint (vLLM)

prompts:
  - role: system
    content: |
      You are a helpful assistant. Output STRICT JSON matching the schema.
  - role: user
    content: "Classify this email and propose an action: {{email_text}}"

tests:
  - vars:
      email_text: "Refund request: order #1234"
    assert:
      - type: contains-json
      - type: is-json-schema
        value:
          type: object
          properties:
            label: { type: string }
            action: { type: string }
          required: [label, action]
```

### Langfuse Tracing (Python)

```python
from langfuse import Langfuse

lf = Langfuse()
trace = lf.trace(name="ask", input=payload)
span = trace.span(name="retrieve", input=query)
# ... do work ...
span.end(output={"k": 10})
trace.end(output=answer)
```

### vLLM Quick Serve (OpenAI-compatible API)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen2.5-7B-Instruct \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --port 8000
```

---

## Project Template

```
.
├── apps/
│   ├── api/                 # FastAPI (auth, rate-limits, JSON schemas)
│   ├── worker/              # ingestion, indexing, async jobs
│   └── web/                 # small UI (Next.js/Vite)
├── data/                    # raw docs, embeddings cache, indices
├── evals/                   # ragas datasets, promptfoo suites
├── prompts/                 # versioned prompts (semver tags)
├── scripts/                 # indexer, bench, export
├── docker-compose.yml
├── .env.example
└── README.md
```

**.env.example**

```
OPENAI_API_BASE=http://localhost:8000/v1   # vLLM or compatible
OPENAI_API_KEY=sk-local-demo
VECTOR_DB=faiss
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk_xxx
LANGFUSE_SECRET_KEY=sk_xxx
```

---

## FAQ

**Q:** Do I need full fine-tuning?

**A:** Usually no. Start with good prompts + RAG; use PEFT if you must adapt behavior.

**Q:** Which model do I start with?

**A:** Small, instruction-tuned 7B–13B for prototypes; scale only if evals say so.

**Q:** How do I keep costs sane?

**A:** Batch, cache, quantize, keep contexts short, and route to smaller models by default.

---

## License

This README is released under **CC BY 4.0**. Build cool stuff, and share your learnings.

