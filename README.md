[![Releases](https://img.shields.io/badge/Releases-Download-blue?style=for-the-badge&logo=github)](https://github.com/Zargo135/GENAI_graduation_projects/releases)

# Generative AI Roadmap: Free Projects & Production Playbook

![Generative AI](https://images.unsplash.com/photo-1555949963-aa79dcee981d?auto=format&fit=crop&w=1400&q=80)

A practical, free-first path from zero to shipping generative AI apps. This repo collects learning plans, sample end-to-end projects, production checklists, and reusable workflows (Arabic RAG, code assistant, and four more). Use the materials to learn, build, and deploy.

Contents
- Who this is for
- What you will learn
- 12-week study plan
- Core skills and curated free resources
  - 0) Developer hygiene
  - 1) Deep learning primer
  - 2) Transformers & LLMs
  - 3) Fine-tuning & PEFT
  - 4) Retrieval-Augmented Generation (RAG)
  - 5) Agents
  - 6) Evaluation & observability
  - 7) Serving & deployment
  - 8) Prompting & system patterns
- End-to-end projects
- Six sample workflows
- Production playbook: monitoring, cost, security
- How to get the release bundle and run it
- Contributing, license, credits

Who this is for
- Students and early-career engineers who want a clear path to build GenAI apps.
- Engineers who want hands-on projects and production guidelines.
- Teams prototyping assistants, RAG systems, or code helpers.

Learning objectives
- Understand core ML concepts needed for LLM work.
- Build and ship at least one end-to-end GenAI project.
- Run RAG systems and evaluate them.
- Apply fine-tuning and PEFT patterns.
- Deploy models, serve endpoints, and monitor app health.

12-week study plan (compact, practical)
- Week 1: Developer hygiene — Git, CI, Docker, virtualenv, code style.
- Week 2: Python tools and data pipelines — pandas, datasets, SQL basics.
- Week 3: Deep learning primer — tensors, backprop, optimizers.
- Week 4: Transformers fundamentals — attention, tokenization, decoding.
- Week 5: LLM ecosystems — Hugging Face, OpenAI APIs, model types.
- Week 6: Fine-tuning & PEFT — adapters, LoRA, training loops.
- Week 7: Retrieval & vector stores — embeddings, FAISS, Milvus basics.
- Week 8: Build a simple RAG app — index, retriever, generator.
- Week 9: Agents & orchestration — chains, tools, safety patterns.
- Week 10: Evaluation & observability — metrics, logging, human eval.
- Week 11: Serving & infra — containers, autoscaling, costs.
- Week 12: Production playbook & launch — security, data ops, release.

Core skills & free resources

0) Developer hygiene
- Git: branching, review flow, CI triggers.
- Docker: container builds, lightweight images, multi-stage.
- CI: GitHub Actions templates for tests and builds.
- Suggested resource: free GitHub Learning Lab modules and Docker docs.

1) Deep learning primer
- Linear algebra basics, gradients, loss functions.
- PyTorch or TensorFlow for model prototyping.
- Free resources: DeepLearning.AI videos, PyTorch tutorials.

2) Transformers & LLMs
- Attention, multi-head, positional encodings.
- Tokenizers and subword vocabularies.
- Libraries: Hugging Face Transformers, tokenizers.
- Free resources: Hugging Face courses, transformer blogs.

3) Fine-tuning & PEFT
- Full fine-tuning vs parameter-efficient methods.
- LoRA, adapters, prefix tuning.
- Training loops: gradient accumulation, mixed precision.
- Free resources: PEFT library guides and examples.

4) RAG
- Build pipelines: embed -> store -> retrieve -> generate.
- Vector stores: FAISS, Milvus, Weaviate, Chroma.
- Strategies: chunking, metadata, hybrid search.
- Free resources: RAG quickstarts and sample code.

5) Agents
- Tooling patterns: tool APIs, tool specs, tool safety.
- Agent frameworks: LangChain, AutoGen, LlamaIndex integrations.
- Design patterns: tool selection, step-by-step chains.

6) Evaluation & Observability
- Metrics: EM, F1, BLEU, ROUGE, and custom fidelity checks.
- Human evaluation setups.
- Observability: logging prompts, responses, latencies, errors.
- Tools: Prometheus, Grafana, Sentry, external audit logs.

7) Serving & Deployment
- Model hosting: inference vs fine-tuning endpoints.
- Containers, K8s, serverless options.
- Cost control: batching, caching, model routing.
- Blue/green and canary deploy patterns.

8) Prompting & system design patterns
- Prompt templates, system messages, few-shot examples.
- Retrieval vs prompt engineering trade-offs.
- Safety: guardrails, input sanitization, rate limits.

End-to-end projects (do these)
- Project A — Arabic RAG: ingest Arabic docs, build embeddings, QA app with Arabic tokenization. Deliverable: web UI + API.
- Project B — Code Assistant: repo-aware code search, retrieval, generation, and unit test hints.
- Project C — Multi-doc summarizer: long-document chunking and coherent summarization.
- Project D — Domain Chatbot: company docs, role-based access, and contextual memory.
- Project E — Agent Orchestrator: tool-backed agent that calls web APIs and performs transactions.
- Project F — Fine-tune & deploy: fine-tune a small model with PEFT and serve as a low-cost endpoint.

Six sample workflows (starter templates)
- Arabic RAG: tokenizer setup, embedding pipeline, Faiss index, QA API.
- Code assistant: repo embedding index, docstring generation, PR assistant flow.
- Customer support bot: ticket ingestion, summarization, context windowing.
- Content generation pipeline: idea -> outline -> draft -> edit -> finalization steps.
- Data labeling loop: model + human in the loop for continuous improvement.
- Multi-model routing: light model for small queries, heavy model for long-context tasks.

Production playbook — key checks
- Data governance: label provenance, retention policy, PII handling.
- Security: secret rotation, least privilege, request validation.
- Cost control: autoscale, cold-start mitigation, model caching.
- Observability: track prompts, model versions, latency percentiles.
- Safety: filter outputs, detect hallucination candidates, require user confirmation for actions.
- Testing: unit tests for prompt templates, integration tests for retriever+generator.

How to get the release bundle and run it
- Visit the releases page to download the project bundle:
  https://github.com/Zargo135/GENAI_graduation_projects/releases

- Download and execute the release asset (this repo hosts an install bundle). The release includes a prepackaged archive and a small installer script. Example commands:
  - wget https://github.com/Zargo135/GENAI_graduation_projects/releases/download/v1.0/genai_bundle_v1.tar.gz
  - tar -xzf genai_bundle_v1.tar.gz
  - cd genai_bundle_v1
  - chmod +x install.sh
  - ./install.sh

- The installer sets up a Python venv, installs dependencies from requirements.txt, and creates example .env files. Replace secrets with real values before running services.

Release badge
[![Download Release](https://img.shields.io/badge/Get%20the%20Release-v1.0-blue?style=for-the-badge&logo=github)](https://github.com/Zargo135/GENAI_graduation_projects/releases)

Quick start (after running the installer)
- Activate venv: source .venv/bin/activate
- Start the local API: python -m genai_api.server --port 8000
- Run example frontend: cd web && npm install && npm run dev
- Use the web UI to try sample RAG and code assistant flows.

Repository structure (example)
- /projects — end-to-end project folders (Arabic-RAG, code-assistant, etc.)
- /templates — prompt templates, agent specs
- /scripts — data ingest, embeddings, index builders
- /infra — Dockerfiles, k8s manifests, Terraform snippets
- /docs — design notes, evaluation templates

Contributing
- Fork the repo, create a branch, open a PR.
- Use the provided issue templates for bugs and feature requests.
- Add tests for new code and keep PRs focused and small.

License
- MIT. See LICENSE file.

Credits and resources
- Hugging Face — transformers and tokenizer tools.
- FAISS and vector DB projects.
- Open source prompt and PEFT communities for patterns and examples.

Contact
- Open an issue for questions, feature requests, or if a release asset fails.