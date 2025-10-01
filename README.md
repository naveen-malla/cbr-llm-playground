# CBR Lab (Local) — CBR + Local LLM (Ollama)

Learn Case-Based Reasoning (Retrieve–Reuse–Revise–Retain) with a local LLM for adaptation/explanations.

## Quickstart
1) Install Ollama: https://ollama.com  
2) Pull a small model (free): `ollama pull phi3:mini`  
   (You can also try `mistral:7b` if you want bigger.)  
3) Setup:
```bash
make setup
cp .env.example .env
# optional: edit OLLAMA_MODEL in .env
```

4. Run:

```bash
make run-app
```

## What you’ll see

* **Inspect Cases**: browse toy cases, predicate view, tiny graph view.
* **Retrieve Similar**: k-NN with **embeddings** (Sentence-Transformers) or **TF-IDF fallback**.
* **LLM Adaptation**: compare rule-based vs LLM-assisted adaptation via **Ollama**.
* **Hybrid RAG**: fuse symbolic similarity and vector similarity (α slider).

Offline-friendly: If sentence-transformers is not downloaded, we **fallback to TF-IDF** so it still works. LLM runs fully local via Ollama.

## 2025 relevance (super short)

* Transparent **CBR similarity** + **LLM adaptation/explanation** = practical hybrid reasoning.
* Hybrid symbolic + vector retrieval mirrors modern **RAG** patterns, but with clearer logic.

