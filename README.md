# Multilingual Multihop QA with RAG: English & Urdu

This repository supports the thesis:

**"Enhancing Multihop Question Answering in Low-Resource Languages: A Retrieval-Augmented Generation Framework for Urdu"**

---

## 🧠 Project Overview

This work tackles multihop question answering (QA) in low-resource languages using Retrieval-Augmented Generation (RAG). While standard RAG pipelines handle single-hop queries effectively, they struggle with complex multihop queries, particularly in languages like Urdu with limited computational resources.

To address this, we introduce two improved architectures:

- **Layered Retrieval RAG (LR-RAG):** Sequential multistage retrieval and relevance filtering.
- **Dual-Query RAG (DQ-RAG):** Parallel retrieval from decomposed sub-queries and evidence merging.

Both systems are benchmarked against a **Simple RAG** pipeline in English and Urdu.

---

## 🚀 Key Features

- 🔎 Query Classification (Single-hop vs. Multihop) using LLMs.
- 🔄 Sub-query Decomposition for multihop queries.
- 📚 Semantic Retrieval using FAISS and embedding models.
- 🧪 Relevance Filtering using multilingual LLMs.
- 📝 Answer Generation based on curated context.
- 📊 Evaluation using:
  - NLP Metrics: BLEU, ROUGE, METEOR, SacreBLEU, BERTScore
  - RAGAS Metrics: Faithfulness, Relevance, Context Recall, Precision, Entity Recall
  - Inference Time Comparison

---

## 🧪 Experimental Setup

All experiments were conducted using:
- **NVIDIA GTX 1080 GPU (8GB VRAM)**
- **24 GB RAM**
- **Local inference with [Ollama](https://ollama.com)** for all LLMs

---

## 📊 Example Results (Urdu)

| Pipeline   | Faithfulness | BERT F1 | Total Time (s) |
|------------|--------------|---------|----------------|
| Simple RAG | **0.801**    | 0.819   | **6.93**       |
| LR-RAG     | 0.807        | **0.821** | 16.22        |
| DQ-RAG     | **0.810**    | 0.814   | 27.18          |

---

## 🔍 Models Used

- **Query Classification**: `gemma:7b-instruct` (English), `llama3:8b` (Urdu)
- **Embedding Models**:
  - English: `sentence-transformers/all-MiniLM-L6-v2`
  - Urdu: `intfloat/e5-large`
- **Language Models for Answering**: `llama2:latest` (English), `llama3.1:8b` (Urdu)
- **Vector Store**: FAISS with Flat2Index
- **Translation**: Meta’s `SeamlessM4T` for Urdu dataset generation

---

## 📂 Dataset

- Based on **HotpotQA** (90K+ English multihop queries)
- Urdu version: 598 translated QnAs (500 multihop) using `SeamlessM4T`
- Synthetic query classification sets (50 each for English and Urdu)

---

## 📌 Citation

If you use this project or parts of it, please cite:

```bibtex
@thesis{hammad2025multihoprag,
  author    = {Muhammad Hammad},
  title     = {Enhancing Multihop Question Answering in Low-Resource Languages: A Retrieval-Augmented Generation Framework for Urdu},
  year      = {2025},
  institution = {LUMS}
}

