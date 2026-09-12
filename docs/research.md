# Phase 2 — Research: Arabic Embeddings & Existing RAG Systems

This document exists so every architecture decision in this project traces
back to something real — not intuition. All figures below come from public
benchmarks/papers; sources are linked at the bottom of each section.

## 1. Arabic Embedding Model Landscape

**The core finding that justifies this entire project:** even the strongest
general-purpose multilingual embedding models score noticeably lower on
Arabic than on English. On one 2026 MTEB comparison, the best models scored
~72 on English but only ~64 on Arabic — a real, measurable gap, not a guess.

Dedicated Arabic benchmarks exist specifically because general MTEB
under-represents the language:

- **ArabicMTEB** — 94 datasets across 8 tasks, built specifically to cover
  Modern Standard Arabic *and* dialectal/cultural variation that generic
  benchmarks miss.
- **Swan embedding models** — Arabic-centric models that outperform
  Multilingual-E5 (both large and base sizes) on most Arabic tasks while
  costing significantly less to run.
- **GATE (Matryoshka-based Arabic embeddings)** — shows that
  Arabic-specific training + Matryoshka representation learning meaningfully
  outperforms base AraBERT and generic multilingual models on Arabic
  semantic similarity tasks.

**Practical shortlist for this project's embedding comparison arm:**

| Model | Why it's in the comparison |
|---|---|
| `BAAI/bge-m3` | MIT-licensed, self-hostable, strong multilingual retrieval scores, $0 cost |
| `intfloat/multilingual-e5-base` | Widely-used baseline, easy to self-host, our current default in `src/embeddings/base.py` |
| Arabic-specific fine-tuned model (e.g. an AraBERT/Matryoshka variant) | Tests whether Arabic-specific training actually beats generic multilingual models *on our own golden dataset* — this is the real question, not the leaderboard number |
| `text-embedding-3-small` (OpenAI) | API comparison arm — only used for the head-to-head experiment, not the default, to keep dev cost at $0 |

**Sources:** ArabicMTEB (arXiv:2411.01192), Swan embeddings (arXiv:2411.01192),
GATE Arabic embeddings (arXiv:2505.24581), MTEB 2026 multilingual comparison
(ailog.fr embedding benchmark report, Jan 2026).

## 2. Existing Arabic RAG Systems (Competitive Landscape)

| System | Domain | What they measured | Gap we can fill |
|---|---|---|---|
| Moroccan Family Code RAG (academic, 2025) | Legal (family law) | Built a custom 2.5k QA dataset, used BGE-m3, measured MRR/Recall@k/F1 + fidelity/relevance | Single-domain, not reusable for other Arabic content types |
| Algerian Tax Law RAG (academic, 2025) | Legal (tax) | Explicitly names poor digitization + dialectal variation as core blockers | Same domain-lock-in problem |
| QIAS 2026 Islamic Inheritance RAG | Legal/religious reasoning | Hybrid retrieval + cross-encoder reranking, scored 0.935 on a competition metric | Highly specialized reasoning pipeline, not a general evaluation platform |
| `derek-thomas/arabic-RAG` (Hugging Face Space) | General demo | No published retrieval/groundedness metrics found | Demo-quality, not evaluation-quality |
| Freelance-marketplace "custom RAG" builds (e.g. Khamsat listings) | Varies | No evaluation methodology published — delivery is "source code + a working link" | This is the exact failure mode SANAD exists to avoid |

**Key insight:** every rigorous Arabic RAG system that exists today is
**single-domain** (one specific legal area) and built as a one-off academic
or freelance project. None of them publish a **reusable, model-agnostic
evaluation harness** that lets you plug in different chunking strategies,
embedding models, and reranking configs and get real precision/recall/
groundedness numbers back. That reusable evaluation layer — not another
single-domain chatbot — is SANAD's actual contribution.

**Sources:** Moroccan Family Code RAG (TELKOMNIKA, Vol 23 No 6, Dec 2025),
Algerian Tax Law RAG (Univ. of Ghardaïa repository, 2025), QIAS 2026
Islamic Inheritance RAG (arXiv:2603.24012), Hugging Face Space
`derek-thomas/arabic-RAG`.

## 3. Decisions This Research Changes

1. **Embedding shortlist updated:** `BAAI/bge-m3` added as a primary
   open-source comparison arm alongside `multilingual-e5-base` — it has
   stronger published multilingual retrieval numbers and is still $0 to
   self-host.
2. **Confirmed the core thesis:** the "Arabic retrieval gap" is real and
   measurable in public benchmarks, not just a hunch — this directly
   supports the problem statement in the main README.
3. **Confirmed the originality angle:** existing rigorous Arabic RAG work is
   domain-locked (legal only) and doesn't ship a reusable evaluation
   harness — this is the gap SANAD targets.
4. **No change to MVP scope:** none of this research changes the Phase 1
   architecture — it validates the choices already made (BM25 + semantic
   hybrid retrieval, pluggable embeddings, evaluation-first design).

## References

1. Swan and ArabicMTEB: Dialect-Aware, Arabic-Centric, Cross-Lingual, and
   Cross-Cultural Embedding Models and Benchmarks —
   https://arxiv.org/abs/2411.01192
2. GATE: General Arabic Text Embedding for Enhanced Semantic Textual
   Similarity with Matryoshka Representation Learning and Hybrid Loss
   Training — https://arxiv.org/abs/2505.24581
3. Embedding Models 2026: Benchmark and Comparison (MTEB 2026 ranking,
   per-language breakdown including Arabic) —
   https://app.ailog.fr/en/blog/news/embedding-models-2026
4. MTEB Leaderboard 2026: Best Embedding Models for RAG —
   https://www.codesota.com/benchmarks/mteb
5. Hrimech, J., Mghari, M., Zaz, Y. — "Retrieval-augmented generation for
   Arabic legal information: the family code case study," TELKOMNIKA
   Telecommunication Computing Electronics and Control, Vol 23 No 6, Dec
   2025 — https://telkomnika.uad.ac.id/index.php/TELKOMNIKA/article/download/27400/12162
6. Hamani, W., Benyounes, Z. — "RAG-based Question-Answering System for
   Algerian Tax Law Context," Univ. of Ghardaïa repository, 2025 —
   https://dspace.univ-ghardaia.edu.dz/items/97ade049-079e-4cab-bc30-db6ca5d23994/full
7. Swaileh, W. et al. — "CVPD at QIAS 2026: RAG-Guided LLM Reasoning for
   Al-Mawarith Share Computation and Heir Allocation" —
   https://arxiv.org/abs/2603.24012
8. Hugging Face Space — `derek-thomas/arabic-RAG` —
   https://huggingface.co/spaces/derek-thomas/arabic-RAG

*All figures cited in this document (MTEB scores, dataset sizes, competition
results) are taken directly from the sources above — none are estimated.*
