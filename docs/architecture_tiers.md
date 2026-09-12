# Architecture Tiers: MVP vs Production vs Hyperscale

This document exists to prevent over-engineering. One student building alone
should target the **MVP** column.

> **These architectures are documented to demonstrate scalability awareness,
> not implemented in the current version.** The point is not "I don't know
> Kubernetes" — it's "Kubernetes is unnecessary for the current workload."
> The Production and Hyperscale columns are the *Architecture Evolution*
> path: what changes, and why, as real load and real users show up.

## MVP data flow (what's actually built right now)

```
User
  ↓
API (FastAPI)
  ↓
Query Processing (normalization)
  ↓
Embedding Model
  ↓
Qdrant (vector search) + BM25 (keyword search)
  ↓
Retriever / Reranker
  ↓
LLM (generation + grounding check)
  ↓
Response (+ citations)
```

## Architecture Evolution

| Concern | MVP (this project, now) | Production (real company, small team) | Hyperscale (millions of users) |
|---|---|---|---|
| Vector store | Single Qdrant instance, local Docker | Managed Qdrant Cloud / Pinecone, replicated | Sharded vector DB across regions |
| Embedding compute | On-demand, single machine (CPU/small GPU) | Batched embedding jobs, dedicated worker pool | Distributed embedding pipeline, GPU autoscaling |
| **LLM / Model Serving** | Single API/model + fallback model | Model gateway + multiple providers/models, routed by complexity/cost | Dedicated inference cluster + intelligent routing across regions |
| **AI Evaluation** | Small curated golden dataset + retrieval metrics (precision/recall/latency) | Automated evaluation pipeline running on every change (CI-gated) | Continuous evaluation + live model/quality monitoring + drift detection |
| Caching | In-memory dict / simple Redis | Redis cluster with TTL policies | Multi-tier cache (edge + regional + origin) |
| Auth | Simple API key | OAuth2 / JWT with role-based access | Full IAM, per-tenant isolation |
| Observability | Structured logs + local metrics file | Centralized logging (e.g. Grafana/Loki) | Full distributed tracing, SLOs, on-call rotation |
| Deployment | Docker Compose, single host | Kubernetes, CI/CD pipeline | Multi-region Kubernetes, blue/green deploys |
| Cost model | Pay-as-you-go, single developer budget | Monitored per-team budget, alerts | FinOps team, reserved capacity |
| Failure handling | Retries + basic fallback model | Circuit breakers, graceful degradation | Chaos engineering, automated failover |

**Rule for this project:** every component built now must be justified for the
MVP column — *simple + measurable + modular + Dockerized*. Production means
*scalable + observable + secure*. Hyperscale means *distributed + multi-region
+ highly available*. If a feature only makes sense in the Production or
Hyperscale column, it gets documented here as a future direction, not
implemented. The strongest interview answer is never "why didn't you use
Kubernetes" — it's "how do you know your AI system actually works," which is
exactly what the AI Evaluation row and `src/evaluation/metrics.py` answer.
