# BookMatch — Collaborative-Filtering Book Recommender with LLM Re-Ranking

A Goodreads book recommender that pairs a collaborative-filtering engine with an
LLM re-ranking layer, then **evaluates whether the LLM layer actually helps** using
an LLM-as-a-Judge framework in Braintrust.

> Georgetown MSBA · OPAN 6604 (AI Modeling in Practice) · Team project
> Jaci Goode · Stephanie Ong · Matt Drew

---

## What it does

1. **Collaborative filtering** generates each user's top-N candidates from the
   Goodreads dataset (9,964 books · 164K ratings · 1,192 users · 98.5% sparse).
2. **LLM re-ranking** (Gemini 2.5 Flash Lite) reorders those candidates against a
   user's stated preference and writes an explanation for each pick — constrained
   to the candidate list only (hallucination guard).
3. **Streamlit app** shows CF vs. AI-personalized results side by side.
4. **Evaluation** (Project 3) tests the re-ranking prompts with two independent
   LLM judges to measure whether a designed prompt beats a baseline.

## Key results

| Model (CF selection) | RMSE | Precision@10 | Recall@10 |
|---|---|---|---|
| Baseline (mean) | 0.84 | 0.657 | 0.791 |
| **UBCF · Pearson · k=50** ✅ | 1.03 | **0.660** | **0.794** |
| IBCF · Cosine · k=50 | 0.86 | 0.637 | 0.768 |

**Evaluation finding (Project 3):** the designed prompt lifted Explanation Quality
(0.55 → 0.88) but Ranking Fit only modestly (0.35 → 0.43) — ranking quality is
capped by CF retrieval upstream, not by prompt wording.

## Tech stack

Python · Surprise · pandas · scikit-learn · Gemini API · Streamlit · Braintrust

## Repo structure
