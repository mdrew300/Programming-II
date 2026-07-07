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

## Running it locally

```bash
git clone https://github.com/<you>/bookmatch.git
cd bookmatch
pip install -r requirements.txt

# Gemini API key (required for the re-ranking layer)
export GEMINI_API_KEY="your-key"        # or add to .streamlit/secrets.toml

streamlit run app/goodreads_app.py
```

## Data

Uses the Goodreads `Books.csv` and `Ratings.csv`. [Add: where you got them /
Kaggle link.] Data files are not committed — download and place in `data/`.

## Evaluation design (Project 3)

Two system prompts (baseline vs. designed) run across 20 cases, each scored by two
judges in Braintrust: **Ranking Fit** (do the top-3 picks match the request?) and
**Explanation Quality** (are explanations specific and grounded?). Scale: 1 / 0.5 / 0.
See `slides/Project_3.pdf` for prompts, judges, traces, and the ship recommendation.

## Limitations

- CF retrieval sets a ceiling on ranking quality; the LLM can't recommend a good
  book that isn't in the candidate pool.
- Re-ranker grounds explanations partly on its own knowledge of well-known titles;
  metadata passed is title + author only.
- Evaluation is 20 cases, single run, same model family for generation and judging.
