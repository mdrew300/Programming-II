# -*- coding: utf-8 -*-
"""
"""Model comparison and EDA for the Goodreads recommender.

Produces the CF results reported in README.md. The deployed app is goodreads_app.py.
"""
### Key Requirements
1. Explore the dataset and provide any insights you find about users, reviews, and the books. This is an important step as unsupervised learning projects are exploratory by nature. Highlight patterns, anomalies, or questions the data raises that could affect modeling or recommendations.
2. Compare and evaluate a set of recommendation models based on user-based and item-based collaborative filtering (in Python, using surprise). Include a simple popularity/mean baseline as a benchmark. Evaluate performance using RMSE for rating prediction and Precision@K / Recall@K for Top-N recommendations. Which model performs better and why? If your collaborative-filtering models do not outperform the baseline, discuss why (e.g., data sparsity, the baseline being a strong "popularity" recommender) and what that implies for model selection.
3. Add an AI re-ranking layer on top of your recommender (Refer to Week 4 materials). Take the Top-N candidates from your best collaborative-filtering model and use an LLM to re-rank and personalize them to a user’s stated preference (e.g., a mood, favorite genre, or other preferences), returning a short explanation for each pick. The LLM should re-rank the candidate recommendations produced by your collaborative-filtering model rather than generate entirely new recommendations using the book metadata (e.g., author, title, year, average rating, ...) as context. You may experiment with different prompt styles or personalization strategies. You may use any LLM (cloud or local) for this layer - Gemini is recommended because it's free (see the API-key guide) - but cite the model and provider you used, and never include your key in the submission.
4. Build a Streamlit app that wraps your recommender. (Refer to Week 3 and 4 demos and class materials and the Guides section below on how to build and run a Streamlit app). The app should let a user be selected, show the collaborative-filtering Top-N, and offer the LLM re-ranking.

5. Record a short (~2–3 minute) video demo walking through your app and briefly explaining how it works - the CF step, the LLM step, and one design choice you made. Do not include your API key in your code or submission.
Business discussion. What are the business applications of each model (collaborative filtering and the LLM layer) and what challenges might a business face when setting these up? What would be your recommended approach for this dataset?
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict
from surprise import KNNBasic, BaselineOnly, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

from pydantic import BaseModel, Field
from google import genai
import getpass


books = pd.read_csv("Books.csv")
ratings = pd.read_csv("Ratings.csv")

"""# Data Dictionary
---
---

## Books dataset:
---

book_id: unique book ID

isbn: the International Standard Book Number (10-digit)

authors: book authors, separated by comma

original_publication_year: the year the book was published

title: main title of the book (also indicates whether it is part of a series)

language_code: language of the edition

average_rating: average rating submitted by reviewers

ratings_count: number of submitted ratings

text_reviews_count: number of submitted text reviews

ratings_1 … ratings_5: number of 1- through 5-star reviews

image_url / small_image_url: links to the book image


## Rating dataset:
---

book_id: the book being rated

user_id: the user who submitted the rating

rating: the user's rating, from 1 to 5

# EDA
---
---
"""

books.head()

ratings.head()

print(books.info())

print(ratings.info())

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)

# Create subplots for books numeric columns
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
fig.suptitle('Books Dataset - Numeric Columns Distribution', fontsize=16, fontweight='bold', y=1.00)

# Books columns to visualize
books_cols = ['original_publication_year', 'average_rating', 'ratings_count',
              'text_reviews_count', 'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5']

# Flatten axes for easier iteration
axes_flat = axes.flatten()

for idx, col in enumerate(books_cols):
    ax = axes_flat[idx]

    # Use log scale for count columns to better see the distribution
    if 'count' in col.lower():
        data = books[col][books[col] > 0]  # Remove zeros for log scale
        ax.hist(data, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xscale('log')
        ax.set_title(f'{col} (log scale)', fontweight='bold')
    else:
        ax.hist(books[col], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_title(f'{col}', fontweight='bold')

    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary statistics for books numeric columns
print("\n" + "="*80)
print("BOOKS DATASET - SUMMARY STATISTICS")
print("="*80)
print(books[books_cols].describe())

# Visualize ratings distribution from ratings dataset
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Ratings Dataset - Distribution Analysis', fontsize=16, fontweight='bold')

# 1. Distribution of individual ratings
ax1 = axes[0]
rating_counts = ratings['rating'].value_counts().sort_index()
colors = ['#d62728', '#ff7f0e', '#ffdd57', '#2ca02c', '#1f77b4']
ax1.bar(rating_counts.index, rating_counts.values, color=colors, edgecolor='black', alpha=0.8, width=0.6)
ax1.set_xlabel('Rating (1-5)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
ax1.set_title('Distribution of Individual User Ratings', fontsize=13, fontweight='bold')
ax1.set_xticks([1, 2, 3, 4, 5])
ax1.grid(True, alpha=0.3, axis='y')

# Add count labels on bars
for i, v in enumerate(rating_counts.values):
    ax1.text(rating_counts.index[i], v + max(rating_counts.values)*0.01, str(v),
             ha='center', va='bottom', fontweight='bold')

# 2. Distribution of average user ratings (per user)
ax2 = axes[1]
user_avg_ratings = ratings.groupby('user_id')['rating'].mean()
ax2.hist(user_avg_ratings, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax2.set_xlabel('Average Rating per User', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Users', fontsize=12, fontweight='bold')
ax2.set_title('Distribution of Average User Ratings', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axvline(user_avg_ratings.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {user_avg_ratings.mean():.2f}')
ax2.axvline(user_avg_ratings.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {user_avg_ratings.median():.2f}')
ax2.legend()

plt.tight_layout()
plt.show()

# Print ratings summary statistics
print("\n" + "="*80)
print("RATINGS DATASET - SUMMARY STATISTICS")
print("="*80)
print(f"Total Ratings: {len(ratings)}")
print(f"Unique Users: {ratings['user_id'].nunique()}")
print(f"Unique Books: {ratings['book_id'].nunique()}")
print(f"\nRating Distribution (counts):")
print(rating_counts)
print(f"\nAverage Rating Statistics (per user):")
print(user_avg_ratings.describe())
print(f"\nSparsity: {1 - (len(ratings) / (ratings['user_id'].nunique() * ratings['book_id'].nunique())): .4%}")

"""## Top 100 Books
---
"""

top100 = books.sort_values("average_rating", ascending=False).head(n=100)

top100

# Overall rating distribution from the Books dataset
books["average_rating"].describe()

plt.figure(figsize=(10, 6))
sns.histplot(books["average_rating"], bins=30, kde=True)

plt.title("Distribution of Average Book Ratings")
plt.xlabel("Average Rating")
plt.ylabel("Number of Books")
plt.show()

# Average rating by publication year
rating_by_year = (
    books
    .dropna(subset=["original_publication_year"])
    .groupby("original_publication_year")
    .agg(
        avg_rating=("average_rating", "mean"),
        book_count=("book_id", "count")
    )
    .reset_index()
)

# Filter to years with enough books so the trend is not driven by tiny samples
rating_by_year_filtered = rating_by_year[rating_by_year["book_count"] >= 10]

plt.figure(figsize=(12, 6))
sns.lineplot(
    data=rating_by_year_filtered,
    x="original_publication_year",
    y="avg_rating"
)

plt.title("Average Rating by Publication Year")
plt.xlabel("Original Publication Year")
plt.ylabel("Average Rating")
plt.show()

# Average rating by author
author_summary = (
    books
    .groupby("authors")
    .agg(
        avg_rating=("average_rating", "mean"),
        book_count=("book_id", "count"),
        total_ratings=("ratings_count", "sum")
    )
    .reset_index()
)

# Filter to authors with multiple books and meaningful rating volume
author_summary_filtered = author_summary[
    (author_summary["book_count"] >= 3) &
    (author_summary["total_ratings"] >= 10000)
]

top_authors = author_summary_filtered.sort_values("avg_rating", ascending=False).head(15)

plt.figure(figsize=(12, 7))
sns.barplot(
    data=top_authors,
    x="avg_rating",
    y="authors"
)

plt.title("Highest Rated Authors with Multiple Books")
plt.xlabel("Average Rating")
plt.ylabel("Author")
plt.show()

# Reviewer-level behavior: identify optimistic and pessimistic reviewers
user_rating_behavior = (
    ratings
    .groupby("user_id")
    .agg(
        user_avg_rating=("rating", "mean"),
        rating_count=("rating", "count")
    )
    .reset_index()
)

# Filter to users with enough ratings to make the pattern meaningful
active_reviewers = user_rating_behavior[user_rating_behavior["rating_count"] >= 20]

active_reviewers["reviewer_type"] = np.where(
    active_reviewers["user_avg_rating"] >= 4.5,
    "Overly optimistic",
    np.where(
        active_reviewers["user_avg_rating"] <= 2.5,
        "Overly pessimistic",
        "Moderate"
    )
)

active_reviewers["reviewer_type"].value_counts()

plt.figure(figsize=(10, 6))
sns.histplot(active_reviewers["user_avg_rating"], bins=30, kde=True)

plt.title("Distribution of Average Ratings by Active Reviewers")
plt.xlabel("User Average Rating")
plt.ylabel("Number of Users")
plt.show()

plt.figure(figsize=(8, 5))
sns.countplot(
    data=active_reviewers,
    x="reviewer_type",
    order=["Overly pessimistic", "Moderate", "Overly optimistic"]
)

plt.title("Reviewer Rating Tendencies")
plt.xlabel("Reviewer Type")
plt.ylabel("Number of Active Reviewers")
plt.show()

"""The ratings are generally skewed toward higher values, which suggests that users tend to rate books favorably overall. Ratings also appear to vary by publication year and author, although author-level comparisons should be interpreted carefully because some authors have far more books and ratings than others. At the user level, some reviewers consistently rate books very high or very low, which may introduce bias into recommendation models.

---

#Recommendation System Code
"""



# Wrap the Goodreads ratings into a Surprise dataset.
# Goodreads ratings use a 1–5 star scale.
reader = Reader(rating_scale=(1, 5))

data = Dataset.load_from_df(
    ratings[["user_id", "book_id", "rating"]],
    reader
)

# Use all ratings to build the training set.
# This is useful for generating recommendations; later we'll use
# a train/test split to evaluate recommendation performance.
full_trainset = data.build_full_trainset()

print(
    f"Trainset: {full_trainset.n_users:,} users, "
    f"{full_trainset.n_items:,} books, "
    f"{full_trainset.n_ratings:,} ratings"
)

"""##Collaborative Filtering

###User-Based CF
"""

# User-based Collaborative Filtering (UBCF)
# Similar readers tend to enjoy similar books.
# Pearson correlation measures similarity between users based on
# their shared rating behavior.
# The model uses the 10 nearest neighbors to estimate unknown ratings.

ubcf = KNNBasic(
    k=10,
    sim_options={"name": "pearson", "user_based": True},
    verbose=False
)

ubcf.fit(full_trainset)

"""###Top-N Recommendations"""

MIN_RATINGS = 20
counts = ratings["book_id"].value_counts()
popular_books = set(counts[counts >= MIN_RATINGS].index)

book_lookup = books.set_index("book_id")[["title", "authors"]].to_dict("index")

def top_n_for_user(model, user_id, top_n=5):
    seen = set(ratings.loc[ratings["user_id"] == user_id, "book_id"])

    scored = [
        (
            book_lookup[b]["title"],
            book_lookup[b]["authors"],
            model.predict(user_id, b).est
        )
        for b in books["book_id"]
        if b not in seen and b in popular_books
    ]

    return sorted(scored, key=lambda x: -x[2])[:top_n]

top_n_for_user(ubcf, user_id=1, top_n=5)

"""###Item-Based CF"""

# Item-based collaborative filtering:
# Similarity is calculated between books using cosine similarity.
# Books that are rated similarly by readers are treated as similar items.

ibcf = KNNBasic(
    k=10,
    sim_options={
        "name": "cosine",
        "user_based": False
    },
    verbose=False
)

ibcf.fit(full_trainset)

print("Top 5 book recommendations from Item-Based Collaborative Filtering:")
top_n_for_user(ibcf, user_id=1, top_n=5)

"""##Evaluation

###Precision and Recall @ K
"""

# Top-N precision and recall for recommendation quality:
#   - a book is considered "relevant" if its true rating is >= threshold (4.0 here),
#   - for each user we rank predicted book ratings from highest to lowest,
#   - precision = fraction of recommended books in the top-N that were actually relevant,
#   - recall    = fraction of all relevant books that were captured in the top-N recommendations.
#
# top_n = number of recommendations evaluated
# k     = neighborhood size used by the collaborative filtering model
#
# Returns average precision and recall across all users.

def precision_recall_at_k(predictions, top_n=10, threshold=4.0):
    """Precision@K and Recall@K per user, averaged across users.

    Relevant:     a book whose true rating >= threshold.
    Recommended:  a book ranked in the user's top-N by predicted rating,
                  capped at the number of items actually available for that user.
    Precision = (# relevant and recommended) / (# recommended)
    Recall    = (# relevant and recommended) / (# relevant)
    """
    user_data = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_data[uid].append((est, true_r))

    precisions, recalls = [], []
    for items in user_data.values():
        items.sort(key=lambda x: x[0], reverse=True)

        top_items = items[:top_n]
        n_recommended = len(top_items)                       # min(top_n, available)
        n_relevant = sum(1 for _, t in items if t >= threshold)
        n_rel_and_rec = sum(1 for _, t in top_items if t >= threshold)

        if n_recommended > 0:
            precisions.append(n_rel_and_rec / n_recommended)
        if n_relevant > 0:
            recalls.append(n_rel_and_rec / n_relevant)

    return np.mean(precisions), np.mean(recalls)

"""###Train/Test Split"""

# Build a 90/10 train/test split so we can evaluate recommendations
# against held-out Goodreads ratings.
# random_state ensures the split is reproducible across runs.

trainset, testset = train_test_split(
    data,
    test_size=0.10,
    random_state=6604
)

print(
    f"Training set: {trainset.n_ratings:,} book ratings | "
    f"Test set: {len(testset):,} book ratings"
)

"""###

###Model Comparison
"""

# In practice, you would not hand-pick k or the similarity metric.
# Hyperparameters can be tuned using cross-validation
# (GridSearchCV or RandomizedSearchCV in Surprise).
#
# For a book recommendation system, evaluation should focus on ranking quality
# metrics such as Precision@K, Recall@K, or NDCG@K rather than RMSE alone.
#
# Production recommendation systems often use time-based train/test splits
# and validate improvements through online A/B testing.

models = {
    "Baseline": BaselineOnly(verbose=False),

    "User-Based CF (Pearson)": KNNBasic(
        k=10,
        sim_options={"name": "pearson", "user_based": True},
        verbose=False
    ),

    "Item-Based CF (Cosine)": KNNBasic(
        k=10,
        sim_options={"name": "cosine", "user_based": False},
        verbose=False
    ),
}

results = []

for name, model in models.items():

    model.fit(trainset)

    preds = model.test(testset)

    precision, recall = precision_recall_at_k(
        preds,
        top_n=10,
        threshold=4.0
    )

    results.append({
        "Model": name,
        "RMSE": accuracy.rmse(preds, verbose=False),
        "Precision@10": precision,
        "Recall@10": recall
    })

results_df = pd.DataFrame(results).round(4)

results_df

# Identify best models by each metric
best_rmse = results_df.sort_values("RMSE", ascending=True).iloc[0]
best_precision = results_df.sort_values("Precision@10", ascending=False).iloc[0]
best_recall = results_df.sort_values("Recall@10", ascending=False).iloc[0]

print(f"Best RMSE model: {best_rmse['Model']} with RMSE = {best_rmse['RMSE']}")
print(f"Best Precision@10 model: {best_precision['Model']} with Precision@10 = {best_precision['Precision@10']}")
print(f"Best Recall@10 model: {best_recall['Model']} with Recall@10 = {best_recall['Recall@10']}")

"""## Model Evaluation Results

Results from the 90/10 train-test split (random_state=6604):

| Model                   | RMSE   | Precision@10 | Recall@10 |
|-------------------------|--------|--------------|-----------|
| Baseline (bias)         | 0.8423 | 0.6854       | 0.7912    |
| User-Based CF (Pearson) | 1.0371 | 0.6852       | 0.7899    |
| Item-Based CF (Cosine)  | 0.9173 | 0.6531       | 0.7526    |

**Note on the baseline:** the "Baseline" here is Surprise's `BaselineOnly` estimator — a
global-mean + per-user bias + per-item bias model. It is not a simple popularity or
mean-rating model; it explicitly accounts for individual user and item rating tendencies.

**Which model wins and why:**

The bias baseline (`BaselineOnly`) wins on every metric — RMSE (0.8423), Precision@10
(0.6854), and Recall@10 (0.7912). User-Based CF is essentially tied on Precision@10 and
Recall@10 (within 0.001-0.001 of the baseline) but is substantially worse on RMSE
(1.037 vs 0.842). Item-Based CF trails on all three metrics, most notably on Precision@10
(0.653 vs 0.685).

**Why does the baseline outperform CF here?**

This is the expected outcome for a sparse Goodreads-style dataset. With a 90/10 split and
a matrix this sparse, most users in the test set have very few held-out ratings. KNNBasic
(the CF method used) requires meaningful neighborhood overlap to outperform a well-fitted
bias model — overlap that sparse interaction histories cannot reliably provide. The bias
baseline, by contrast, needs only global statistics and per-user/per-item offsets, which
remain stable even with sparse data. This is not a failure of the CF approach; it is a
well-known property of sparse collaborative filtering data and precisely the scenario the
assignment notes as a possibility.

**Recommendation:**

For this dataset and sparsity level, `BaselineOnly` is the recommended model. It produces
the best RMSE and equal-or-better ranking quality (Precision@10, Recall@10) with far
lower computational cost and no cold-start risk from neighborhood search. The LLM
re-ranking layer is still applied on top of IBCF (chosen for its item-level similarity
semantics, which align well with "books like this one" personalization), but the
evaluation makes clear that a stronger production deployment would start with the bias
model as the CF backbone.

# AI Re-Ranking Layer
---
---
"""

client = genai.Client(api_key=getpass.getpass("Enter your Gemini API key: "))
MODEL = "gemini-2.5-flash-lite"

class BookPick(BaseModel):
    title: str = Field(description="Exact book title from the candidate list.")
    authors: str = Field(description="Book author(s) as listed in the candidate list.")
    reason: str = Field(description="One sentence on why this book fits the preference.")


def rerank_with_gemini(user_id=1, preference="I want a warm, hopeful read with a strong emotional payoff.", top_n=5, model_name=ibcf):
    """Re-rank the collaborative-filtering top-N results with Gemini."""
    candidates = top_n_for_user(model_name, user_id=user_id, top_n=top_n)
    catalog = "\n".join(
        f"{i + 1}. {title} by {authors} (CF score: {score:.2f})"
        for i, (title, authors, score) in enumerate(candidates)
    )

    system_instruction = (
        "You are a book concierge. You must ONLY use books from the candidate list provided. "
        "Do NOT add any new books or remove any books. Re-rank ALL books from the candidate list "
        "based on the viewer’s preference, returning all of them in a new order with a short reason. "
        "Use the exact titles and author names from the candidate list. "
        "Your response must contain exactly the same number of books as the candidate list."
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=(
            f"Viewer preference: {preference}\n\n"
            f"Candidate books (already ranked by collaborative filtering):\n{catalog}"
        ),
        config={
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
            "response_schema": list[BookPick],
        },
    )

    return response.parsed


user_preference = "I want a warm, hopeful read with a strong emotional payoff."
llm_ranked = rerank_with_gemini(user_id=1, preference=user_preference, top_n=5, model_name=ibcf)

for pick in llm_ranked:
    print(f"{pick.title} by {pick.authors} -> {pick.reason}")
