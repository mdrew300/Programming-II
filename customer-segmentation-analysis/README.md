# European Traveler Segmentation

Unsupervised market segmentation of European travelers from Google review ratings, built to support **Travelbiz**, a U.S. online travel firm planning to launch European travel packages. The analysis surfaces eight behaviorally distinct traveler segments to replace generic marketing with targeted package design and channel strategy.

**Stack:** Python · scikit-learn · UMAP · t-SNE · PCA · K-Means · pandas · seaborn

> Graduate group project — Georgetown McDonough MSBA, OPAN-6603 (Dr. Tommy Jones).

---

## Problem

The European travel market is too large and heterogeneous to reach with a single offering. Travelbiz needs to know *who* its prospective customers are and *what* they want before spending marketing dollars. We approach this as an unsupervised segmentation problem: find latent traveler groups from behavior alone, then translate those groups into product, positioning, and targeting recommendations.

## Data

- **Source:** `Travel_Review.xlsx` — Google review ratings across European destinations (Travel Review Ratings dataset).
- **Shape:** 5,455 users × 24 attraction categories, ratings scaled 0–5. Each row is a user; each column is that user's average rating for an attraction type.
- **Categories:** Churches, Resorts, Beaches, Parks, Theatres, Museums, Malls, Zoo, Restaurants, Pubs/Bars, LocalServices, Burger/PizzaShops, Hotels/OtherLodgings, JuiceBars, ArtGalleries, DanceClubs, Swimming Pools, Gyms, Bakeries, BeautySpas, Cafes, ViewPoints, Monuments, Gardens.

### Preparation
- Missing values treated as `0` (interpreted as no engagement with that category).
- **Low-engagement filter:** users with `0` ratings across ≥25% of categories are dropped to reduce noise.
- **Transform:** Yeo-Johnson power transform to reduce skew, followed by standardization (`StandardScaler`) so all categories carry equal weight.
- A six-group **meta-feature** mapping is also built for interpretation: Cultural Heritage, Nature & Outdoor, Shopping, Food & Beverage, Accommodation, Recreation & Wellness.

## Method

The workflow combines dimensionality reduction, clustering, and supervised models used purely to *explain* the clusters.

1. **EDA** — distributions, summary stats, and a Pearson correlation heatmap to identify correlated attribute groups (e.g., food/nightlife; nature/recreation).
2. **Dimensionality reduction** — PCA (variance/scree + meta-feature PCA), plus t-SNE (`perplexity=30`) and UMAP (`n_neighbors=25`, `min_dist=0.1`) for 2-D visualization of structure.
3. **Clustering** — K-Means on the standardized feature space. `k` chosen via elbow (WCSS) + silhouette diagnostics across `k = 2…20`; **K = 8** selected as the balance of interpretability and separation.
4. **Validation**
   - Silhouette analysis per cluster.
   - **Stability:** Adjusted Rand Index across 30 bootstrap resamples (80% subsamples) to confirm assignments aren't an artifact of a single fit.
   - **Cross-method agreement:** Ward hierarchical clustering compared to K-Means via ARI + crosstab.
5. **Cluster interpretation (supervised augmentation)** — a multinomial logistic regression and a depth-4 decision tree are trained to *predict* cluster membership. Their coefficients and Gini feature importances explain which attractions drive each segment (5-fold CV reported for sanity, not as a predictive deliverable).

Hierarchical clustering recovers broadly similar structure (ARI 0.51), supporting but not confirming the eight-segment solution.

## Results — Eight Segments

| Tier | Segment | Profile |
|------|---------|---------|
| 1 | **Shopping & Dining Enthusiasts** | Largest, most cohesive; food, nightlife, retail |
| 1 | **Urban Experience Seekers** | Dense city experiences |
| 1 | **Cultural & Natural Heritage Lovers** | Cross-cutting cultural + nature interest |
| 2 | **Convenience-Focused Lodgers** | Lodging/logistics-driven |
| 2 | **Local Culture & Social Travelers** | Local, social experiences |
| 2 | **Arts & Entertainment Aficionados** | Niche arts focus |
| 3 | **Nature & Outdoor Explorers** | Beaches, parks, outdoors |
| 3 | **Balanced Budget Travelers** | Lower overall engagement; broad, price-sensitive |

PCA reduces the 24 attractions to roughly six intuitive meta-categories that mirror how European tourism boards classify products. Decision-tree importances flag churches, pubs/bars, bakeries, and museums as the strongest differentiators between segments.

**Recommendations** (detailed in the report): build experience-based, all-inclusive packages with à la carte add-ons; embed a cultural component as a baseline across all packages; and target via influencer partnerships matched to each segment's channels, supported by in-house social/SEO.

## Repository Structure

```
.
├── README.md
├── data/
│   └── Travel_Review.xlsx              # input dataset (5,455 × 24)
├── notebooks/
│   └── Saxa_4_Project_3_Full_Dataset.ipynb   # full analysis notebook
├── report/
│   └── OPAN-6603_Final_Project_Saxa_4.pdf    # written report + figures
└── requirements.txt
```

## Running It
```bash
pip install -r requirements.txt
jupyter lab   # or: jupyter notebook
```

### `requirements.txt`
```
numpy
pandas
matplotlib
seaborn
scikit-learn
umap-learn
scipy
openpyxl
```

## Reproducibility Notes
- t-SNE/UMAP are used for visualization; the reported K-Means segmentation is fit on the standardized feature space.

## Limitations
- Dropping high-zero users biases toward already-engaged travelers and may underrepresent first-time or low-activity visitors.
- Review behavior is a proxy for preference, not booking intent — it doesn't capture budget, travel companions, or schedule constraints.
- The data may not reflect the most recent market trends.

## Authors
Saxa Team 4 — Matthew Drew, Anya Satyawadi, Jaci Goode, Prince Yeboah, Traore Rouguiatou.
> My contribution: Exploratory Data and Primary Component Analyses

Course: OPAN-6603, Georgetown University McDonough School of Business · Instructor: Dr. Tommy Jones.

Generative AI tools assisted with code development and report editing.
