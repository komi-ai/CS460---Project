# CS460 – Recommender Systems Project

Collaborative-filtering recommender systems evaluated on the **MovieLens 100K** and **MovieLens 1M** datasets.

## Methods

| Method | Description | Benchmark Target (RMSE) |
|--------|-------------|------------------------|
| **KNN** | User-based k-nearest-neighbours with Pearson similarity (`KNNWithMeans` via Surprise) | ≤ 0.98 |
| **SVD** | Biased matrix factorisation (Simon Funk SGD-SVD via Surprise) | ≤ 0.93 |
| **ALS** | **Alternating Least Squares** – closed-form matrix factorisation designed for scalability | competitive |
| **Hybrid** | Weighted ensemble of KNN + SVD + ALS; weights are optimised on a held-out validation split | best of all |

### Why ALS?

ALS solves for each latent factor vector in closed form (a small linear system), making every update embarrassingly parallel. This makes it especially well-suited to the larger 1M dataset and trivial to scale further (e.g., distributed Spark ALS).

## Project Structure

```
CS460---Project/
├── main.py                 # Entry point – trains & evaluates all methods
├── requirements.txt
├── src/
│   ├── data_loader.py      # Downloads / loads MovieLens 100K & 1M
│   ├── knn_recommender.py  # KNN collaborative filtering
│   ├── svd_recommender.py  # SVD matrix factorisation
│   ├── als_recommender.py  # ALS matrix factorisation (new)
│   ├── hybrid_recommender.py  # Weighted ensemble
│   └── evaluation.py       # RMSE, MAE, timing utilities
└── data/                   # Downloaded datasets (auto-created, git-ignored)
```

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
# Run all methods on both datasets (downloads data automatically)
python main.py

# Run on 100K only (faster)
python main.py --datasets 100k

# Skip the Hybrid model
python main.py --no-hybrid

# Skip the results plot
python main.py --no-plot
```

Results are printed to the terminal and saved as `results.png`.

## Datasets

The datasets are downloaded automatically on first run from GroupLens:

- **MovieLens 100K** – 100,000 ratings, 943 users, 1,682 movies
- **MovieLens 1M** – 1,000,209 ratings, 6,040 users, 3,952 movies

## Dependencies

- `numpy`, `scipy` – numerical computing and sparse matrix operations
- `pandas` – data loading and manipulation
- `scikit-surprise` – KNN and SVD implementations
- `scikit-learn` – train/test split
- `matplotlib` – results visualisation