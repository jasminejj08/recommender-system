# Recommender Systems — Top-N Book Recommendations

A practical comparison of four recommendation algorithms built from scratch on the GoodReads dataset. Explores trade-offs in accuracy, sparsity handling, and recommendation diversity across collaborative filtering, latent factor, and content-based approaches.

---

## Algorithms Implemented

| Algorithm | File |
|---|---|
| Item-Item Collaborative Filtering | `cf_item_item_git.py` |
| User-User Collaborative Filtering | `cf_user_user_git.py` |
| SVD-based Latent Factor Model | `latent_model_svd_git.py` |
| Content-Based Filtering | `content_based_git.py` |

---

## Results

RMSE evaluated by hiding known ratings and measuring prediction error across two training sizes.

| Algorithm | RMSE (train_n = 100) | RMSE (train_n = 300) |
|---|---|---|
| **Item-Item CF** | **0.832** (BEST) | 0.855 |
| User-User CF | 0.926 | 0.990 |
| SVD (k=2) | 0.882 | 0.916 |
| SVD (k=5) | 0.869 | 0.911 |
| SVD (k=10) | 0.849 | 0.901 |
| SVD (k=25) | 0.852 | 0.909 |
| Content-Based | — | — |

**Key observations:**
- Item-Item CF achieved the lowest RMSE overall (0.832), outperforming User-User CF by ~10%
- SVD accuracy improved consistently from k=2 to k=10, with diminishing returns beyond that; suggesting k=10 as the sweet spot for this dataset
- User-User CF was most sensitive to training size, with RMSE degrading notably at larger `train_n`, likely due to increased sparsity in the user-user similarity matrix
- Due to no training data for the content-based algorithm, no RMSE is provided

---

## Dataset

**GoodReads Books Dataset** — available on [Kaggle](https://www.kaggle.com/datasets/bahramjannesarr/goodreads-book-datasets-10m/data)

The raw data files are not included in this repository due to size. To reproduce results:

1. Download the dataset from Kaggle
2. Place the user rating CSV files and book metadata CSV files in the correct directory (check clean_data.py main function to edit path to files)
---

## How to Run

> **Important:** Scripts must be run in order. Each algorithm depends on the output of `clean_data.py`. So run this first!

### Step 1: Data Preprocessing
```bash
python clean_data.py
```
This creates a new folder in your working directory containing:
- `utility_matrix.csv` — the user-item matrix used by all algorithm scripts
- `cleaned_books_metadata.csv` (zipped) — cleaned book metadata for content-based filtering

Note that you must have utility_matrix.csv file created to run all other scripts!!!

### Step 2: Run the Algorithms

Run each script individually. All parameters (target user, training size, number of recommendations) can be adjusted in the `main()` function of each file.

```bash
python cf_item_item_git.py    # Item-Item Collaborative Filtering
python cf_user_user_git.py    # User-User Collaborative Filtering
python latent_model_svd_git.py      # SVD Latent Factor Model
python content_based_git.py         # Content-Based Filtering
```

Each script outputs Top-N recommended book IDs for a chosen target user.

> **Note:** Some files may take a moment to run due to dataset size. Do not uncomment the debugging print statements unless you want to see intermediate outputs; since these are inside loops, they will print many lines.

---

## Skills Demonstrated

`Python` `Pandas` `NumPy` `Scikit-learn` `Collaborative Filtering` `Matrix Factorization` `SVD` `Content-Based Filtering` `Data Preprocessing` `Cosine Similarity` `Recommendation Systems`
