import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")

URLS = {
    "100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    #"10m": "https://files.grouplens.org/datasets/movielens/ml-10m.zip", if we have time we can try the larger set
}

def make_folders():
    # Create folders
    DATA_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)


def download_dataset(name):
    # Download dataset
    zip_path = DATA_DIR / f"ml-{name}.zip"
    folder_path = DATA_DIR / f"ml-{name}"

    if not zip_path.exists():
        print(f"Downloading MovieLens {name}...")
        r = requests.get(URLS[name], timeout=120)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            f.write(r.content)

    if not folder_path.exists():
        print(f"Extracting MovieLens {name}...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR)

    return folder_path

def load_data(name):
    # Load data
    folder = download_dataset(name)

    if name == "100k":
        df = pd.read_csv(
            folder / "u.data",
            sep="\t",
            names=["user_id", "item_id", "rating", "timestamp"]
        )
    else:
        df = pd.read_csv(
            folder / "ratings.dat",
            sep="::",
            engine="python",
            names=["user_id", "item_id", "rating", "timestamp"]
        )

    df["user_id"] = df["user_id"].astype(int)
    df["item_id"] = df["item_id"].astype(int)
    df["rating"] = df["rating"].astype(float)
    return df

def get_stats(df):
    # Get stats
    return len(df), df["user_id"].nunique(), df["item_id"].nunique()


def calc_rmse_mae(y_true, y_pred):
    # Calc errors
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    return rmse, mae


def make_surprise_data(df):
    # Prep data
    from surprise import Dataset, Reader

    train_df, test_df = train_test_split(df[["user_id", "item_id", "rating"]], test_size=0.2, random_state=42)

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_df, reader)
    trainset = data.build_full_trainset()
    testset = list(test_df.itertuples(index=False, name=None))

    return trainset, testset

def run_knn(df, dataset_name):
    # KNN model
    from surprise import KNNBaseline, accuracy

    trainset, testset = make_surprise_data(df)

    model = KNNBaseline(
        k=40,
        sim_options={"name": "pearson_baseline", "user_based": False},
        verbose=False
    )

    start_train = time.time()
    model.fit(trainset)
    train_time = time.time() - start_train

    start_test = time.time()
    preds = model.test(testset)
    test_time = time.time() - start_test

    rmse = accuracy.rmse(preds, verbose=False)
    mae = accuracy.mae(preds, verbose=False)

    return {
        "dataset": dataset_name,
        "model": "KNN",
        "rmse": rmse,
        "mae": mae,
        "train_time": train_time,
        "test_time": test_time,
    }
    
def run_svd(df, dataset_name):
    # SVD model
    from surprise import SVD, accuracy

    trainset, testset = make_surprise_data(df)

    model = SVD(n_factors=100, n_epochs=20, random_state=42)

    start_train = time.time()
    model.fit(trainset)
    train_time = time.time() - start_train

    start_test = time.time()
    preds = model.test(testset)
    test_time = time.time() - start_test

    rmse = accuracy.rmse(preds, verbose=False)
    mae = accuracy.mae(preds, verbose=False)

    return {
        "dataset": dataset_name,
        "model": "SVD",
        "rmse": rmse,
        "mae": mae,
        "train_time": train_time,
        "test_time": test_time,
    }
    # Hybrid model
def run_hybrid(df, dataset_name):
    from surprise import KNNBaseline, SVD

    trainset, testset = make_surprise_data(df)

    knn = KNNBaseline(
        k=40,
        sim_options={"name": "pearson_baseline", "user_based": False},
        verbose=False
    )
    svd = SVD(n_factors=100, n_epochs=20, random_state=42)

    start_train = time.time()
    knn.fit(trainset)
    svd.fit(trainset)
    train_time = time.time() - start_train

    alpha = 0.7
    y_true = []
    y_pred = []

    start_test = time.time()
    for uid, iid, true_r in testset:
        knn_pred = knn.predict(uid, iid).est
        svd_pred = svd.predict(uid, iid).est
        final_pred = alpha * svd_pred + (1 - alpha) * knn_pred
        final_pred = max(1, min(5, final_pred))

        y_true.append(true_r)
        y_pred.append(final_pred)

    test_time = time.time() - start_test
    rmse, mae = calc_rmse_mae(y_true, y_pred)

    return {
        "dataset": dataset_name,
        "model": "Hybrid",
        "rmse": rmse,
        "mae": mae,
        "train_time": train_time,
        "test_time": test_time,
    }
    
    #als
def run_als(df, dataset_name):
    from pyspark.sql import SparkSession
    from pyspark.ml.recommendation import ALS

    spark = (
        SparkSession.builder
        .appName("MovieLensALS")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    sdf = spark.createDataFrame(df[["user_id", "item_id", "rating"]])
    train_df, test_df = sdf.randomSplit([0.8, 0.2], seed=42)

    als = ALS(
        userCol="user_id",
        itemCol="item_id",
        ratingCol="rating",
        rank=50,
        maxIter=10,
        regParam=0.1,
        coldStartStrategy="drop",
        seed=42
    )

    start_train = time.time()
    model = als.fit(train_df)
    train_time = time.time() - start_train

    start_test = time.time()
    preds = model.transform(test_df).dropna(subset=["prediction"]).toPandas()
    test_time = time.time() - start_test

    spark.stop()

    rmse, mae = calc_rmse_mae(preds["rating"], preds["prediction"])

    return {
        "dataset": dataset_name,
        "model": "ALS",
        "rmse": rmse,
        "mae": mae,
        "train_time": train_time,
        "test_time": test_time,
    }
    
    
def main():
    # Main to run all models
    make_folders()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["100k"],
        choices=["100k", "1m"]
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["knn", "svd", "als"],
        choices=["knn", "svd", "als", "hybrid"]
    )

    args = parser.parse_args()

    results = []
    for dataset_name in args.datasets:
        print(f"Loading dataset {dataset_name}...")
        df = load_data(dataset_name)
        num_ratings, num_users, num_items = get_stats(df)
        print(f"Dataset {dataset_name}: {num_ratings} ratings, {num_users} users, {num_items} items")

        for model_name in args.models:
            print(f"Running {model_name} on {dataset_name}...")
            if model_name == "knn":
                result = run_knn(df, dataset_name)
            elif model_name == "svd":
                result = run_svd(df, dataset_name)
            elif model_name == "hybrid":
                result = run_hybrid(df, dataset_name)
            elif model_name == "als":
                result = run_als(df, dataset_name)
            else:
                print(f"Model {model_name} not implemented yet, skipping.")
                continue
            results.append(result)
            print(f"Results: {result}")

    # Save results
    results_file = RESULTS_DIR / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")


if __name__ == "__main__":
    main()


