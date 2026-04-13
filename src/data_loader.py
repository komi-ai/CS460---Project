"""
Data loader for MovieLens datasets (100K and 1M).
Downloads and caches datasets locally in the data/ directory.
"""

import os
import zipfile
import urllib.request
import pandas as pd
from pathlib import Path

# Dataset URLs and metadata
DATASETS = {
    "100k": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
        "zip_name": "ml-100k.zip",
        "folder": "ml-100k",
        "ratings_file": "u.data",
        "sep": "\t",
        "columns": ["userId", "movieId", "rating", "timestamp"],
    },
    "1m": {
        "url": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "zip_name": "ml-1m.zip",
        "folder": "ml-1m",
        "ratings_file": "ratings.dat",
        "sep": "::",
        "columns": ["userId", "movieId", "rating", "timestamp"],
    },
}

# Default data directory relative to this file's location
_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data"


def _get_data_dir():
    """Return the data directory, creating it if needed."""
    data_dir = _DEFAULT_DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def download_dataset(name: str, data_dir: Path = None) -> Path:
    """
    Download and extract a MovieLens dataset if not already present.

    Parameters
    ----------
    name : str
        Dataset identifier: '100k' or '1m'.
    data_dir : Path, optional
        Directory where data should be stored. Defaults to project data/.

    Returns
    -------
    Path
        Path to the extracted dataset folder.
    """
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASETS.keys())}")

    if data_dir is None:
        data_dir = _get_data_dir()

    meta = DATASETS[name]
    folder_path = data_dir / meta["folder"]

    if folder_path.exists():
        return folder_path

    zip_path = data_dir / meta["zip_name"]
    if not zip_path.exists():
        print(f"Downloading MovieLens {name.upper()} dataset...")
        urllib.request.urlretrieve(meta["url"], zip_path)
        print(f"  Saved to {zip_path}")

    print(f"Extracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    print(f"  Extracted to {folder_path}")

    return folder_path


def load_ratings(name: str, data_dir: Path = None) -> pd.DataFrame:
    """
    Load a MovieLens ratings dataset as a DataFrame.

    Parameters
    ----------
    name : str
        Dataset identifier: '100k' or '1m'.
    data_dir : Path, optional
        Directory where data is stored (or will be downloaded to).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: userId, movieId, rating, timestamp.
    """
    folder = download_dataset(name, data_dir)
    meta = DATASETS[name]
    ratings_path = folder / meta["ratings_file"]

    df = pd.read_csv(
        ratings_path,
        sep=meta["sep"],
        names=meta["columns"],
        engine="python",
        encoding="latin-1",
    )
    df["userId"] = df["userId"].astype(int)
    df["movieId"] = df["movieId"].astype(int)
    df["rating"] = df["rating"].astype(float)
    return df


def train_test_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Split a ratings DataFrame into train and test sets.

    Parameters
    ----------
    df : pd.DataFrame
        Full ratings DataFrame.
    test_size : float
        Fraction of data to use for testing.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split as sk_split

    train, test = sk_split(df, test_size=test_size, random_state=random_state)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def dataset_summary(df: pd.DataFrame, name: str = ""):
    """Print a summary of a ratings dataset."""
    label = f"[{name}] " if name else ""
    print(f"{label}Ratings : {len(df):,}")
    print(f"{label}Users   : {df['userId'].nunique():,}")
    print(f"{label}Movies  : {df['movieId'].nunique():,}")
    print(f"{label}Rating range: {df['rating'].min():.1f} – {df['rating'].max():.1f}")
    print(f"{label}Mean rating : {df['rating'].mean():.3f}")
