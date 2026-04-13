"""
KNN-based collaborative filtering recommender using the Surprise library.
Uses KNNWithMeans (user-based Pearson similarity by default).
Benchmark target: RMSE ≈ 0.98
"""

import pandas as pd
import numpy as np
from surprise import Dataset, Reader, KNNWithMeans
from surprise.model_selection import cross_validate


class KNNRecommender:
    """
    User-based KNN collaborative filtering via Surprise's KNNWithMeans.

    Parameters
    ----------
    k : int
        Maximum number of neighbours to consider.
    sim_name : str
        Similarity metric ('pearson', 'cosine', 'msd').
    user_based : bool
        If True use user-based CF; if False use item-based CF.
    """

    def __init__(self, k: int = 40, sim_name: str = "pearson", user_based: bool = True):
        self.k = k
        self.sim_name = sim_name
        self.user_based = user_based
        self._algo = None
        self._global_mean = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "KNNRecommender":
        """
        Fit the KNN model on a ratings DataFrame.

        Parameters
        ----------
        train_df : pd.DataFrame
            Must contain columns: userId, movieId, rating.

        Returns
        -------
        self
        """
        self._global_mean = float(train_df["rating"].mean())

        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(
            train_df[["userId", "movieId", "rating"]], reader
        )
        trainset = data.build_full_trainset()

        sim_options = {"name": self.sim_name, "user_based": self.user_based}
        self._algo = KNNWithMeans(k=self.k, sim_options=sim_options, verbose=False)
        self._algo.fit(trainset)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict the rating a user would give to a movie.

        Returns the global mean for unknown users / items.
        """
        if self._algo is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        pred = self._algo.predict(str(user_id), str(movie_id))
        return float(np.clip(pred.est, 1.0, 5.0))

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        """Return an array of predictions for all rows in test_df."""
        return np.array(
            [self.predict(row.userId, row.movieId) for row in test_df.itertuples()]
        )

    # ------------------------------------------------------------------
    # Cross-validation helper
    # ------------------------------------------------------------------

    def cross_validate(self, df: pd.DataFrame, cv: int = 5) -> dict:
        """
        Run k-fold cross-validation and return mean RMSE and MAE.

        Parameters
        ----------
        df : pd.DataFrame
            Full ratings DataFrame.
        cv : int
            Number of folds.

        Returns
        -------
        dict with keys 'rmse' and 'mae'.
        """
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader)
        sim_options = {"name": self.sim_name, "user_based": self.user_based}
        algo = KNNWithMeans(k=self.k, sim_options=sim_options, verbose=False)
        results = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=cv, verbose=False)
        return {
            "rmse": float(np.mean(results["test_rmse"])),
            "mae": float(np.mean(results["test_mae"])),
        }

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"KNNRecommender(k={self.k}, sim={self.sim_name}, "
            f"user_based={self.user_based})"
        )
