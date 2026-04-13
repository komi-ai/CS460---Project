"""
SVD-based matrix factorisation recommender using the Surprise library.
Implements Simon Funk's gradient-descent SVD (biased MF).
Benchmark target: RMSE ≈ 0.93
"""

import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate


class SVDRecommender:
    """
    SVD matrix factorisation via Surprise's SVD implementation.

    Parameters
    ----------
    n_factors : int
        Number of latent factors.
    n_epochs : int
        Number of gradient-descent epochs.
    lr_all : float
        Learning rate for all parameters.
    reg_all : float
        Regularisation term for all parameters.
    """

    def __init__(
        self,
        n_factors: int = 100,
        n_epochs: int = 20,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self._algo = None
        self._global_mean = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "SVDRecommender":
        """
        Fit the SVD model on a ratings DataFrame.

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

        self._algo = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            verbose=False,
        )
        self._algo.fit(trainset)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict the rating a user would give to a movie."""
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
        algo = SVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            verbose=False,
        )
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
            f"SVDRecommender(n_factors={self.n_factors}, n_epochs={self.n_epochs}, "
            f"lr_all={self.lr_all}, reg_all={self.reg_all})"
        )
