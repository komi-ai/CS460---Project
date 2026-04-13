"""
Hybrid recommender that combines KNN, SVD, and ALS predictions via a
weighted average.  Weights are optimised on a held-out validation set
(or can be supplied manually).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .knn_recommender import KNNRecommender
from .svd_recommender import SVDRecommender
from .als_recommender import ALSRecommender


class HybridRecommender:
    """
    Weighted ensemble of KNN, SVD, and ALS recommenders.

    Parameters
    ----------
    knn_weight : float, optional
        Weight for KNN predictions.  If None, weights are optimised
        on a validation split of the training data.
    svd_weight : float, optional
        Weight for SVD predictions.
    als_weight : float, optional
        Weight for ALS predictions.
    val_fraction : float
        Fraction of training data used for weight optimisation when
        weights are not provided.
    knn_params : dict, optional
        Keyword arguments forwarded to KNNRecommender.
    svd_params : dict, optional
        Keyword arguments forwarded to SVDRecommender.
    als_params : dict, optional
        Keyword arguments forwarded to ALSRecommender.
    """

    def __init__(
        self,
        knn_weight: float = None,
        svd_weight: float = None,
        als_weight: float = None,
        val_fraction: float = 0.1,
        knn_params: dict = None,
        svd_params: dict = None,
        als_params: dict = None,
    ):
        self.knn_weight = knn_weight
        self.svd_weight = svd_weight
        self.als_weight = als_weight
        self.val_fraction = val_fraction

        self.knn = KNNRecommender(**(knn_params or {}))
        self.svd = SVDRecommender(**(svd_params or {}))
        self.als = ALSRecommender(**(als_params or {"verbose": False}))

        self._weights: np.ndarray = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "HybridRecommender":
        """
        Fit all component models and determine blending weights.

        Parameters
        ----------
        train_df : pd.DataFrame
            Must contain columns: userId, movieId, rating.

        Returns
        -------
        self
        """
        # --- Decide whether to optimise weights --------------------------
        manual_weights = (
            self.knn_weight is not None
            and self.svd_weight is not None
            and self.als_weight is not None
        )

        if manual_weights:
            total = self.knn_weight + self.svd_weight + self.als_weight
            self._weights = np.array(
                [self.knn_weight, self.svd_weight, self.als_weight], dtype=float
            ) / total
            fit_df = train_df
            val_df = None
        else:
            # Hold out a validation set for weight learning
            from sklearn.model_selection import train_test_split as sk_split
            fit_df, val_df = sk_split(
                train_df, test_size=self.val_fraction, random_state=0
            )

        # --- Fit all component models ------------------------------------
        print("  [Hybrid] Fitting KNN...")
        self.knn.fit(fit_df)
        print("  [Hybrid] Fitting SVD...")
        self.svd.fit(fit_df)
        print("  [Hybrid] Fitting ALS...")
        self.als.fit(fit_df)

        # --- Optimise weights on validation set (if needed) --------------
        if not manual_weights and val_df is not None:
            self._weights = self._optimise_weights(val_df)

        print(
            f"  [Hybrid] Blending weights — "
            f"KNN: {self._weights[0]:.3f}  "
            f"SVD: {self._weights[1]:.3f}  "
            f"ALS: {self._weights[2]:.3f}"
        )
        return self

    def _optimise_weights(self, val_df: pd.DataFrame) -> np.ndarray:
        """
        Find blend weights (w_knn, w_svd, w_als) that minimise RMSE on val_df.
        Weights are constrained to be non-negative and sum to 1.
        """
        knn_preds = self.knn.predict_batch(val_df)
        svd_preds = self.svd.predict_batch(val_df)
        als_preds = self.als.predict_batch(val_df)
        actuals = val_df["rating"].to_numpy()

        stacked = np.column_stack([knn_preds, svd_preds, als_preds])  # (n, 3)

        def rmse(w):
            blend = stacked @ w
            return np.sqrt(np.mean((actuals - blend) ** 2))

        # Simplex constraint: w ≥ 0, sum(w) = 1
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
        bounds = [(0.0, 1.0)] * 3
        result = minimize(
            rmse,
            x0=np.array([1 / 3, 1 / 3, 1 / 3]),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        return result.x

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, user_id: int, movie_id: int) -> float:
        """Predict the rating a user would give to a movie."""
        if self._weights is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        preds = np.array([
            self.knn.predict(user_id, movie_id),
            self.svd.predict(user_id, movie_id),
            self.als.predict(user_id, movie_id),
        ])
        return float(np.clip(self._weights @ preds, 1.0, 5.0))

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        """Return an array of predictions for all rows in test_df."""
        knn_p = self.knn.predict_batch(test_df)
        svd_p = self.svd.predict_batch(test_df)
        als_p = self.als.predict_batch(test_df)
        stacked = np.column_stack([knn_p, svd_p, als_p])
        blended = stacked @ self._weights
        return np.clip(blended, 1.0, 5.0)

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if self._weights is not None:
            w = self._weights
            return (
                f"HybridRecommender(KNN={w[0]:.3f}, SVD={w[1]:.3f}, ALS={w[2]:.3f})"
            )
        return "HybridRecommender(weights=not fitted)"
