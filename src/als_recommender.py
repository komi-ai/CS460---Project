"""
Alternating Least Squares (ALS) matrix factorisation recommender.

ALS is purpose-built for scalability:
  - Each latent-factor update reduces to a closed-form least-squares solve.
  - Updates for all users (or items) are independent, so they can be
    parallelised trivially.
  - Works well with sparse rating matrices, making it suitable for the
    larger MovieLens 1M dataset.

Algorithm (explicit-feedback ALS):
  Given ratings matrix R (users × items), find U (users × k) and V (items × k)
  such that  R ≈ U Vᵀ.

  Alternating updates:
    Fix V → solve for each row of U:
      uᵤ = (Vᵢᵤᵀ Vᵢᵤ + λI)⁻¹ Vᵢᵤᵀ rᵤ
    Fix U → solve for each row of V:
      vᵢ = (Uᵤᵢᵀ Uᵤᵢ + λI)⁻¹ Uᵤᵢᵀ rᵢ

  where Iᵤ / Uᵢ are the indices of items rated by user u (or users who
  rated item i) and rᵤ / rᵢ are the corresponding (centred) ratings.
"""

import time
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix


class ALSRecommender:
    """
    Explicit-feedback ALS matrix factorisation.

    Parameters
    ----------
    n_factors : int
        Number of latent factors (dimensionality of the embedding).
    n_iter : int
        Number of ALS iterations (each iteration updates U then V).
    reg : float
        L2 regularisation strength (λ).
    random_state : int
        Seed for reproducible initialisation.
    verbose : bool
        Print per-iteration timing and training RMSE.
    """

    def __init__(
        self,
        n_factors: int = 50,
        n_iter: int = 20,
        reg: float = 0.1,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.n_factors = n_factors
        self.n_iter = n_iter
        self.reg = reg
        self.random_state = random_state
        self.verbose = verbose

        self._user_factors: np.ndarray = None
        self._item_factors: np.ndarray = None
        self._user_map: dict = {}
        self._item_map: dict = {}
        self._user_unmap: dict = {}
        self._item_unmap: dict = {}
        self._global_mean: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_index_maps(series: pd.Series):
        """Return (forward_map id→idx, inverse_map idx→id) for a pandas Series."""
        unique = series.unique()
        fwd = {v: i for i, v in enumerate(unique)}
        inv = {i: v for v, i in fwd.items()}
        return fwd, inv

    @staticmethod
    def _als_update(factors_to_update: np.ndarray,
                    fixed_factors: np.ndarray,
                    R_sparse,  # CSR when updating users, CSC when updating items
                    reg: float) -> np.ndarray:
        """
        Update one set of factors (users or items) given the other is fixed.

        Parameters
        ----------
        factors_to_update : ndarray, shape (n, k)
            The factors that will be recomputed (mutated in-place and returned).
        fixed_factors : ndarray, shape (m, k)
            The factors held constant during this half-step.
        R_sparse : scipy sparse matrix
            For a user update pass the CSR form of R so rows are users.
            For an item update pass the CSC form so columns are items
            (each column is one item's rating vector).
        reg : float
            L2 regularisation coefficient λ.

        Returns
        -------
        ndarray
            Updated factors (same object as factors_to_update).
        """
        k = fixed_factors.shape[1]
        reg_eye = reg * np.eye(k)

        for idx in range(factors_to_update.shape[0]):
            # Retrieve non-zero entries for this user (CSR row) or item (CSC col)
            start = R_sparse.indptr[idx]
            end = R_sparse.indptr[idx + 1]
            if start == end:
                # No ratings — leave factor unchanged (or zero)
                continue

            other_indices = R_sparse.indices[start:end]   # rated items / raters
            r = R_sparse.data[start:end]                  # (centred) ratings

            F = fixed_factors[other_indices]              # shape (n_rated, k)
            A = F.T @ F + reg_eye                         # (k, k)
            b = F.T @ r                                   # (k,)
            factors_to_update[idx] = np.linalg.solve(A, b)

        return factors_to_update

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "ALSRecommender":
        """
        Fit the ALS model on a ratings DataFrame.

        Parameters
        ----------
        train_df : pd.DataFrame
            Must contain columns: userId, movieId, rating.

        Returns
        -------
        self
        """
        # --- Build id → index mappings -----------------------------------
        self._user_map, self._user_unmap = self._build_index_maps(train_df["userId"])
        self._item_map, self._item_unmap = self._build_index_maps(train_df["movieId"])

        n_users = len(self._user_map)
        n_items = len(self._item_map)

        # --- Centre ratings by the global mean ---------------------------
        self._global_mean = float(train_df["rating"].mean())
        user_idx = train_df["userId"].map(self._user_map).to_numpy(dtype=np.int32)
        item_idx = train_df["movieId"].map(self._item_map).to_numpy(dtype=np.int32)
        ratings = (train_df["rating"] - self._global_mean).to_numpy(dtype=np.float64)

        # --- Build sparse matrices ---------------------------------------
        R_csr = csr_matrix(
            (ratings, (user_idx, item_idx)), shape=(n_users, n_items), dtype=np.float64
        )
        R_csc = csc_matrix(R_csr)

        # --- Initialise factor matrices ----------------------------------
        rng = np.random.default_rng(self.random_state)
        self._user_factors = rng.standard_normal((n_users, self.n_factors)) * 0.1
        self._item_factors = rng.standard_normal((n_items, self.n_factors)) * 0.1

        if self.verbose:
            print(
                f"\nALS | n_users={n_users:,}  n_items={n_items:,}  "
                f"k={self.n_factors}  λ={self.reg}  iters={self.n_iter}"
            )

        # --- Alternating updates -----------------------------------------
        for iteration in range(1, self.n_iter + 1):
            t0 = time.time()

            # Update user factors (fix item factors)
            self._als_update(
                self._user_factors, self._item_factors, R_csr, self.reg
            )
            # Update item factors (fix user factors)
            self._als_update(
                self._item_factors, self._user_factors, R_csc, self.reg
            )

            if self.verbose:
                elapsed = time.time() - t0
                train_rmse = self._train_rmse(R_csr)
                print(
                    f"  Iter {iteration:>2}/{self.n_iter}  "
                    f"train RMSE={train_rmse:.4f}  ({elapsed:.1f}s)"
                )

        return self

    def _train_rmse(self, R_csr: csr_matrix) -> float:
        """Compute RMSE on the (centred) training data."""
        preds = np.array(R_csr[R_csr.nonzero()]).flatten()
        rows, cols = R_csr.nonzero()
        estimates = np.einsum("ij,ij->i", self._user_factors[rows], self._item_factors[cols])
        return float(np.sqrt(np.mean((preds - estimates) ** 2)))

    def predict(self, user_id: int, movie_id: int) -> float:
        """
        Predict the rating a user would give to a movie.

        Falls back to the global mean for unseen users or items.
        """
        if self._user_factors is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        if user_id not in self._user_map or movie_id not in self._item_map:
            return self._global_mean

        u = self._user_map[user_id]
        i = self._item_map[movie_id]
        pred = float(self._user_factors[u] @ self._item_factors[i]) + self._global_mean
        return float(np.clip(pred, 1.0, 5.0))

    def predict_batch(self, test_df: pd.DataFrame) -> np.ndarray:
        """Return an array of predictions for all rows in test_df."""
        return np.array(
            [self.predict(row.userId, row.movieId) for row in test_df.itertuples()]
        )

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ALSRecommender(n_factors={self.n_factors}, n_iter={self.n_iter}, "
            f"reg={self.reg})"
        )
