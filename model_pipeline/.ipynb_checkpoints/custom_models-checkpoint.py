# merf_wrapper_embed.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from merf.merf import MERF

class MERFWrapperEmbed(BaseEstimator, RegressorMixin):
    """
    A MERF wrapper that embeds random-effect features Z and cluster IDs 
    within the feature matrix X (the last columns).
    Exposes hyperparameters for MERF's EM (max_iterations) and the random forest (rf__n_estimators).
    
    By default:
      - The last column of X is cluster IDs (cluster_col_idx=-1).
      - The preceding q_cols columns are Z features (z_start_col = -1 - q_cols).
      - The rest of the columns are the fixed-effect features.
    """

    def __init__(
        self,
        gll_early_stop_threshold=0.01, 
        max_iterations=20,
        rf__n_estimators=300,  # new param to tune the random forest
        cluster_col_idx=-1,
        z_start_col=-2,
    ):
        """
        Parameters
        ----------
        gll_early_stop_threshold : float
            Early stopping threshold for MERF's EM.
        max_iterations : int
            Maximum EM iterations for MERF.
        rf__n_estimators : int
            Number of trees in the random forest (fixed effects).
        cluster_col_idx : int
            Column index in X that contains cluster IDs.
        z_start_col : int
            The start column index in X for Z features.

        """
        self.gll_early_stop_threshold = gll_early_stop_threshold
        self.max_iterations = max_iterations
        self.rf__n_estimators = rf__n_estimators
        self.cluster_col_idx = cluster_col_idx
        self.z_start_col = z_start_col


        fe_model = RandomForestRegressor(n_estimators=self.rf__n_estimators, n_jobs=-1)
        self.merf_model = MERF(
            fixed_effects_model=fe_model,
            gll_early_stop_threshold=self.gll_early_stop_threshold,
            max_iterations=self.max_iterations
        )

    def fit(self, X, y):
        """ 
        X => DataFrame or array with shape (n_samples, n_fixed + q_cols + 1).
        The last columns must contain [Z columns, cluster column].
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 1) Extract cluster IDs
        clusters_series = X.iloc[:, self.cluster_col_idx]
        if not isinstance(clusters_series, pd.Series):
            clusters_series = pd.Series(clusters_series.values, index=X.index)

        # 2) Extract Z features
        Z_part = X.iloc[:, [self.z_start_col]]

        # 3) Extract fixed-effect columns => exclude cluster col + Z columns
        fixed_cols_mask = np.ones(X.shape[1], dtype=bool)
        fixed_cols_mask[self.cluster_col_idx] = False
        fixed_cols_mask[self.z_start_col] = False
        X_fixed = X.loc[:, fixed_cols_mask]

        # Fit MERF
        self.merf_model.fit(X_fixed.values, Z_part.values, clusters_series, np.array(y))
        return self
        
    def predict(self, X):
            """
            For predicting, we slice out [Z, cluster], feed them to MERF's .predict(...).
            """
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X)
    
            clusters_series = X.iloc[:, self.cluster_col_idx]
            if not isinstance(clusters_series, pd.Series):
                clusters_series = pd.Series(clusters_series.values, index=X.index)
            # 2) Extract Z features
            Z_part = X.iloc[:, [self.z_start_col]]
    
            fixed_cols_mask = np.ones(X.shape[1], dtype=bool)
            fixed_cols_mask[self.cluster_col_idx] = False
            fixed_cols_mask[self.z_start_col] = False
            X_fixed = X.loc[:, fixed_cols_mask]
    
            return self.merf_model.predict(X_fixed.values, Z_part.values, clusters_series)

    def get_params(self, deep=True):
        """Exposing hyperparameters for scikit-learn's GridSearchCV."""
        return {
            "gll_early_stop_threshold": self.gll_early_stop_threshold,
            "max_iterations": self.max_iterations,
            "rf__n_estimators": self.rf__n_estimators,
            "cluster_col_idx": self.cluster_col_idx,
            "z_start_col": self.z_start_col,
        }

    def set_params(self, **params):
        """
        If scikit-learn changes 'max_iterations' or 'rf__n_estimators', 
        we re-init MERF with a new RandomForestRegressor(...) 
        so each hyperparam combo is tested properly.
        """
        for param, value in params.items():
            setattr(self, param, value)

        # If relevant MERF or RF params changed, re-init
        if any(k in params for k in ["gll_early_stop_threshold","max_iterations","rf__n_estimators"]):
            fe_model = RandomForestRegressor(n_estimators=self.rf__n_estimators, n_jobs=-1)
            self.merf_model = MERF(
                fixed_effects_model=fe_model,
                gll_early_stop_threshold=self.gll_early_stop_threshold,
                max_iterations=self.max_iterations
            )
        return self       
