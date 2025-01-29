# merf_wrapper_embed.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from merf.merf import MERF

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class GroupwiseStandardizingRegressor(BaseEstimator, RegressorMixin):
    """
    Wraps any model and applies standardization to the target variable y **per customer**.
    """

    def __init__(self, base_model, group_col="customer"):
        self.base_model = base_model
        self.group_col = group_col  # Name of customer ID column
        self.scalers_ = {}  # Store separate scalers per customer

    def fit(self, X, y):
        """Standardize `y` per customer before training."""
        X = pd.DataFrame(X) if isinstance(X, np.ndarray) else X

        if self.group_col not in X.columns:
            raise ValueError(f"Input data must contain '{self.group_col}' column.")

        df = X.copy()
        df["y"] = y  # Add `y` to DataFrame

        # Standardize `y` per customer
        y_scaled = np.zeros_like(y, dtype=np.float64)
        for customer, group in df.groupby(self.group_col):
            scaler = StandardScaler()
            y_scaled[group.index] = scaler.fit_transform(group["y"].values.reshape(-1, 1)).flatten()
            self.scalers_[customer] = scaler  # Store scaler

        # Fit model on standardized target
        X_train = df.drop(columns=["y"])
        self.base_model.fit(X_train, y_scaled)
        return self

    def predict(self, X):
        """Predict using trained model and inverse transform `y` per customer."""
        X = pd.DataFrame(X) if isinstance(X, np.ndarray) else X

        if self.group_col not in X.columns:
            raise ValueError(f"Input data must contain '{self.group_col}' column.")

        y_pred_scaled = self.base_model.predict(X)  # Predict on standardized scale
        y_pred = np.zeros_like(y_pred_scaled, dtype=np.float64)

        # Inverse transform per customer
        for customer, group in X.groupby(self.group_col):
            scaler = self.scalers_.get(customer, StandardScaler())  # Use stored scaler
            y_pred[group.index] = scaler.inverse_transform(y_pred_scaled[group.index].reshape(-1, 1)).flatten()

        return y_pred



class MERFWrapperEmbed(BaseEstimator, RegressorMixin):
    """
    A MERF wrapper that embeds cluster IDs within the feature matrix X.
    Expects 'cluster' and 'intercept' columns to be present in X.
    """

    def __init__(
        self,
        gll_early_stop_threshold=0.01, 
        max_iterations=20,
        rf__n_estimators=300,  # Hyperparameter to tune the random forest
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
        """
        self.gll_early_stop_threshold = gll_early_stop_threshold
        self.max_iterations = max_iterations
        self.rf__n_estimators = rf__n_estimators

        # Initialize MERF model
        fe_model = RandomForestRegressor(n_estimators=self.rf__n_estimators, n_jobs=-1)
        self.merf_model = MERF(
            fixed_effects_model=fe_model,
            gll_early_stop_threshold=self.gll_early_stop_threshold,
            max_iterations=self.max_iterations
        )
    def fit(self, X, y):
        """ Fit the MERF model. """
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
    
        #  Debugging Step 1: Print Initial Columns
        print("🔍 Columns in X before customer check:", X.columns.tolist())
    
        # 1) Extract customer IDs
        if 'customer' not in X.columns:
            raise ValueError(" Input data must contain 'customer' column. Available columns: " + str(X.columns.tolist()))
    
        clusters_series = X['customer']
        if not isinstance(clusters_series, pd.Series):
            clusters_series = pd.Series(clusters_series.values, index=X.index)
    
        #  Debugging Step 2: Print After Extracting `customer`
        print("Customer column is present. First few values:", clusters_series.head())
    
        # 2) Extract intercept column
        if 'intercept' not in X.columns:
            raise ValueError(" Input data must contain 'intercept' column.")
    
        intercept_series = X['intercept']
        if not np.all(intercept_series.values == 1):
            raise ValueError(" Intercept column must be a constant value of 1.")
    
        #  Debugging Step 3: Print After Extracting `intercept`
        print("Intercept column is present. First few values:", intercept_series.head())
    
        # 3) Extract fixed-effect columns => exclude 'customer' and 'intercept'
        X_fixed = X.drop(columns=['customer', 'intercept'])
    
        # Debugging Step 4: Print Remaining Columns After Dropping
        print(" Columns in X_fixed after dropping 'customer' and 'intercept':", X_fixed.columns.tolist())
    
        # Fit MERF
        self.merf_model.fit(X_fixed.values, intercept_series.values, clusters_series, np.array(y))
        return self

    def predict(self, X):
        """
        Predict using the MERF model.

        Parameters
        ----------
        X : DataFrame or array-like, shape (n_samples, n_features)
            Feature matrix. Must include 'customer' and 'intercept' columns.

        Returns
        -------
        y_pred : array-like, shape (n_samples,)
            Predicted values.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # 1) Extract cluster IDs
        if 'customer' not in X.columns:
            raise ValueError("Input data must contain 'customer' column.")

        clusters_series = X['customer']
        if not isinstance(clusters_series, pd.Series):
            clusters_series = pd.Series(clusters_series.values, index=X.index)

        # 2) Extract intercept column
        if 'intercept' not in X.columns:
            raise ValueError("Input data must contain 'intercept' column.")

        intercept_series = X['intercept']
        if not np.all(intercept_series.values == 1):
            raise ValueError("Intercept column must be a constant value of 1.")

        # 3) Extract fixed-effect columns => exclude 'customer' and 'intercept'
        X_fixed = X.drop(columns=['customer', 'intercept'])

        return self.merf_model.predict(X_fixed.values, intercept_series.values, clusters_series)

    def get_params(self, deep=True):
        """Exposing hyperparameters for scikit-learn's GridSearchCV."""
        return {
            "gll_early_stop_threshold": self.gll_early_stop_threshold,
            "max_iterations": self.max_iterations,
            "rf__n_estimators": self.rf__n_estimators,
        }

    def set_params(self, **params):
        """
        Set parameters and reinitialize MERF model if necessary.

        Parameters
        ----------
        **params : dict
            Parameters to set.

        Returns
        -------
        self
        """
        for param, value in params.items():
            setattr(self, param, value)

        # Reinitialize MERF model if relevant parameters changed
        if any(k in params for k in ["gll_early_stop_threshold","max_iterations","rf__n_estimators"]):
            fe_model = RandomForestRegressor(n_estimators=self.rf__n_estimators, n_jobs=-1)
            self.merf_model = MERF(
                fixed_effects_model=fe_model,
                gll_early_stop_threshold=self.gll_early_stop_threshold,
                max_iterations=self.max_iterations
            )
        return self

