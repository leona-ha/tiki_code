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
from sklearn.utils.validation import check_is_fitted


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
    A MERF wrapper that automatically detects and renames the cluster ('customer') 
    and intercept column, making it more robust to input variations.
    """

    def __init__(
        self,
        gll_early_stop_threshold=0.01, 
        max_iterations=5,
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

    def _detect_columns(self, X):
        """
        Detects 'customer' (cluster) and 'intercept' columns.
    
        - 'customer' is detected as a column with **letters and exactly 4 characters**.
        - 'intercept' is detected as a **constant column with all values = 1**.
        - Additionally, detects **any column containing non-numeric values like 'NaN', 'None', or strings**.
    
        Returns
        -------
        cluster_col : str
            Name of detected 'customer' column.
        intercept_col : str
            Name of detected 'intercept' column.
        """
        X = X.copy()  # Avoid modifying the original DataFrame
    
        # 🔍 **Step 1: Detect 'customer' column (contains letters + exactly 4-character strings)**
        cluster_col = None
        for col in X.columns:
            if X[col].apply(lambda v: isinstance(v, str) and len(v) == 4 and any(c.isalpha() for c in v)).all():
                cluster_col = col
                break  # Stop at the first match
    
        # 🔍 **Step 2: Detect 'intercept' column (all values == 1)**
        intercept_col = None
        for col in X.columns:
            if np.all(X[col].astype(str) == "1"):
                intercept_col = col
                break  # Stop at the first match
    
        # 🔍 **Step 3: Detect columns containing "NaN", "None", or other non-numeric values**
        non_numeric_cols = []
        for col in X.columns:
            if X[col].astype(str).str.lower().isin(["nan", "none", "null", "inf", "-inf"]).any():
                non_numeric_cols.append(col)
    
    
        return cluster_col, intercept_col
        

    def fit(self, X, y):
        """ Fit the MERF model. """
    
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
    
        # Detect and rename cluster/intercept columns
        cluster_col, intercept_col = self._detect_columns(X)
        X = X.rename(columns={cluster_col: "cluster", intercept_col: "intercept"})
    
        # Extract cluster IDs
        clusters_series = X["cluster"]
    
        # Extract intercept column
        intercept_series = X["intercept"]
    
        # Extract fixed-effect columns => exclude 'cluster' and 'intercept'
        X_fixed = X.drop(columns=["cluster", "intercept"])
    
        # 🚀 Ensure Z is 2D
        Z = intercept_series.values.reshape(-1, 1).astype(np.float64)
    
        # 🚀 Strict Type Conversion
        X_fixed = X_fixed.astype(np.float64)
        Z = Z.astype(np.float64)
        y = np.array(y).astype(np.float64)

    
        # 🚀 Fit MERF
        self.merf_model.fit(X_fixed.values, Z, clusters_series, y)
    
        # ✅ Mark the model as fitted using sklearn's standard API
        setattr(self, "is_fitted_", True)  
    
        return self

    def predict(self, X):
        """ Predict using the MERF model. """
    
        # 🚨 Check if the model is fitted
        check_is_fitted(self, "is_fitted_")
    
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
    
        # Detect and rename cluster/intercept columns
        cluster_col, intercept_col = self._detect_columns(X)
        X = X.rename(columns={cluster_col: "cluster", intercept_col: "intercept"})
    
        # Extract cluster IDs
        clusters_series = X["cluster"]
    
        # Extract intercept column
        intercept_series = X["intercept"]
    
        # Extract fixed-effect columns => exclude 'cluster' and 'intercept'
        X_fixed = X.drop(columns=["cluster", "intercept"])
    
    
        Z = intercept_series.values.reshape(-1, 1).astype(np.float64)
    
        # **Strict Type Conversion**
        Z = Z.astype(np.float64)
        X_fixed = X_fixed.astype(np.float64)
    
        return self.merf_model.predict(X_fixed.values, Z, clusters_series)



    def get_params(self, deep=True):
        """Expose hyperparameters for GridSearchCV."""
        
        # 🚀 Ensure the model is trained before allowing GridSearchCV to score
        if not hasattr(self.merf_model, "trained_b"):
            print("❌ WARNING: GridSearchCV is trying to access an untrained MERF model!")
    
        params = {
            "gll_early_stop_threshold": self.gll_early_stop_threshold,
            "max_iterations": self.max_iterations,
            "rf__n_estimators": self.rf__n_estimators,
        }
        return params

    
    def set_params(self, **params):
        """
        Set hyperparameters and reinitialize MERF model if necessary.
        """
    
        for param, value in params.items():
            setattr(self, param, value)
    
        # Reinitialize MERF model if relevant parameters changed
        if any(k in params for k in ["gll_early_stop_threshold", "max_iterations", "rf__n_estimators"]):
            fe_model = RandomForestRegressor(n_estimators=self.rf__n_estimators, n_jobs=-1)
            self.merf_model = MERF(
                fixed_effects_model=fe_model,
                gll_early_stop_threshold=self.gll_early_stop_threshold,
                max_iterations=self.max_iterations
            )
        return self



