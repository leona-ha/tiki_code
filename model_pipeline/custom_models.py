# merf_wrapper_embed.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin,clone
from sklearn.ensemble import RandomForestRegressor
from merf.merf import MERF

from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted

from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score



class GlobalInterceptModel(BaseEstimator, RegressorMixin):
    """ Baseline model that always predicts the global mean of the target variable. """
    
    def fit(self, X, y):
        self.global_mean_ = np.mean(y)
        return self
    
    def predict(self, X):
        return np.full(shape=(X.shape[0],), fill_value=self.global_mean_)



class PerUserInterceptModel(BaseEstimator, RegressorMixin):
    """
    A model that predicts the per-user mean outcome.
    
    - Detects the user column automatically (like MERFWrapperEmbed).
    - Computes and stores per-user means based on training data.
    - If a user is unseen during training, predicts the global mean.
    """

    def __init__(self, cluster_col=None):
        """
        Parameters
        ----------
        user_column : str, optional
            Name of the user identifier column. If None, it will be detected.
        """
        self.cluster_col = cluster_col
        self.user_means_ = None
        self.global_mean_ = None

    def _detect_user_column(self, X):
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
            if X[col].dtype == "object" and X[col].apply(lambda v: isinstance(v, str) and len(v) == 4).all():
                cluster_col=col
                break  # Stop at the first match
        return cluster_col
        

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
    
        # 🔍 Detect and store user column name
        cluster_col = self._detect_user_column(X)
        X = X.rename(columns={cluster_col: "cluster"})
    
        # ✅ Reset index for alignment
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
    
        # ✅ Merge to maintain user alignment
        df_train = pd.concat([X, y], axis=1)
    
        # ✅ Compute per-user mean
        self.user_means_ = df_train.groupby("cluster")[y.name].mean()
        self.global_mean_ = df_train[y.name].mean()  # Global fallback for unseen users
    
        print(f"✅ Fit completed. Detected {len(self.user_means_)} unique users.")
    
        return self



    def predict(self, X):
        """
        Predicts using stored per-user means.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (must contain user column).
        
        Returns
        -------
        y_pred : np.array
            Predictions based on per-user means.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # ✅ Ensure user column is detected
        cluster_col =  self._detect_user_column(X)
        X = X.rename(columns={cluster_col: "cluster"})

        # ✅ Generate predictions
        y_pred = X["cluster"].map(self.user_means_).fillna(self.global_mean_).values

        return y_pred



class PerUserLabelScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        # This dictionary will hold the computed mean and std per user.
        self.user_stats_ = {}

    def fit(self, y, user_ids):
        # y: array-like of labels; user_ids: array-like of corresponding user IDs.
        df = pd.DataFrame({'y': y, 'user': user_ids})
        # Compute mean and std for each user.
        stats = df.groupby('user')['y'].agg(['mean', 'std'])
        # Replace std of 0 (or NaN) with 1 to avoid division by zero.
        stats['std'] = stats['std'].replace(0, 1).fillna(1)
        self.user_stats_ = stats.to_dict('index')
        return self

    def transform(self, y, user_ids):
        y_scaled = []
        for val, user in zip(y, user_ids):
            if user in self.user_stats_:
                mean = self.user_stats_[user]['mean']
                std = self.user_stats_[user]['std']
            else:
                # If the user wasn’t seen during fit, fall back to no scaling.
                mean, std = 0.0, 1.0
            y_scaled.append((val - mean) / std)
        return np.array(y_scaled)

    def inverse_transform(self, y_scaled, user_ids):
        y_inv = []
        for val, user in zip(y_scaled, user_ids):
            if user in self.user_stats_:
                mean = self.user_stats_[user]['mean']
                std = self.user_stats_[user]['std']
            else:
                mean, std = 0.0, 1.0
            y_inv.append(val * std + mean)
        return np.array(y_inv)


class PerUserTransformedTargetRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, regressor, transformer, user_col=None, drop_user=True):
        """
        Parameters:
            regressor: Underlying regressor (e.g., LinearRegression, MERFWrapperEmbed, etc.)
            transformer: An instance of a per-user target transformer (e.g., PerUserLabelScaler)
            user_col: Name of the user column (e.g., "customer"). If None, it will be auto-detected.
            drop_user: Whether to drop the user column from X before fitting/predicting.
                       Set to False for MERF pipelines where the user column is required.
        """
        self.regressor = regressor
        self.transformer = transformer
        self.user_col = user_col
        self.drop_user = drop_user

    def _ensure_dataframe(self, X):
        """Converts X to a DataFrame if it is not one already."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X

    def _detect_user_column(self, X):
        """
        Detects the user column automatically. For example, it might check for
        an object column where all values are strings of a given length.
        Adjust this logic as needed.
        """
        for col in X.columns:
            # This heuristic looks for a column of strings with length 4.
            if X[col].dtype == "object" and X[col].apply(lambda v: isinstance(v, str) and len(v) == 4).all():
                return col
        if self.user_col is not None:
            return self.user_col
        raise ValueError("User column could not be detected automatically and was not provided.")

    def fit(self, X, y):
        X = self._ensure_dataframe(X)
        # Determine the user column: either use the provided one or detect automatically.
        user_col = self.user_col if (self.user_col is not None and self.user_col in X.columns) else self._detect_user_column(X)
        
        # Extract user IDs
        user_ids = X[user_col].values

        # Fit the transformer on the target using these user IDs.
        self.transformer_ = clone(self.transformer)
        self.transformer_.fit(y, user_ids)
        y_trans = self.transformer_.transform(y, user_ids)

        # Depending on the flag, drop the user column from the features before fitting.
        if self.drop_user:
            X_model = X.drop(columns=[user_col])
        else:
            X_model = X

        # Fit the underlying regressor on the modified X.
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X_model, y_trans)
        return self

    def predict(self, X):
        X = self._ensure_dataframe(X)
        # Determine the user column.
        user_col = self.user_col if (self.user_col is not None and self.user_col in X.columns) else self._detect_user_column(X)
        user_ids = X[user_col].values

        # Depending on the flag, drop the user column before prediction.
        if self.drop_user:
            X_model = X.drop(columns=[user_col])
        else:
            X_model = X

        y_pred_trans = self.regressor_.predict(X_model)
        # Inverse-transform the predictions to return them to the original scale.
        y_pred = self.transformer_.inverse_transform(y_pred_trans, user_ids)
        return y_pred

    def score(self, X, y):
        return r2_score(y, self.predict(X))



##############################################################################
# MERF Wrapper
##############################################################################



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
        """ Predict using the MERF model, handling unseen users. """
    
        # 🚨 Check if the model is fitted
        check_is_fitted(self, "is_fitted_")
    
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
    
        # Detect and rename cluster/intercept columns
        cluster_col, intercept_col = self._detect_columns(X)
        X = X.rename(columns={cluster_col: "cluster", intercept_col: "intercept"})
    
        # Extract cluster IDs
        clusters_series = X["cluster"]
    
        # Extract intercept column (random effects input)
        intercept_series = X["intercept"]
    
        # Extract fixed-effect columns => exclude 'cluster' and 'intercept'
        X_fixed = X.drop(columns=["cluster", "intercept"]).astype(np.float64)
        Z = intercept_series.values.reshape(-1, 1).astype(np.float64)

        
        # Predict with MERF (handles unseen users by default)
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



