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
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Concatenate
from tensorflow.keras.models import Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Embedding, Concatenate, Lambda, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from sklearn.base import BaseEstimator, RegressorMixin




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
                print("User was not seen during fit! No Scaling fallback.", flush=True)
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

##############################################################################
# PersonEmbedding
##############################################################################


class SplitFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Splits a DataFrame into a list for a Keras model that expects two inputs:
      - sensor_input: numeric sensor features (all columns except the user column)
      - user_input: encoded user IDs

    If sensor_feature_cols is not provided, it will automatically select all columns
    except the user column. This transformer uses a heuristic _detect_user_column
    if the specified user_col is not actually found in X.columns.
    """
    def __init__(self, sensor_feature_cols=None, user_col=None):
        self.sensor_feature_cols = sensor_feature_cols  # optional; if None, auto-detect
        self.user_col = user_col
        self.encoder = LabelEncoder()
        
    def _detect_user_column(self, X):
        """
        Attempts to detect candidate user columns using a heuristic:
          - Look for a column whose values are all strings of exactly 4 characters
            and containing at least one alphabetic character.
        If multiple candidate columns are detected, it prints a warning and returns the first one.
        """
        # Find all columns matching the heuristic
        candidate_cols = [
            col for col in X.columns 
            if X[col].apply(lambda v: isinstance(v, str) and len(v) == 4 and any(c.isalpha() for c in v)).all()
        ]
        
        if not candidate_cols:
            return None
        elif len(candidate_cols) > 1:
            print(f"[DEBUG] Multiple user columns detected: {candidate_cols}. Using '{candidate_cols[0]}' as user column.")
            return candidate_cols[0]
        else:
            return candidate_cols[0]


    def _detect_sensor_columns(self, X):
        """Return all columns except the user column (which might be numeric or string)."""
        return [col for col in X.columns if col != self.user_col]

    def fit(self, X, y=None):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        # 1) If the user_col isn't found in X.columns, try to detect it.
        if self.user_col not in X.columns:
            detected = self._detect_user_column(X)
            if detected is None:
                raise KeyError(f"[ERROR] User column '{self.user_col}' not found or detected.")
            self.user_col = detected

        # 2) Now that we know the user column, detect or filter out sensor columns
        if self.sensor_feature_cols is None:
            # auto-detect => all columns except the user column
            self.sensor_feature_cols = self._detect_sensor_columns(X)
        else:
            # even if sensor_feature_cols is given, remove the user column if present
            self.sensor_feature_cols = [c for c in self.sensor_feature_cols if c != self.user_col]


        # 3) Fit the LabelEncoder on user column
        self.encoder.fit(X[self.user_col])
        return self

    def transform(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        sensor_data = X[self.sensor_feature_cols].astype(float).values
        user_ids = self.encoder.transform(X[self.user_col])
        user_data = user_ids.reshape(-1, 1).astype(float)
        
        X_merged = np.hstack([sensor_data, user_data]).astype(float)
        
        return X_merged



# --- NEW: FFNN Model with Embedding Layer ---



class KerasFFNNRegressor(BaseEstimator, RegressorMixin):
    """
    A custom scikit-learn estimator wrapping a Keras FFNN.
    
    When use_embedding=True, it expects sensor features plus a user column (and applies an embedding).
    When use_embedding=False, it expects only sensor features.
    
    If sensor_feature_dim is set to None, the number of input features will be determined from X in fit().
    """
    def __init__(self,
                 sensor_feature_dim=None,  # Allow None so we can determine input shape at fit time.
                 num_users=158,
                 embedding_dim=32,
                 hidden_units=(64, 32),
                 epochs=20,
                 batch_size=32,
                 learning_rate=1e-3,
                 dropout_rate=0.25,
                 use_embedding=True,
                 verbose=0):
        self.sensor_feature_dim = sensor_feature_dim
        self.num_users = num_users
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.use_embedding = use_embedding
        self.verbose = verbose
        self.model_ = None

        # If embeddings are not used, the embedding_dim parameter is irrelevant.
        if not self.use_embedding:
            self.embedding_dim = None

    def _build_model(self):
        if self.use_embedding:
            # Expect input shape (sensor_feature_dim + 1,), where the last column is the user ID.
            main_input = Input(shape=(self.sensor_feature_dim + 1,), name="merged_input")
            sensor_data = Lambda(lambda x: x[:, :self.sensor_feature_dim])(main_input)
            user_ids = Lambda(lambda x: tf.cast(x[:, self.sensor_feature_dim], tf.int32))(main_input)
            user_embedding = Embedding(input_dim=self.num_users,
                                       output_dim=self.embedding_dim,
                                       embeddings_initializer='he_normal')(user_ids)
            user_embedding = Flatten()(user_embedding)
            x = Concatenate()([sensor_data, user_embedding])
        else:
            # Expect input shape (sensor_feature_dim,) only.
            main_input = Input(shape=(self.sensor_feature_dim,), name="sensor_input")
            x = main_input

        # Build fully connected layers.
        for units in self.hidden_units:
            x = Dense(units, activation="relu", kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout_rate)(x)
        output = Dense(1, activation="linear")(x)
        model = Model(inputs=main_input, outputs=output)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mean_squared_error")
        return model

    def fit(self, X, y):
        # If sensor_feature_dim is None, infer it from the data X.
        if self.sensor_feature_dim is None:
            self.sensor_feature_dim = X.shape[1] if not self.use_embedding else X.shape[1] - 1
            # Explanation:
            # - For use_embedding=True, we expect X to have sensor features + one extra column (user IDs),
            #   so sensor_feature_dim = total columns - 1.
            # - For use_embedding=False, X is assumed to contain only sensor features.
        self.model_ = self._build_model()
        self.model_.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose
        )
        return self

    def predict(self, X):
        return self.model_.predict(X, verbose=self.verbose).ravel()
