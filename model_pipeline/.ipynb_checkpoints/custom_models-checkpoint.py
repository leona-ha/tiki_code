# merf_wrapper_embed.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin,clone
from sklearn.ensemble import RandomForestRegressor
from merf.merf import MERF

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer

from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Embedding, Concatenate, Lambda, BatchNormalization, LayerNormalization,Dropout, Layer
from tensorflow.keras.models import Model
from sklearn.base import BaseEstimator, RegressorMixin
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping




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


##############################################################################
# Label Scaler
##############################################################################

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
####### PerUserFeatureImputer using KNN
##############################################################################



class PerUserFeatureScaler(BaseEstimator, TransformerMixin):
    """
    Scales numeric columns separately for each user.

    Parameters
    ----------
    user_col_index : int
        The column index where 'customer' resides in the post-ColumnTransformer output.
    numeric_indices : list of int
        The column indices for numeric features that we want to scale per user.
    strategy : str
        'minmax' for MinMaxScaler, otherwise StandardScaler.
    """
    def __init__(self, user_col_index, numeric_indices, strategy='standard'):
        self.user_col_index = user_col_index
        self.numeric_indices = numeric_indices
        self.strategy = strategy
    
        self.scalers_ = {}

        if strategy == 'minmax':
            self._scaler_class = MinMaxScaler
        else:
            self._scaler_class = StandardScaler

    def fit(self, X, y=None):
        unique_customers = np.unique(X[:, self.user_col_index])
        for customer in unique_customers:
            mask = (X[:, self.user_col_index] == customer)
            user_data = X[mask][:, self.numeric_indices]

            scaler = self._scaler_class()
            scaler.fit(user_data)
            self.scalers_[customer] = scaler


        return self

    def transform(self, X, y=None):
        X_out = X.copy()

        for customer, scaler in self.scalers_.items():
            mask = (X_out[:, self.user_col_index] == customer)
            if np.any(mask):
                user_data = X_out[mask][:, self.numeric_indices]
                X_out[mask][:, self.numeric_indices] = scaler.transform(user_data)

        # If there are unseen customers at predict-time, decide how to handle them (not shown).
        return X_out


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
        rf__n_estimators=100,  # Number of trees in the random forest
        rf__max_depth=5     # New hyperparameter to control the maximum depth of trees
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
        rf__max_depth : int or None
            Maximum depth of each tree in the random forest. Limiting the depth can reduce overfitting.
        """
        self.gll_early_stop_threshold = gll_early_stop_threshold
        self.max_iterations = max_iterations
        self.rf__n_estimators = rf__n_estimators
        self.rf__max_depth = rf__max_depth

        # Initialize MERF model with the updated RandomForestRegressor settings
        fe_model = RandomForestRegressor(
            n_estimators=self.rf__n_estimators, 
            max_depth=self.rf__max_depth, 
            n_jobs=-1
        )
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
    
        # 🔍 **Step 1: Detect 'customer' column**
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
    
        # Extract cluster IDs and intercept column
        clusters_series = X["cluster"]
        intercept_series = X["intercept"]
    
        # Extract fixed-effect columns => exclude 'cluster' and 'intercept'
        X_fixed = X.drop(columns=["cluster", "intercept"])
    
        # Ensure Z is 2D
        Z = intercept_series.values.reshape(-1, 1).astype(np.float64)
    
        X_fixed = X_fixed.astype(np.float64)
        y = np.array(y).astype(np.float64)
    
        self.merf_model.fit(X_fixed.values, Z, clusters_series, y)
    
        # Mark the model as fitted
        setattr(self, "is_fitted_", True)  
        return self

    def predict(self, X):
        """ Predict using the MERF model, handling unseen users. """
    
        # Check if the model is fitted
        check_is_fitted(self, "is_fitted_")
    
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
    
        # Detect and rename cluster/intercept columns
        cluster_col, intercept_col = self._detect_columns(X)
        X = X.rename(columns={cluster_col: "cluster", intercept_col: "intercept"})
    
        # Extract cluster IDs and intercept column
        clusters_series = X["cluster"]
        intercept_series = X["intercept"]
    
        # Extract fixed-effect columns
        X_fixed = X.drop(columns=["cluster", "intercept"]).astype(np.float64)
        Z = intercept_series.values.reshape(-1, 1).astype(np.float64)
        
        # Predict with MERF
        return self.merf_model.predict(X_fixed.values, Z, clusters_series)

    def get_params(self, deep=True):
        """Expose hyperparameters for GridSearchCV."""
        params = {
            "gll_early_stop_threshold": self.gll_early_stop_threshold,
            "max_iterations": self.max_iterations,
            "rf__n_estimators": self.rf__n_estimators,
            "rf__max_depth": self.rf__max_depth,
        }
        return params

    def set_params(self, **params):
        """
        Set hyperparameters and reinitialize MERF model if necessary.
        """
        for param, value in params.items():
            setattr(self, param, value)
    
        # Reinitialize MERF model if relevant parameters changed
        if any(k in params for k in ["gll_early_stop_threshold", "max_iterations", "rf__n_estimators", "rf__max_depth"]):
            fe_model = RandomForestRegressor(
                n_estimators=self.rf__n_estimators, 
                max_depth=self.rf__max_depth,
                n_jobs=-1
            )
            self.merf_model = MERF(
                fixed_effects_model=fe_model,
                gll_early_stop_threshold=self.gll_early_stop_threshold,
                max_iterations=self.max_iterations
            )
        return self


##############################################################################
# Keras Split
##############################################################################


class UnknownLabelEncoder:
    def __init__(self):
        self.classes_ = None
        self.class_to_index = {}
        self.unseen_mapping = {}
        self.unseen_counter = None

    def fit(self, y):
        # Assume y is a Pandas Series or list of training user IDs.
        unique = pd.Series(y).unique()
        self.classes_ = list(unique)
        self.class_to_index = {cls: i for i, cls in enumerate(self.classes_)}
        # Start unseen indices right after training ones.
        self.unseen_counter = len(self.classes_)
        return self

    def transform(self, y):
        # y is a list or array of user IDs.
        result = []
        for val in y:
            if val in self.class_to_index:
                result.append(self.class_to_index[val])
            else:
                # If this unseen value is already assigned, use that.
                if val in self.unseen_mapping:
                    result.append(self.unseen_mapping[val])
                else:
                    # Otherwise, assign a new unique index.
                    new_index = self.unseen_counter
                    self.unseen_mapping[val] = new_index
                    self.unseen_counter += 1
                    result.append(new_index)
        return np.array(result)



class SplitFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Splits a DataFrame into sensor features and user IDs.
    The user IDs are encoded using an UnknownLabelEncoder that assigns a special UNK index to unseen IDs.
    """
    def __init__(self, sensor_feature_cols=None, user_col=None):
        self.sensor_feature_cols = sensor_feature_cols  # If None, auto-detect all columns except user_col.
        self.user_col = user_col
        self.encoder = UnknownLabelEncoder()  # Our custom encoder

    def _detect_user_column(self, X):
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
        return [col for col in X.columns if col != self.user_col]

    def fit(self, X, y=None):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        if self.user_col is None or self.user_col not in X.columns:
            detected = self._detect_user_column(X)
            if detected is None:
                raise KeyError(f"[ERROR] User column '{self.user_col}' not found or detected.")
            self.user_col = detected

        if self.sensor_feature_cols is None:
            self.sensor_feature_cols = self._detect_sensor_columns(X)
        else:
            self.sensor_feature_cols = [c for c in self.sensor_feature_cols if c != self.user_col]

        # Store the final sensor columns for later use.
        self.fitted_sensor_feature_cols_ = self.sensor_feature_cols.copy()

        self.encoder.fit(X[self.user_col])
        return self

    def transform(self, X):
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X

        # Use the stored sensor columns
        sensor_data = X[self.fitted_sensor_feature_cols_].astype(float).values
        user_ids = self.encoder.transform(X[self.user_col])
        user_data = user_ids.reshape(-1, 1).astype(float)
        X_merged = np.hstack([sensor_data, user_data]).astype(float)
        return X_merged



##############################################################################
# Keras Regressor
##############################################################################



class SliceAndCastLayer(Layer):
    """
    A custom layer that slices the user IDs from column `sensor_feature_dim`
    and casts them to int32.
    """
    def __init__(self, sensor_feature_dim, **kwargs):
        super().__init__(**kwargs)
        self.sensor_feature_dim = sensor_feature_dim

    def call(self, inputs, training=None):
        # The 'inputs' is shape [None, sensor_feature_dim + 1]
        # We slice out inputs[:, sensor_feature_dim], then cast to int32
        user_ids = tf.cast(inputs[:, self.sensor_feature_dim], tf.int32)
        return user_ids




class KerasFFNNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 sensor_feature_dim=None,
                 num_users=None,  # Set to None to auto-detect from training data
                 embedding_dim=32,
                 hidden_units=(64, 32),
                 epochs=18,
                 batch_size=32,
                 learning_rate=1e-3,
                 dropout_rate=0.25,
                 l2_reg=0.0,
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
        self.l2_reg = l2_reg
        self.use_embedding = use_embedding
        self.verbose = verbose
        self.model_ = None

        if not self.use_embedding:
            self.embedding_dim = None

    def _build_model(self):
        if self.use_embedding:
            main_input = Input(shape=(self.sensor_feature_dim + 1,), name="merged_input")
    
            # sensor_data as before
            sensor_data = Lambda(
                lambda x: x[:, :self.sensor_feature_dim],
                output_shape=(self.sensor_feature_dim,)
            )(main_input)
    
            # user_ids with a custom layer
            user_ids = SliceAndCastLayer(self.sensor_feature_dim)(main_input)
    
            user_embedding = Embedding(input_dim=self.num_users,
                                       output_dim=self.embedding_dim,
                                       embeddings_initializer='he_normal')(user_ids)
            user_embedding = Flatten()(user_embedding)
            x = Concatenate()([sensor_data, user_embedding])
        else:
            main_input = Input(shape=(self.sensor_feature_dim,), name="sensor_input")
            x = main_input
    
        # Build fully connected layers.
        # Apply dropout only to the first two layers.
        for i, units in enumerate(self.hidden_units):
            x = Dense(units, activation="relu", kernel_initializer='he_normal')(x)
            x = LayerNormalization()(x)
            if i < 2:  # only add dropout for the first two layers
                x = Dropout(self.dropout_rate)(x)
                
        output = Dense(1, activation="linear")(x)
        model = Model(inputs=main_input, outputs=output)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])
        return model

    def fit(self, X, y):
        # Infer sensor_feature_dim if not provided
        if self.sensor_feature_dim is None:
            self.sensor_feature_dim = X.shape[1] - 1 if self.use_embedding else X.shape[1]
    
        # Auto-detect num_users from the last column if needed
        if self.num_users is None and self.use_embedding:
            if isinstance(X, np.ndarray):
                max_user_id = int(X[:, self.sensor_feature_dim].max())
            else:
                max_user_id = int(X.iloc[:, self.sensor_feature_dim].max())
            self.num_users = max_user_id + 1
            print(f"[DEBUG] Auto-detected num_users={self.num_users} from training data.")
    
        self.model_ = self._build_model()
        # Optionally, you can add early stopping here (uncomment if needed)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
        self.history_ = self.model_.fit(
            X, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            #validation_split=0.1,
            #callbacks=[early_stopping],
            verbose=self.verbose
        )
        return self

    def predict(self, X):
        return self.model_.predict(X, verbose=self.verbose).ravel()

    def get_config(self):
        """
        Return a dictionary of the hyperparameters.
        This is needed for serialization.
        """
        return {
            'sensor_feature_dim': self.sensor_feature_dim,
            'num_users': self.num_users,
            'embedding_dim': self.embedding_dim,
            'hidden_units': self.hidden_units,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg,
            'use_embedding': self.use_embedding,
            'verbose': self.verbose
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)