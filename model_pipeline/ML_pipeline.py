import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV,TimeSeriesSplit
from sklearn.base import BaseEstimator, TransformerMixin,RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer,OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer,IterativeImputer
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import BaseCrossValidator
import logging
import sys
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.model_selection import train_test_split
from custom_models import PerUserInterceptModel
import scipy.stats as st
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
import random
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)



# Configure logging for the module
logging.basicConfig(
    level=logging.DEBUG,  # Capture all logs; adjust as needed
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MLpipelineConfig")



##############################################################################
# CUSTOM SCORERS FOR MERF (if needed)
##############################################################################


def mae_scorer(y_true, y_pred):
    """Return negative MAE (since scikit-learn expects higher = better)."""
    return -mean_absolute_error(y_true, y_pred)

def rmse_scorer(y_true, y_pred):
    """Return negative RMSE."""
    return -np.sqrt(mean_squared_error(y_true, y_pred))

def r2_custom_scorer(y_true, y_pred):
    """Return R^2 as is (higher is better)."""
    return r2_score(y_true, y_pred)

def safe_log1p(X):
    # Ensure all values are > -1
    return np.log1p(np.clip(X, -0.999, None))


# We collect these into a dictionary recognized by scikit-learn:
custom_scorers = {
    "mae": make_scorer(mae_scorer, greater_is_better=False),
    "rmse": make_scorer(rmse_scorer, greater_is_better=False),
    "r2": make_scorer(r2_custom_scorer, greater_is_better=True),
}

def rename_holdout_columns(X_df, sensor_cols, user_col):
    """
    If X_df has integer columns [0..N-1], rename them back to the
    known sensor_cols + user_col. This is only for holdout/adaptation usage.
    """
    # Ensure X_df is a DataFrame
    X_df = pd.DataFrame(X_df) if not isinstance(X_df, pd.DataFrame) else X_df

    if all(isinstance(col, int) for col in X_df.columns):
        expected_num_cols = len(sensor_cols) + 1  # sensor columns + user column
        if X_df.shape[1] != expected_num_cols:
            raise ValueError(
                f"[rename_holdout_columns] Mismatch: expected {expected_num_cols} columns, "
                f"but got {X_df.shape[1]} for holdout data."
            )
        # The first len(sensor_cols) columns will become sensor_cols, and
        # the last column will become user_col
        new_cols = sensor_cols + [user_col]
        X_df.columns = new_cols

    return X_df


class DebugColumnsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, name="DebugColumns"):
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            print(f"[{self.name}] Shape: {X.shape}")
            print(f"[{self.name}] Columns: {X.columns.tolist()}")
            duplicates = X.columns[X.columns.duplicated()].tolist()
            if duplicates:
                print(f"[{self.name}] Duplicate columns detected: {duplicates}")
            else:
                print(f"[{self.name}] No duplicate columns.")
        else:
            print(f"[{self.name}] Array shape: {X.shape}")
        return X


##############################################################################
# FORWARD CHAINING CROSS-VALIDATOR
##############################################################################

class PerUserForwardChainingCV(BaseCrossValidator):
    """
    A custom cross-validator for user-level time series.
    Skips the first fold (fold 0) to avoid an empty training set.
    Yields n_splits - 1 splits.

    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. The actual number of yielded splits will be n_splits - 1.
    """

    def __init__(self, n_splits=5):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2 to perform cross-validation.")
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Returns the number of splits that will be yielded.
        Since fold 0 is skipped, it returns n_splits - 1.
        """
        return self.n_splits - 1

    def split(self, X, y=None, groups=None):
        if groups is None:
            raise ValueError("Groups (user IDs) are required for PerUserForwardChainingCV.")

        groups = np.array(groups)
        unique_users = np.unique(groups)

        logger.info(f"[PerUserForwardChainingCV] Unique users found: {len(unique_users)}")

        # Build user -> chunks
        user_chunks = {}
        for user in unique_users:
            user_indices = np.where(groups == user)[0]
            user_indices = np.sort(user_indices)
            n_samples = len(user_indices)

            if n_samples < self.n_splits:
                logger.warning(f"User '{user}' has only {n_samples} samples, less than n_splits={self.n_splits}. Assigning all to training.")
                user_chunks[user] = [user_indices]
                continue

            # Partition into n_splits chunks
            fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
            fold_sizes[: (n_samples % self.n_splits)] += 1

            starts = np.cumsum(fold_sizes)
            chunks = []
            prev = 0
            for end in starts:
                chunks.append(user_indices[prev:end])
                prev = end
            user_chunks[user] = chunks

            logger.debug(f"User '{user}' has {n_samples} samples divided into {len(chunks)} chunks.")

        # Yield splits, skipping fold 0
        for i in range(self.n_splits):
            if i == 0:
                logger.debug(f"Skipping fold {i} to avoid empty training set.")
                continue  # Skip fold 0

            train_idx_list = []
            test_idx_list = []

            for user in unique_users:
                chunks = user_chunks[user]
                if i < len(chunks):
                    test_idx_list.append(chunks[i])   # chunk i => test
                for j in range(i):
                    if j < len(chunks):
                        train_idx_list.append(chunks[j])  # chunks < i => train

            train_idx = np.concatenate(train_idx_list) if train_idx_list else np.array([], dtype=int)
            test_idx  = np.concatenate(test_idx_list)  if test_idx_list  else np.array([], dtype=int)

            # Skip fold if no training or test data
            if len(train_idx) == 0:
                logger.warning(f"Fold {i} has no training data => skipping fold.")
                continue
            if len(test_idx) == 0:
                logger.warning(f"Fold {i} has no test data => skipping fold.")
                continue

            logger.debug(f"Fold {i}: train_size={len(train_idx)}, test_size={len(test_idx)}")
            yield train_idx, test_idx



##############################################################################
# MAIN ML PIPELINE CLASS
##############################################################################

class MLpipeline:
    def __init__(self, config, random_state=42):
        self.cfg = config
        self.random_state = random_state
        self.df_all = None
        self.df_holdout = None
        self.df_known = None
        self.df_inner_train = None
        self.df_inner_test = None

        # Configure logger for MLpipeline
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs

        self.logger.info("[ML_Pipeline] Configuration Loaded:")
        self.logger.info(f"  Impute Strategy: {self.cfg.IMPUTE_STRATEGY}")
        self.logger.info(f"  Scaler Strategy: {self.cfg.SCALER_STRATEGY}")
        self.logger.info(f"  Holdout Ratio: {self.cfg.HOLDOUT_RATIO}")
        self.logger.info(f"  Time Ratio: {self.cfg.TIME_RATIO}")
        self.logger.info(f"  Number of Jobs: {self.cfg.N_JOBS}")
        self.logger.info(f"  Number of Folds (Inner CV): {self.cfg.N_INNER_CV}")
        self.logger.info(f"  CV Method: {self.cfg.CV_METHOD} (expected 'forwardchaining' here)")

    ##########################################################################
    # Basic Data Setup
    ##########################################################################

    def set_data(self, df):
        self.df_all = df.copy().reset_index(drop=True)
        self.logger.info(f"[set_data] DataFrame with {len(self.df_all)} rows loaded in pipeline.")
    

    def outer_user_split(self):
        """
        Splits users into holdout and known sets, ensuring stratification by `STRATIFY_COL`.
        
        - Uses user-level stratification based on `n_quest_binned`.
        - Guarantees exactly `HOLDOUT_RATIO` users go to holdout.
        
        Configurable options (`self.cfg.HOLDOUT_DATA_USAGE`):
            - "all": Use all data from holdout users.
            - "first_20": Use the first 20% of data per holdout user.
            - "last_20": Use the last 20% of data per holdout user.
        """
        stratify_col = self.cfg.STRATIFY_COL  # e.g., 'n_quest_binned'
        holdout_data_usage = self.cfg.HOLDOUT_DATA_USAGE  # One of ["all", "first_20", "last_20"]
        
        if stratify_col not in self.df_all.columns:
            raise ValueError(f"Stratify column '{stratify_col}' not found in dataset!")
        
        if holdout_data_usage not in ["all", "first_20", "last_20"]:
            raise ValueError("HOLDOUT_DATA_USAGE must be one of ['all', 'first_20', 'last_20'].")
    
        # ✅ Ensure stratify column is categorical
        self.df_all[stratify_col] = self.df_all[stratify_col].astype("category")
    
        # ✅ Aggregate stratify values per user
        user_stratify_values = self.df_all[[self.cfg.USER_COL, stratify_col]].drop_duplicates()
    
        # ✅ Compute exact holdout size
        unique_users = user_stratify_values[self.cfg.USER_COL].values
        stratify_values = user_stratify_values[stratify_col].values
    
        n_users = len(unique_users)
        n_holdout = int(np.floor(n_users * self.cfg.HOLDOUT_RATIO))
    
        # ✅ Ensure exact holdout size by manual selection
        if n_holdout >= 2:
            users_train, users_holdout = train_test_split(
                unique_users, test_size=n_holdout,  # Use absolute count instead of ratio
                stratify=stratify_values, random_state=self.random_state
            )
        else:
            raise ValueError(f"Too few users to stratify with holdout ratio {self.cfg.HOLDOUT_RATIO}.")
    
        # ✅ Create masks for holdout and known users
        mask_holdout = self.df_all[self.cfg.USER_COL].isin(users_holdout)
        df_holdout_all = self.df_all[mask_holdout].copy().reset_index(drop=True)
        df_known = self.df_all[~mask_holdout].copy().reset_index(drop=True)
    
        self.df_holdout = df_holdout_all  # Keep all data
    
        self.df_known = df_known
    
        self.logger.info(f"[outer_user_split] Holdout Users: {len(users_holdout)}/{n_users} | "
                         f"Holdout Strategy: {holdout_data_usage} | "
                         f"Holdout Size: {len(self.df_holdout)} rows.")



    def inner_time_split(self):
        """
        For each user in df_known, do a time-based split:
        first (TIME_RATIO)% => df_inner_train, last => df_inner_test.
        e.g. if TIME_RATIO=0.8 => 80% for 'inner_train' and 20% for 'inner_test'.
        """
        def split_by_time(df_user):
            df_sorted = df_user.sort_values(by=self.cfg.TIME_COL)
            cut = int(np.floor(len(df_sorted) * self.cfg.TIME_RATIO))
            return df_sorted.iloc[:cut], df_sorted.iloc[cut:]

        train_list, test_list = [], []
        for user, group in self.df_known.groupby(self.cfg.USER_COL):
            train_part, test_part = split_by_time(group)
            train_list.append(train_part)
            test_list.append(test_part)

        self.df_inner_train = pd.concat(train_list, axis=0).reset_index(drop=True)
        self.df_inner_test = pd.concat(test_list, axis=0).reset_index(drop=True)

        self.logger.info(f"[inner_time_split] Inner train size: {len(self.df_inner_train)}, "
                         f"test size: {len(self.df_inner_test)}.")

    ##########################################################################
    # CV Splits: Using PerUserForwardChainingCV or TimeSeriesSplit
    ##########################################################################

    def create_forwardchaining_cv(self):
        """
        Create a custom forward-chaining cross-validator for the 'inner_train' portion.
        Returns:
            cv: instance of PerUserForwardChainingCV
            df: the DataFrame used for training (df_inner_train)
        """
        cv = PerUserForwardChainingCV(n_splits=self.cfg.N_INNER_CV)
        self.logger.debug(f"Using PerUserForwardChainingCV with n_splits = {self.cfg.N_INNER_CV}")
        return cv, self.df_inner_train

    def create_forwardchaining_cv_test(self):
        """
        Create a TimeSeriesSplit cross-validator for testing purposes.
        Returns:
            cv: instance of TimeSeriesSplit
            df: the DataFrame used for training (df_inner_train)
        """
        try:
            cv = TimeSeriesSplit(n_splits=5)
            self.logger.debug(f"Using TimeSeriesSplit with n_splits = {cv.get_n_splits()}")
        except Exception as e:
            self.logger.error("Error initializing TimeSeriesSplit:")
            self.logger.error(e)
            raise
        return cv, self.df_inner_train

    ##########################################################################
    # Feature & Pipeline Setup
    ##########################################################################

    def assign_feature_columns(self, feature_cols, pipeline_name=None):
        """
        Assign feature columns for skewed, non-skewed, and categorical groups.
        Optionally include the USER_COL for MERF pipelines.
        """
        skewed_cols = [col for col in feature_cols if col in self.cfg.SKEWED_FEATURES]
        non_skewed_cols = [
            col for col in feature_cols
            if col in self.cfg.numeric_features and col not in skewed_cols
        ]
        cat_cols = [col for col in feature_cols if col in self.cfg.categorical_features]

        return skewed_cols, non_skewed_cols, cat_cols

    
    def _get_feature_cols(self, pipeline_name):
        """
        Get the feature columns for the given pipeline name.
    
        Ensures 'customer' (USER_COL) is included only for MERF pipelines.
        """
        base_features = (
            self.cfg.numeric_features +
            self.cfg.binary_features +
            self.cfg.categorical_features
        )
        # Include or exclude person_static_features based on pipeline type
        if "with_PS" in pipeline_name:
            feature_cols = list(set(base_features + self.cfg.person_static_features))
        else:
            feature_cols = list(set(base_features) - set(self.cfg.person_static_features))
    
        # Add MERF-specific columns for MERF pipelines
        if "MERF" in pipeline_name:
            feature_cols.extend(self.cfg.merf_cols)  
        elif ("PerUser" in pipeline_name) or ("Embeddings" in pipeline_name):
            feature_cols.append(self.cfg.USER_COL)

        print("features:",list(set(feature_cols)), flush=True)
        print("number of features:",len(list(set(feature_cols))), flush=True)

            
        # Return unique feature columns (no duplicates)
        return list(set(feature_cols))


    
    def build_preprocessor(self, feature_cols, pipeline_name):
        """
        Build a ColumnTransformer pipeline for numeric/categorical data.
        Passes only 'intercept' through without transformation.
        """
        # Define imputers based on configuration
    
        if self.cfg.IMPUTE_STRATEGY == "knn":
            imputer_numeric = KNNImputer(n_neighbors=5)
        elif self.cfg.IMPUTE_STRATEGY == "iterative":
            imputer_numeric =IterativeImputer(max_iter=15)
        else:
            imputer_numeric = SimpleImputer(strategy=self.cfg.IMPUTE_STRATEGY)
    
        # Define scalers based on configuration
        if self.cfg.SCALER_STRATEGY == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
    
        # Get feature groups
        skewed_cols, non_skewed_cols, cat_cols = self.assign_feature_columns(feature_cols, pipeline_name)
        actual_fixed_categories = [self.cfg.categorical_features_categories[col] for col in cat_cols]

    
        # Define pipelines for different feature types
        skewed_numeric_pipeline = Pipeline([
            ("impute", imputer_numeric),
            ("log_transform", FunctionTransformer(safe_log1p, validate=False)),
            ("varth", VarianceThreshold()),  # Apply VarianceThreshold here
            ("scale", scaler)
        ])
    
        non_skewed_numeric_pipeline = Pipeline([
            ("impute", imputer_numeric),
            ("varth", VarianceThreshold()),  # Apply VarianceThreshold here
            ("scale", scaler)
        ])
        categorical_pipeline = Pipeline([
            ("onehot", OneHotEncoder(categories=actual_fixed_categories, handle_unknown="ignore"))
        ])
            
        # Initialize list of transformers
        transformers = [
            ("skewed_num", skewed_numeric_pipeline, skewed_cols),
            ("non_skewed_num", non_skewed_numeric_pipeline, non_skewed_cols),
            ("cat", categorical_pipeline, cat_cols),
        ]

        # Pass through 'customer' and 'intercept' without transformation
        if "MERF" in pipeline_name:
            transformers.append(("customer_pass", "passthrough", ["customer"]))
            transformers.append(("intercept_pass", "passthrough", ["intercept"]))
      
        elif ("PerUser" in pipeline_name)or ("Embedding" in pipeline_name):
            transformers.append(("customer_pass", "passthrough", ["customer"]))

    
        # Instantiate the ColumnTransformer
        column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder='drop'  # Pass any columns not specified in transformers
        )
        
        preprocessor = Pipeline([
            ("column_transformer", column_transformer),
        ])
    
        return preprocessor
        
    def evaluate_holdout_all(self, results_timebased):
        """
        Evaluate trained models on the holdout users.
        
        Baseline Evaluation:
          For each model, for each holdout user, use the last 20% of that user's holdout data
          (based on TIME_COL) for evaluation.
        
        Adaptation Evaluation:
          For embedding pipelines (FFNN_with_Embeddings):
             - For each holdout user, sort by TIME_COL and split into:
                 Adaptation set: first 80%
                 Test set: last 20%
             - Transform these splits using a partial pipeline (all steps except the final keras_model).
             - Overwrite the last column (the user column) with the encoded value obtained
               from the fitted unknown label encoder. This encoder is assumed to return new unique indices
               (starting at len(training_users)) for unseen holdout users.
             - Clone the keras_model, freeze non‑embedding layers, fine‑tune on the adaptation set,
               then predict on the test set.
             - Compute per-user metrics.
          
          For intercept pipelines (PerUser_Intercept):
             - For each holdout user, sort by TIME_COL and split into:
                 Adaptation set: first 80%
                 Test set: last 20%
             - Use the mean of the adaptation set as the prediction.
             - Compute per-user metrics.
        
        Additionally, logs the number of unique holdout users.
        
        Returns
        -------
        holdout_results : list
            A list of dictionaries with baseline (holdout) scores per pipeline.
        adaptation_results : dict
            A dictionary mapping pipeline names to per‑user adaptation metrics.
            For each pipeline, this is a dict of {user_id: {r2, mae, rmse, n_adapt, n_test}}.
        """
    
        holdout_results = []
        adaptation_results = {}
    
        # Log number of unique holdout users.
        unique_users = self.df_holdout[self.cfg.USER_COL].nunique()
        self.logger.info(f"Number of unique holdout users: {unique_users}")
    
        # --- Baseline evaluation: for each user, take the last 20% of holdout data ---
        baseline_subsets = []
        for user, group in self.df_holdout.groupby(self.cfg.USER_COL):
            group_sorted = group.sort_values(by=self.cfg.TIME_COL)
            n = len(group_sorted)
            if n < 5:
                continue  # Skip users with too few samples.
            subset = group_sorted.iloc[-int(0.2 * n):]
            baseline_subsets.append(subset)
        if baseline_subsets:
            df_holdout_baseline = pd.concat(baseline_subsets, axis=0).reset_index(drop=True)
        else:
            self.logger.error("No valid holdout users for baseline evaluation.")
            df_holdout_baseline = pd.DataFrame()
    
        # --- Baseline evaluation for all models ---
        for model_result in results_timebased:
            pipeline_name = model_result["pipeline_name"]
            
            # _get_feature_cols returns string column names used during training.
            feature_cols = self._get_feature_cols(pipeline_name)
            self.logger.info(f"[{pipeline_name}] features: {feature_cols}")
            self.logger.info(f"[{pipeline_name}] number of features: {len(feature_cols)}")
            
            # For Embeddings pipelines, ensure the user column is included.
            if "Embeddings" in pipeline_name and self.cfg.USER_COL not in feature_cols:
                feature_cols.append(self.cfg.USER_COL)
            
            model = model_result["best_estimator"]
            
            # Baseline slice & predict:
            try:
                X_eval = df_holdout_baseline.loc[:, feature_cols]
                y_eval = df_holdout_baseline[self.cfg.LABEL_COL]
            except Exception as e:
                self.logger.error(f"[{pipeline_name}] Error extracting features from holdout baseline: {e}")
                continue
            
            try:
                y_pred = model.predict(X_eval).ravel()
                base_scores = {
                    "r2": r2_score(y_eval, y_pred),
                    "mae": mean_absolute_error(y_eval, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_eval, y_pred))
                }
                self.logger.info(f"[{pipeline_name}] Holdout (baseline) Scores: {base_scores}")
                holdout_results.append({
                    "pipeline_name": pipeline_name,
                    "holdout_scores": base_scores
                })
            except Exception as e:
                self.logger.error(f"[{pipeline_name}] Error during baseline evaluation: {e}")
                continue
            
            # --- Adaptation evaluation ---
            user_adapt_results = {}
            if "Embeddings" in pipeline_name:
                # For FFNN_with_Embeddings: use a partial pipeline (all steps except final keras_model)
                if len(model.steps) < 2:
                    self.logger.error(f"[{pipeline_name}] Not enough steps to build partial pipeline.")
                    adaptation_results[pipeline_name] = {}
                    continue
                partial_pipeline = Pipeline(model.steps[:-1])
                # Try to get the encoder from the split_features transformer.
                try:
                    user_encoder = model.named_steps["split_features"].encoder
                except Exception as e:
                    self.logger.error(f"[{pipeline_name}] Could not obtain encoder: {e}")
                    user_encoder = None
                
                for user_id, group in self.df_holdout.groupby(self.cfg.USER_COL):
                    group_sorted = group.sort_values(by=self.cfg.TIME_COL)
                    n = len(group_sorted)
                    self.logger.info(f"User {user_id} has {n} samples.")
                    if n < 5:
                        continue
                    split_idx = int(0.8 * n)
                    try:
                        X_adapt_raw = group_sorted.iloc[:split_idx].loc[:, feature_cols]
                        y_adapt = group_sorted.iloc[:split_idx][self.cfg.LABEL_COL]
                        X_test_raw = group_sorted.iloc[split_idx:].loc[:, feature_cols]
                        y_test = group_sorted.iloc[split_idx:][self.cfg.LABEL_COL]
                    except Exception as e:
                        self.logger.error(f"[{pipeline_name}] Error slicing adaptation data for user {user_id}: {e}")
                        continue
                    try:
                        X_adapt = partial_pipeline.transform(X_adapt_raw)
                        X_test = partial_pipeline.transform(X_test_raw)
                        self.logger.info(f"[{pipeline_name}] Transformed adaptation shape for user {user_id}: {X_adapt.shape}")
                        if isinstance(X_adapt, np.ndarray) and X_adapt.shape[1] > 0:
                            self.logger.info(f"[{pipeline_name}] First 5 values in transformed user column for user {user_id}: {X_adapt[:, -1][:5]}")
                        else:
                            self.logger.error(f"[{pipeline_name}] Transformed adaptation output is not a NumPy array.")
                    except Exception as e:
                        self.logger.error(f"[{pipeline_name}] Error during partial pipeline transform for user {user_id}: {e}")
                        continue
                    try:
                        # Use the encoder to obtain the encoded user index.
                        if user_encoder is not None:
                            unique_user_val = int(user_encoder.transform([user_id])[0])
                        else:
                            # Fallback if no encoder: you might want to raise an error.
                            unique_user_val = 143  
                        # Overwrite the last column with the encoded value.
                        X_adapt[:, -1] = unique_user_val
                        X_test[:, -1] = unique_user_val
                        self.logger.info(f"[{pipeline_name}] Overwrote user column with encoded value {unique_user_val} for user {user_id}.")
                    except Exception as e:
                        self.logger.error(f"[{pipeline_name}] Error overwriting user column for user {user_id}: {e}")
                        continue
                    try:
                        base_model = model.named_steps["keras_model"].model_
                        adapted_model = tf.keras.models.clone_model(base_model)
                        adapted_model.set_weights(base_model.get_weights())
                        # Freeze all layers except embedding layers.
                        for layer in adapted_model.layers:
                            if "embedding" not in layer.name.lower():
                                layer.trainable = False
                            self.logger.info(f"Layer: {layer.name}, trainable: {layer.trainable}")
                        adapted_model.compile(
                            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                            loss="mean_squared_error",
                            metrics=["mae"]
                        )
                        adapted_model.fit(
                            X_adapt, y_adapt,
                            epochs=20,
                            batch_size=model.named_steps["keras_model"].batch_size,
                            verbose=0
                        )
                        y_pred_adapt = adapted_model.predict(X_test).ravel()
                    except Exception as e:
                        self.logger.error(f"[{pipeline_name}] Error during adaptation for user {user_id}: {e}")
                        continue
                    user_scores = {
                        "r2": r2_score(y_test, y_pred_adapt),
                        "mae": mean_absolute_error(y_test, y_pred_adapt),
                        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_adapt)),
                        "n_adapt": len(X_adapt),
                        "n_test": len(X_test)
                    }
                    user_adapt_results[user_id] = user_scores
                adaptation_results[pipeline_name] = user_adapt_results
    
            elif "PerUser_Intercept" in pipeline_name:
                user_adapt_results = {}
                for user_id, group in self.df_holdout.groupby(self.cfg.USER_COL):
                    group_sorted = group.sort_values(by=self.cfg.TIME_COL)
                    n = len(group_sorted)
                    self.logger.info(f"User {user_id} has {n} samples.")
                    if n < 5:
                        continue
                    split_idx = int(0.8 * n)
                    try:
                        X_adapt = group_sorted.iloc[:split_idx][feature_cols]
                        y_adapt = group_sorted.iloc[:split_idx][self.cfg.LABEL_COL]
                        X_test = group_sorted.iloc[split_idx:][feature_cols]
                        y_test = group_sorted.iloc[split_idx:][self.cfg.LABEL_COL]
                    except Exception as e:
                        self.logger.error(f"[{pipeline_name}] Error slicing adaptation data for user {user_id}: {e}")
                        continue
                    try:
                        user_prediction = y_adapt.mean()
                        y_pred_adapt = np.full(y_test.shape, user_prediction)
                    except Exception as e:
                        self.logger.error(f"[{pipeline_name}] Error during adaptation for user {user_id}: {e}")
                        continue
                    user_scores = {
                        "r2": r2_score(y_test, y_pred_adapt),
                        "mae": mean_absolute_error(y_test, y_pred_adapt),
                        "rmse": np.sqrt(mean_squared_error(y_test, y_pred_adapt)),
                        "n_adapt": len(y_adapt),
                        "n_test": len(y_test)
                    }
                    user_adapt_results[user_id] = user_scores
                adaptation_results[pipeline_name] = user_adapt_results
    
            self.logger.info(f"Adaptation Results for {pipeline_name}: {adaptation_results.get(pipeline_name)}")
        
        return holdout_results, adaptation_results



    ##########################################################################
    # Main run: cross-validation + final refit
    ##########################################################################

    def run(self, pipeline_grid_dict, task_type="regression", scoring=custom_scorers, refit="mae", do_final_refit=True):
        self.logger.info("[run] Starting the ML pipeline run method.")
    
        if scoring is None:
            scoring = {
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2"
            }
    
        results_timebased = []
    
        # 1) Create the cross-validator on df_inner_train
        try:
            cv, train_df = self.create_forwardchaining_cv()
        except Exception as e:
            self.logger.error("Error creating cross-validator:")
            self.logger.error(e)
            raise
    
        # 2) Iterate over each pipeline configuration
        self.logger.debug("[run] Starting GridSearchCV for each pipeline.")
    
        for pipeline_name, (raw_pipeline, param_grid) in pipeline_grid_dict.items():
            self.logger.info(f"\n[run] Starting pipeline: {pipeline_name}")
    
            feature_cols = self._get_feature_cols(pipeline_name)
            self.logger.debug(f"[{pipeline_name}] Feature columns selected: {feature_cols}")
    
            X_train = train_df[feature_cols]
            y_train = train_df[self.cfg.LABEL_COL]
            y_train.index = train_df[self.cfg.USER_COL].values  # user_col as index
            groups = train_df[self.cfg.USER_COL]
    
            preprocessor = self.build_preprocessor(feature_cols, pipeline_name=pipeline_name)
            combined_pipeline = Pipeline([("preprocessor", preprocessor)] + list(raw_pipeline.steps))
    
            # Fit the preprocessor to see shape
            X_train_pre = preprocessor.fit_transform(X_train)
            df_preprocessed = pd.DataFrame(X_train_pre)
            print("First 5 rows of preprocessed features:")
            print(df_preprocessed.head(), flush=True)
    
            # GridSearch
            gs = GridSearchCV(
                estimator=combined_pipeline,
                param_grid=param_grid,
                scoring=scoring,
                refit=refit,
                cv=cv,
                n_jobs=self.cfg.N_JOBS,
                verbose=1,
                error_score='raise'
            )
    
            try:
                self.logger.info(f"[{pipeline_name}] Fitting GridSearchCV.")
                gs.fit(X_train, y_train, groups=groups)
                self.logger.info(f"[{pipeline_name}] GridSearchCV completed.")
                self.logger.info(f"[{pipeline_name}] Best Params: {gs.best_params_}")
                self.logger.info(f"[{pipeline_name}] Best CV Score ({refit}): {gs.best_score_:.3f}")
            except ValueError as e:
                self.logger.error(f"[ERROR] GridSearchCV failed for pipeline {pipeline_name}: {e}")
                continue
            except Exception as e:
                import traceback
                self.logger.error(f"[{pipeline_name}] [ERROR] GridSearchCV failed:")
                self.logger.error(e)
                print("Full traceback:")
                traceback.print_exc()
                continue
    
            # Evaluate on "inner test"
            X_inner_test = self.df_inner_test[feature_cols].copy()
            y_inner_test = self.df_inner_test[self.cfg.LABEL_COL]
    
            try:
                best_estimator = gs.best_estimator_
                y_pred = best_estimator.predict(X_inner_test)
            except Exception as e:
                self.logger.error(f"[{pipeline_name}] Error during prediction: {e}")
                continue
    
            try:
                test_scores = {
                    "r2": r2_score(y_inner_test, y_pred),
                    "mae": mean_absolute_error(y_inner_test, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_inner_test, y_pred)),
                }
                self.logger.info(f"[{pipeline_name}] Inner Test Scores: {test_scores}")
            except Exception as e:
                self.logger.error(f"[{pipeline_name}] Error computing test metrics: {e}")
                continue
    
            # Store
            results_timebased.append({
                "pipeline_name": pipeline_name,
                "best_cv_score": gs.best_score_,
                "inner_test_scores": test_scores,
                "best_estimator": gs.best_estimator_,
            })
    
        self.logger.info("[run] Done with GridSearch for all pipelines.")
    
        # Evaluate holdout
        holdout_results, adaptation_results = self.evaluate_holdout_all(results_timebased)
    
        # Return 3 things:
        # - results_timebased: Info about each pipeline's best CV
        # - holdout_results: baseline holdout metrics
        # - adaptation_results: per-user adaptation metrics for embedding pipelines
        return results_timebased, holdout_results, adaptation_results
