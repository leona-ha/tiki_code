import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
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
import scipy.stats as st
from custom_models import PerUserFeatureScaler,LMERWrapper,MiceForestImputer
from tensorflow.keras.models import clone_model
import sklearn


# Configure logging for the module
logging.basicConfig(
    level=logging.DEBUG,  # Capture all logs; adjust as needed
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MLpipelineConfig")


class DropCustomerColumn(BaseEstimator, TransformerMixin):
    """
    Drops the column at the given index (e.g., the customer column)
    so that the final estimator only sees numeric data.
    """
    def __init__(self, customer_col_index):
        self.customer_col_index = customer_col_index
        
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.delete(X, self.customer_col_index, axis=1)


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



# We collect these into a dictionary recognized by scikit-learn:
custom_scorers = {
    "mae": make_scorer(mae_scorer, greater_is_better=False),
    "rmse": make_scorer(rmse_scorer, greater_is_better=False),
    "r2": make_scorer(r2_custom_scorer, greater_is_better=True),
}


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
    
        # ✅ Handle different holdout data usage strategies
        if holdout_data_usage == "all":
            self.df_holdout = df_holdout_all  # Keep all data
        else:
            holdout_subsets = []
            for _, group in df_holdout_all.groupby(self.cfg.USER_COL):
                group = group.sort_values(by=self.cfg.TIME_COL)  # Sort by time
    
                if holdout_data_usage == "first_20":
                    subset = group.iloc[:int(0.2 * len(group))]  # First 20%
                elif holdout_data_usage == "last_20":
                    subset = group.iloc[-int(0.2 * len(group)):]  # Last 20%
    
                holdout_subsets.append(subset)
    
            self.df_holdout = pd.concat(holdout_subsets, axis=0).reset_index(drop=True)
    
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
        if ("MERF" in pipeline_name) or ("LMER" in pipeline_name):
            feature_cols.extend(self.cfg.merf_cols)  
        elif ("PUP" in pipeline_name) or ("PerUser" in pipeline_name) or ("Embeddings" in pipeline_name):
            feature_cols.append(self.cfg.USER_COL)


        return list(set(feature_cols))
        
    def build_preprocessor(self, feature_cols, pipeline_name):
        """
        Build a ColumnTransformer pipeline for numeric/categorical data.
        Passes only 'intercept' or 'customer' through without transformation (depending on pipeline_name).
        """
    
        # --- [1] Define imputers based on configuration ---------------
        if self.cfg.IMPUTE_STRATEGY == "knn":
            imputer_numeric = KNNImputer(n_neighbors=5)
        elif self.cfg.IMPUTE_STRATEGY == "iterative":
            imputer_numeric = IterativeImputer(max_iter=30, random_state=42)
        elif self.cfg.IMPUTE_STRATEGY == "mice":
            imputer_numeric = MiceForestImputer(iterations=10, random_state=42)
        else:
            imputer_numeric = SimpleImputer(strategy=self.cfg.IMPUTE_STRATEGY)
    
        # --- [2] Define scalers based on configuration ---------------
        if self.cfg.SCALER_STRATEGY == "minmax":
            scaler = MinMaxScaler()
            scale_strategy = "minmax"
        else:
            scaler = StandardScaler()
            scale_strategy = "standard"
    
        # --- [3] Get feature groups ----------------------------------
        skewed_cols, non_skewed_cols, cat_cols = self.assign_feature_columns(feature_cols, pipeline_name)
        actual_fixed_categories = [self.cfg.categorical_features_categories[col] for col in cat_cols]
    
        # --- [4] Define pipelines for different feature types --------
        if "PUP" in pipeline_name:
            skewed_numeric_pipeline = Pipeline([
                ("impute", imputer_numeric),
    #            ("log_transform", FunctionTransformer(np.log1p, validate=False)),
                ("varth", VarianceThreshold())
            ])
            non_skewed_numeric_pipeline = Pipeline([
                ("impute", imputer_numeric),
                ("varth", VarianceThreshold())
            ])
        else:
            # The original approach: apply the global scaler here
            skewed_numeric_pipeline = Pipeline([
                ("impute", imputer_numeric),
                ("log_transform", FunctionTransformer(np.log1p, validate=False)),
                ("varth", VarianceThreshold()),
                ("scale", scaler)
            ])
            non_skewed_numeric_pipeline = Pipeline([
                ("impute", imputer_numeric),
                ("varth", VarianceThreshold()),
                ("scale", scaler)
            ])
    
        # Categorical pipeline remains the same
        categorical_pipeline = Pipeline([
            ("onehot", OneHotEncoder(categories=actual_fixed_categories, handle_unknown="ignore"))
        ])
    
        transformers = [
            ("skewed_num", skewed_numeric_pipeline, skewed_cols),
            ("non_skewed_num", non_skewed_numeric_pipeline, non_skewed_cols),
            ("cat", categorical_pipeline, cat_cols),
        ]
    
        # --- [6] Pass through 'customer' and/or 'intercept' exactly as in original code
        if "MERF" in pipeline_name:
            transformers.append(("customer_pass", "passthrough", ["customer"]))
            transformers.append(("intercept_pass", "passthrough", ["intercept"]))
        elif "LMER" in pipeline_name:
            transformers.append(("customer_pass", "passthrough", ["customer"]))
            transformers.append(("intercept_pass", "passthrough", ["intercept"]))
    
        elif ("PerUser" in pipeline_name) or ("Embeddings" in pipeline_name):
            transformers.append(("customer_pass", "passthrough", ["customer"]))
    
        elif "PUP" in pipeline_name: 
            transformers.append(("customer_pass", "passthrough", ["customer"]))
    
        column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )

        if "PUP" in pipeline_name:
            all_numeric_cols = skewed_cols + non_skewed_cols
            numeric_dim = len(all_numeric_cols)
    
            cat_dim = sum(len(cats) for cats in actual_fixed_categories)
            user_col_index = numeric_dim + cat_dim
    
            numeric_indices = list(range(numeric_dim))
    
            per_user_scaler = PerUserFeatureScaler(
                user_col_index=user_col_index,
                numeric_indices=numeric_indices,
                strategy=scale_strategy  # (minmax or standard)
            )
    
            if any(x in pipeline_name for x in ["MERF","LMER", "Embeddings", "PerUser"]):
                # For pipelines that need the customer column:
                preprocessor = Pipeline([
                    ("column_transformer", column_transformer),
                    ("per_user_scaler", per_user_scaler)

                ])
            else:
                # For pipelines that do not need the customer column:
                preprocessor = Pipeline([
                    ("column_transformer", column_transformer),
                    ("per_user_scaler", per_user_scaler),
                    ("drop_customer", DropCustomerColumn(customer_col_index=user_col_index))
                ])

    
        else:
            preprocessor = Pipeline([
                ("column_transformer", column_transformer),
            ])
    
        return preprocessor

    ##########################################################################
    # Main run: hold out
    ##########################################################################

    def evaluate_holdout_embedding_adaptation(self, model, X_holdout, y_holdout):
        """
        For models using embeddings, adapt the embedding layer for new users
        based on the first 80% of each user's holdout data.
        For new (unseen) users, instantiate a baseline prediction (here: global mean).
        
        Parameters
        ----------
        model : trained Keras model (with embedding)
        X_holdout : DataFrame of holdout features.
        y_holdout : Series of holdout targets.
        
        Returns
        -------
        adapted_results : dict
            A dictionary mapping user id to their evaluation metrics.
        """
        adapted_results = {}
        
        # Determine known users from training data.
        known_users = set(self.df_known[self.cfg.USER_COL].unique())
        # Use global mean of the target from known users as the baseline value.
        baseline_value = self.df_known[self.cfg.LABEL_COL].mean()
        
        # Group holdout data by user
        for user, group in X_holdout.groupby(self.cfg.USER_COL):
            group = group.sort_values(by=self.cfg.TIME_COL)
            n = len(group)
            if n < 5:
                # Skip users with too few data points.
                continue
            
            split_idx = int(0.8 * n)
            X_adapt = group.iloc[:split_idx]
            X_eval = group.iloc[split_idx:]
            y_adapt = y_holdout.loc[X_adapt.index]
            y_eval = y_holdout.loc[X_eval.index]
            
            # If this user was never seen during training, use the baseline.
            if user not in known_users:
                y_pred_baseline = np.full(len(X_eval), baseline_value)
                r2_baseline = r2_score(y_eval, y_pred_baseline)
                mae_baseline = mean_absolute_error(y_eval, y_pred_baseline)
                rmse_baseline = np.sqrt(mean_squared_error(y_eval, y_pred_baseline))
                adapted_results[user] = {
                    "baseline": True,
                    "n_adapt": 0,
                    "n_eval": len(X_eval),
                    "r2": r2_baseline,
                    "mae": mae_baseline,
                    "rmse": rmse_baseline,
                    "y_true": y_eval.values,
                    "y_pred": y_pred_baseline,
                }
                self.logger.info(f"[Holdout Adaptation] New user baseline for user: {user} | "
                                 f"n_eval: {len(X_eval)} | r2: {r2_baseline:.3f} | mae: {mae_baseline:.3f} | rmse: {rmse_baseline:.3f}")
                continue
            
            # Otherwise, adapt the embedding layer.
            adapted_model = clone_model(model)
            adapted_model.set_weights(model.get_weights())
            # Freeze all layers except the embedding layer (assuming its name is 'user_embedding').
            for layer in adapted_model.layers:
                if layer.name != 'user_embedding':
                    layer.trainable = False
                else:
                    self.logger.info(f"Adapting embedding layer for user {user}.")
            
            adapted_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss=model.loss,
                metrics=["mae"]
            )
            adapted_model.fit(
                X_adapt, y_adapt,
                epochs=5,
                batch_size=32,
                verbose=0
            )
            y_pred_adapt = adapted_model.predict(X_eval).ravel()
            r2_adapt = r2_score(y_eval, y_pred_adapt)
            mae_adapt = mean_absolute_error(y_eval, y_pred_adapt)
            rmse_adapt = np.sqrt(mean_squared_error(y_eval, y_pred_adapt))
            
            adapted_results[user] = {
                "baseline": False,
                "n_adapt": len(X_adapt),
                "n_eval": len(X_eval),
                "r2": r2_adapt,
                "mae": mae_adapt,
                "rmse": rmse_adapt,
                "y_true": y_eval.values,
                "y_pred": y_pred_adapt,
            }
            self.logger.info(f"[Holdout Adaptation] User: {user} | n_adapt: {len(X_adapt)} | "
                             f"n_eval: {len(X_eval)} | r2: {r2_adapt:.3f} | mae: {mae_adapt:.3f} | rmse: {rmse_adapt:.3f}")
        
        return adapted_results

    def evaluate_holdout_all(self, results_timebased):
        """
        Evaluate trained models on the holdout users and return two sets of results:
        
        1. Unadapted holdout results for all models (including FFNN+Embedding, where new users
           get baseline predictions if not seen during training).
        2. Adapted results for FFNN+Embedding and person-static intercept models:
           For each holdout user, if the user was seen during training, adapt the model (fine-tune
           the embedding layer or intercept) on the first 80% of that user's data and then evaluate on the remaining 20%.
        
        Returns
        -------
        holdout_results : list
            Overall evaluation scores from unadapted predictions.
        adaptation_results : dict
            For applicable pipelines (e.g., with "Embedding" or "Intercept" in their name),
            a dict mapping pipeline names to per-user adaptation metrics.
        """
        holdout_results = []
        adaptation_results = {}  # To store adaptation (fine-tuning) results for applicable pipelines
        
        # Ensure holdout data is available
        if self.df_holdout is None or len(self.df_holdout) == 0:
            self.logger.warning("No holdout data available for evaluation.")
            return holdout_results, adaptation_results
    
        for model_result in results_timebased:
            pipeline_name = model_result["pipeline_name"]
            feature_cols = self._get_feature_cols(pipeline_name)
            X_holdout = self.df_holdout[feature_cols]
            y_holdout = self.df_holdout[self.cfg.LABEL_COL]
            model = model_result["best_estimator"]
    
            try:
                # Unadapted predictions for all models:
                y_pred = model.predict(X_holdout)
                scores = {
                    "r2": r2_score(y_holdout, y_pred),
                    "mae": mean_absolute_error(y_holdout, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_holdout, y_pred))
                }
                self.logger.info(f"[{pipeline_name}] Holdout (Unadapted) Scores: {scores}")
                holdout_results.append({
                    "pipeline_name": pipeline_name,
                    "holdout_scores": scores
                })
                
                # For pipelines that use embeddings or person-static intercepts,
                # perform an adaptation evaluation.
                if any(x in pipeline_name for x in ["Embedding", "Intercept"]):
                    self.logger.info(f"Performing adaptation evaluation for pipeline: {pipeline_name}")
                    adapted = self.evaluate_holdout_embedding_adaptation(model, X_holdout, y_holdout)
                    adaptation_results[pipeline_name] = adapted
                    
            except Exception as e:
                self.logger.error(f"[{pipeline_name}] Error evaluating holdout set: {e}")
                continue
    
        return holdout_results, adaptation_results


    ##########################################################################
    # Main run: cross-validation + final refit
    ##########################################################################

    def run(self, pipeline_grid_dict, task_type="regression", scoring=custom_scorers, refit="mae", do_final_refit=True):
        self.logger.info("[run] Starting the ML pipeline run method.")
    
        if scoring is None:
            # Provide a dictionary recognized by scikit-learn
            scoring = {
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2"}


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
            y_train.index = train_df[self.cfg.USER_COL].values  # Set user_column as index

            groups = train_df[self.cfg.USER_COL]
        
            preprocessor = self.build_preprocessor(feature_cols, pipeline_name=pipeline_name)
            combined_pipeline = Pipeline([("preprocessor", preprocessor)] + list(raw_pipeline.steps))
           
            # ✅ Grid Search CV with correct label scaling
            gs = GridSearchCV(
                estimator=combined_pipeline,
                param_grid=param_grid,
                scoring=scoring,
                refit=refit,
                cv=cv,
                n_jobs=self.cfg.N_JOBS,
                error_score='raise'
            )
        
            try:
                self.logger.info(f"[{pipeline_name}] Fitting GridSearchCV.")
                gs.fit(X_train, y_train, groups=groups)
                self.logger.info(f"[{pipeline_name}] GridSearchCV completed.")
                self.logger.info(f"[{pipeline_name}] Best Parameters: {gs.best_params_}")
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
        
            # ✅ Prepare test data
            X_inner_test = self.df_inner_test[feature_cols].copy()
            y_inner_test = self.df_inner_test[self.cfg.LABEL_COL]
            groups_test = self.df_inner_test[self.cfg.USER_COL]
        
            # ✅ Prediction Handling with inverse transformation for GTTR models
            try:
                best_estimator = gs.best_estimator_

                y_pred = best_estimator.predict(X_inner_test)
        
            except Exception as e:
                self.logger.error(f"[{pipeline_name}] Error during prediction:")
                self.logger.error(e)
                continue
        
            # ✅ Calculate test scores
            try:
                test_scores = {
                    "r2": r2_score(y_inner_test, y_pred),
                    "mae": mean_absolute_error(y_inner_test, y_pred),
                    "rmse": np.sqrt(mean_squared_error(y_inner_test, y_pred)),
                }
                self.logger.info(f"[{pipeline_name}] Inner Test Scores: {test_scores}")
            
            except Exception as e:
                self.logger.error(f"[{pipeline_name}] Error during evaluation of inner test set:")
                self.logger.error(e)
                continue
        
            # Store results
            results_timebased.append({
                "pipeline_name": pipeline_name,
                "best_cv_score": gs.best_score_,
                "inner_test_scores": test_scores,
                "best_estimator": gs.best_estimator_,
            })
        
        self.logger.info("[run] ML pipeline run method completed.")

        # ✅ Evaluate models on holdout users
        holdout_results = self.evaluate_holdout_all(results_timebased)
    
        return results_timebased, holdout_results  # ✅ Now returns holdout evaluation too
