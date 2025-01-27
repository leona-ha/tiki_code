import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, mean_squared_error

def mae_scorer(estimator, X, y_true):
    """
    Custom MAE scorer that supports additional parameters for MERF.
    """
    Z = getattr(estimator, "Z_", None)
    clusters = getattr(estimator, "clusters_", None)
    if Z is None or clusters is None:
        raise ValueError("Custom scorer: 'Z_' or 'clusters_' not set in the estimator.")
    y_pred = estimator.predict(X, Z=Z, clusters=clusters)
    return -mean_absolute_error(y_true, y_pred)


def rmse_scorer(estimator, X, y_true):
    """
    Custom RMSE scorer that supports additional parameters for MERF.
    """
    Z = getattr(estimator, "Z_", None)
    clusters = getattr(estimator, "clusters_", None)
    if Z is None or clusters is None:
        raise ValueError("Custom scorer: 'Z_' or 'clusters_' not set in the estimator.")
    y_pred = estimator.predict(X, Z=Z, clusters=clusters)
    return -np.sqrt(mean_squared_error(y_true, y_pred))


def r2_scorer(estimator, X, y_true):
    """
    Custom R2 scorer that supports additional parameters for MERF.
    """
    Z = getattr(estimator, "Z_", None)
    clusters = getattr(estimator, "clusters_", None)
    if Z is None or clusters is None:
        raise ValueError("Custom scorer: 'Z_' or 'clusters_' not set in the estimator.")
    y_pred = estimator.predict(X, Z=Z, clusters=clusters)
    return r2_score(y_true, y_pred)


# Define custom scorers for GridSearchCV
custom_scorers = {
    "mae": make_scorer(mae_scorer, greater_is_better=False),
    "rmse": make_scorer(rmse_scorer, greater_is_better=False),
    "r2": make_scorer(r2_scorer, greater_is_better=True),
}


# ML Pipeline
class MLpipeline:
    def __init__(self, config, random_state=42):
        self.cfg = config
        self.random_state = random_state
        self.df_all = None
        self.df_holdout = None
        self.df_known = None
        self.df_inner_train = None
        self.df_inner_test = None

        print("[ML_Pipeline] Configuration Loaded:")
        print(f"  Impute Strategy: {self.cfg.IMPUTE_STRATEGY}")
        print(f"  Scaler Strategy: {self.cfg.SCALER_STRATEGY}")
        print(f"  Holdout Ratio: {self.cfg.HOLDOUT_RATIO}")
        print(f"  Time Ratio: {self.cfg.TIME_RATIO}")
        print(f"  Number of Jobs: {self.cfg.N_JOBS}")

    def set_data(self, df):
        self.df_all = df.copy().reset_index(drop=True)
        print(f"[set_data] DataFrame with {len(self.df_all)} rows loaded in pipeline.")

    def outer_user_split(self):
        unique_users = self.df_all[self.cfg.USER_COL].unique()
        n_users = len(unique_users)
        n_holdout = int(np.floor(n_users * self.cfg.HOLDOUT_RATIO))
        rng = np.random.default_rng(self.random_state)
        shuffled = rng.permutation(unique_users)
        holdout_users = set(shuffled[:n_holdout])

        mask_holdout = self.df_all[self.cfg.USER_COL].isin(holdout_users)
        self.df_holdout = self.df_all[mask_holdout].copy().reset_index(drop=True)
        self.df_known = self.df_all[~mask_holdout].copy().reset_index(drop=True)

        print(f"[outer_user_split] Held out {n_holdout}/{n_users} users; "
              f"holdout size: {len(self.df_holdout)} rows.")

    def inner_time_split(self):
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

        print(f"[inner_time_split] Inner train size: {len(self.df_inner_train)}, "
              f"test size: {len(self.df_inner_test)}.")

    def assign_feature_columns(self, feature_cols):
        skewed_cols = [col for col in feature_cols if col in self.cfg.SKEWED_FEATURES]
        non_skewed_cols = [col for col in feature_cols if col in self.cfg.numeric_features and col not in skewed_cols]
        cat_cols = [col for col in feature_cols if col in self.cfg.categorical_features]
        return skewed_cols, non_skewed_cols, cat_cols

    def _get_feature_cols(self, pipeline_name):
        base_features = (
            self.cfg.numeric_features +
            self.cfg.binary_features +
            self.cfg.categorical_features
        )
        if "with_PS" in pipeline_name:
            return list(set(base_features + self.cfg.person_static_features))
        return list(set(base_features) - set(self.cfg.person_static_features))

    def create_predefined_splits(self):
        """
        Create a single PredefinedSplit object for cross-validation with time-based folds per user.
        Returns:
            PredefinedSplit: The PredefinedSplit object for cross-validation.
            pd.DataFrame: The training DataFrame.
        """
        test_fold = np.full(len(self.df_inner_train), -1)  # Default all to training (-1)
        user_groups = self.df_inner_train.groupby(self.cfg.USER_COL)
    
        for user, group in user_groups:
            # Sort by time for each user
            group = group.sort_values(by=self.cfg.TIME_COL)
            indices = group.index.to_list()
    
            # Split user's data into 5 folds
            n_samples = len(indices)
            fold_size = n_samples // self.cfg.N_INNER_CV  # Divide user's data into folds
    
            for fold in range(self.cfg.N_INNER_CV):
                start = fold * fold_size
                end = start + fold_size if fold < self.cfg.N_INNER_CV - 1 else n_samples
                test_indices = indices[start:end]
                for idx in test_indices:
                    test_fold[idx] = fold  # Assign fold index to the `test_fold`
    
        predefined_split = PredefinedSplit(test_fold=test_fold)
    
        print(f"[DEBUG] PredefinedSplit test fold counts: {np.bincount(test_fold[test_fold >= 0])}")
        return predefined_split, self.df_inner_train


    def build_preprocessor(self, feature_cols):
        imputer_numeric = KNNImputer(n_neighbors=5) if self.cfg.IMPUTE_STRATEGY == "knn" else SimpleImputer(strategy=self.cfg.IMPUTE_STRATEGY)
        scaler = MinMaxScaler() if self.cfg.SCALER_STRATEGY == "minmax" else StandardScaler()
        skewed_cols, non_skewed_cols, cat_cols = self.assign_feature_columns(feature_cols)
        skewed_numeric_pipeline = Pipeline([("impute", imputer_numeric), ("log", FunctionTransformer(np.log1p, validate=False)), ("scale", scaler)])
        non_skewed_numeric_pipeline = Pipeline([("impute", imputer_numeric), ("scale", scaler)])
        categorical_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])

        preprocessor = ColumnTransformer([
            ("skewed_num", skewed_numeric_pipeline, skewed_cols),
            ("non_skewed_num", non_skewed_numeric_pipeline, non_skewed_cols),
            ("cat", categorical_pipeline, cat_cols),
        ], remainder="drop")

        return preprocessor

    def run(self, pipeline_grid_dict, task_type="regression", scoring=None, refit="mae"):
        if scoring is None:
            scoring = {
                "r2": "r2",
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
            }
    
        results_timebased = []
    
        # Create predefined splits and retrieve training data
        predefined_split, train_df = self.create_predefined_splits()
    
        for pipeline_name, (raw_pipeline, param_grid) in pipeline_grid_dict.items():
            print(f"\n[run] Starting pipeline: {pipeline_name}")
            feature_cols = self._get_feature_cols(pipeline_name)
    
            # Extract train data
            X_train = train_df[feature_cols]
            y_train = train_df[self.cfg.LABEL_COL]
            clusters_train = train_df[self.cfg.USER_COL]
    
            # Define Z for MERF pipelines
            if hasattr(self.cfg, "RANDOM_EFFECT_FEATURES"):
                Z_train = pd.DataFrame(train_df[self.cfg.RANDOM_EFFECT_FEATURES].values, index=X_train.index)
            else:
                Z_train = pd.DataFrame(np.ones((X_train.shape[0], 1)), index=X_train.index)
    
            # Define inner test set
            X_inner_test = self.df_inner_test[feature_cols]
            y_inner_test = self.df_inner_test[self.cfg.LABEL_COL]
            clusters_inner_test = self.df_inner_test[self.cfg.USER_COL]
    
            if hasattr(self.cfg, "RANDOM_EFFECT_FEATURES"):
                Z_inner_test = pd.DataFrame(self.df_inner_test[self.cfg.RANDOM_EFFECT_FEATURES].values, index=X_inner_test.index)
            else:
                Z_inner_test = pd.DataFrame(np.ones((X_inner_test.shape[0], 1)), index=X_inner_test.index)
    
            # Preprocessor and pipeline setup
            preprocessor = self.build_preprocessor(feature_cols)
            steps = [("preprocessor", preprocessor)] + list(raw_pipeline.steps)
            combined_pipeline = Pipeline(steps=steps)
    
            # Perform GridSearchCV
            print(f"[DEBUG] Starting GridSearchCV for {pipeline_name}...")
            gs = GridSearchCV(
                estimator=combined_pipeline,
                param_grid=param_grid,
                scoring=custom_scorers if "MERF" in pipeline_name else scoring,
                refit=refit,
                return_train_score=True,
                cv=predefined_split,
                n_jobs=self.cfg.N_JOBS,
            )
    
            # Fit parameters for MERF
            fit_params = {}
            if "MERF" in pipeline_name:
                fit_params = {
                    "model_MERF__Z": Z_train,
                    "model_MERF__clusters": clusters_train,
                }
    
            # Fit the model
            gs.fit(X_train, y_train, **fit_params)
            best_model = gs.best_estimator_
            best_score = gs.best_score_
    
            print(f"[{pipeline_name}] Best CV Score ({refit}): {best_score:.3f}")
            print(f"[{pipeline_name}] Best Params: {gs.best_params_}")
    
            # Test set evaluation
            if "MERF" in pipeline_name:
                test_predictions = best_model.predict(X_inner_test, Z=Z_inner_test, clusters=clusters_inner_test)
            else:
                test_predictions = best_model.predict(X_inner_test)
    
            # Calculate test scores
            test_scores = {
                "r2": r2_score(y_inner_test, test_predictions),
                "mae": mean_absolute_error(y_inner_test, test_predictions),
                "rmse": np.sqrt(mean_squared_error(y_inner_test, test_predictions)),
            }
    
            print(f"[{pipeline_name}] Inner Test Scores: {test_scores}")
    
            # Append results
            results_timebased.append({
                "pipeline_name": pipeline_name,
                "best_cv_score": best_score,
                "inner_test_scores": test_scores,
                "best_estimator": best_model,
            })
    
        return results_timebased
    
    
    
