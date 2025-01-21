#################################
# ML_Pipeline.py
#################################

import numpy as np
import pandas as pd
from datetime import datetime
from joblib import dump
import copy

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer


class MLpipeline:
    """
    Pipeline for regression analysis with time-based and user-based holdout evaluation.

    Steps:
      1. set_data(df): Optionally set an in-memory DataFrame.
      2. outer_user_split(): Create a user-based holdout set.
      3. inner_time_split(): Perform a time-based split for known users.
      4. run(): Train models (via GridSearchCV) on the inner splits, evaluate on the inner test set.
      5. evaluate_holdout_all(): Evaluate holdout scenario (a) and (b) for each pipeline's best estimator.
    """

    def __init__(self, config, random_state=42):
        self.cfg = config
        self.random_state = random_state

        # Data containers
        self.df_all = None
        self.df_holdout = None
        self.df_known = None
        self.df_inner_train = None
        self.df_inner_test = None

        # Tracking best model across pipelines
        self.best_estimator_ = None
        self.best_score_ = -np.inf

    # -------------------------------
    # DATA HANDLING
    # -------------------------------
    def set_data(self, df):
        """
        Assigns the in-memory DataFrame directly to pipeline's df_all,
        bypassing file-based loading. Useful in a Jupyter notebook setting.
        """
        self.df_all = df.copy().reset_index(drop=True)
        print(f"[set_data] DataFrame with {len(self.df_all)} rows loaded in pipeline.")

    def outer_user_split(self):
        """
        Splits users into a holdout set according to HOLDOUT_RATIO.
        The remaining users become the 'known' set.
        """
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
        """
        Performs time-based splits (training = first TIME_RATIO, test = last TIME_RATIO)
        for each user in df_known.
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
        print(f"[inner_time_split] Inner train size: {len(self.df_inner_train)}, "
              f"test size: {len(self.df_inner_test)}.")

    # -------------------------------
    # FEATURE SELECTION
    # -------------------------------
    def _get_feature_cols(self, pipeline_name):
        """
        If pipeline_name includes 'with_PS', add person_static_features to numeric/binary/cat features.
        Otherwise, use the base sets.
        """
        base_features = (self.cfg.numeric_features
                         + self.cfg.binary_features
                         + self.cfg.categorical_features)
        if "with_PS" in pipeline_name:
            return list(set(base_features + self.cfg.person_static_features))
        else:
            return base_features

    def get_inner_data(self, pipeline_name):
        """
        Returns X_train, y_train, X_test, y_test for the time-based splits.
        """
        features = self._get_feature_cols(pipeline_name)
        X_train = self.df_inner_train[features].values
        y_train = self.df_inner_train[self.cfg.LABEL_COL].values

        X_test = self.df_inner_test[features].values
        y_test = self.df_inner_test[self.cfg.LABEL_COL].values

        return X_train, y_train, X_test, y_test

    # -------------------------------
    # PREPROCESSOR
    # -------------------------------


    def build_preprocessor(self, feature_cols):
        """
        A ColumnTransformer that:
          - log-transforms skewed numeric columns, then scales them
          - scales non-skewed numeric columns
          - one-hot encodes categorical columns
        """
        # Distinguish numeric vs. categorical
        numeric_cols, cat_cols = [], []
        for col in feature_cols:
            if self.cfg.FEATURE_TYPES.get(col, "continuous") == "categorical":
                cat_cols.append(col)
            else:
                numeric_cols.append(col)
    
        # We'll further separate numeric_cols into skewed vs. non-skewed
        skewed_cols = [c for c in numeric_cols if c in self.cfg.SKEWED_FEATURES]
        non_skewed_cols = [c for c in numeric_cols if c not in self.cfg.SKEWED_FEATURES]
    
        # Decide which imputer
        if self.cfg.IMPUTE_STRATEGY == "knn":
            imputer_numeric = KNNImputer(n_neighbors=5)
        elif self.cfg.IMPUTE_STRATEGY == "iterative":
            imputer_numeric = IterativeImputer(random_state=self.random_state)
        elif self.cfg.IMPUTE_STRATEGY in ("mean", "median", "most_frequent"):
            imputer_numeric = SimpleImputer(strategy=self.cfg.IMPUTE_STRATEGY)
        else:
            imputer_numeric = "passthrough"
    
        # Decide which scaler
        if self.cfg.SCALER_STRATEGY == "standard":
            scaler = StandardScaler()
        elif self.cfg.SCALER_STRATEGY == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = "passthrough"
    
        # Function to apply a log transform (log1p => log(1 + x))
        def log_transform(X):
            # If X can have zero or negative, handle carefully. 
            # Typically, data must be strictly positive if using log1p.
            return np.log1p(X)
    
        # Build separate pipelines for skewed vs. non-skewed numeric columns
        skewed_numeric_pipeline = Pipeline([
            ("impute_num", imputer_numeric),
            ("log", FunctionTransformer(log_transform, validate=False)),
            ("scale", scaler)
        ])
    
        non_skewed_numeric_pipeline = Pipeline([
            ("impute_num", imputer_numeric),
            ("scale", scaler)
        ])
    
        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ("impute_cat", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])
    
        # Build the final ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("skewed_num", skewed_numeric_pipeline, skewed_cols),
                ("non_skewed_num", non_skewed_numeric_pipeline, non_skewed_cols),
                ("cat", categorical_pipeline, cat_cols)
            ],
            remainder="drop"
        )
    
        return preprocessor

    # -------------------------------
    # TRAINING AND TIME-BASED EVALUATION
    # -------------------------------
    def run(self, pipeline_grid_dict, task_type="regression", scoring="r2"):
        """
        1) Build pipeline (preprocessor + user pipeline)
        2) Run GridSearchCV on the time-based splits
        3) Evaluate on the time-based test
        4) Return results containing best_cv_score, inner_test_score, plus the best_estimator
        """
        results_timebased = []

        for pipeline_name, (raw_pipeline, param_grid) in pipeline_grid_dict.items():
            print(f"\n[run] Starting pipeline: {pipeline_name}")
            feature_cols = self._get_feature_cols(pipeline_name)
            X_train, y_train, X_test, y_test = self.get_inner_data(pipeline_name)

            # Build preprocessor
            preprocessor = self.build_preprocessor(feature_cols)

            # Combine user pipeline steps with our preprocessor
            steps = [("preprocessor", preprocessor)] + list(raw_pipeline.steps)
            combined_pipeline = Pipeline(steps=steps)

            cv_obj = KFold(n_splits=self.cfg.N_INNER_CV,
                           shuffle=True,
                           random_state=self.random_state)

            gs = GridSearchCV(
                estimator=combined_pipeline,
                param_grid=param_grid,
                scoring=scoring,
                cv=cv_obj,
                refit=True,
                n_jobs=self.cfg.N_JOBS
            )

            # Fit on the time-based training data
            gs.fit(X_train, y_train)

            print(f"{pipeline_name} Best CV Score: {gs.best_score_:.3f}")
            print(f"{pipeline_name} Best Params: {gs.best_params_}")

            # Track overall best
            if gs.best_score_ > self.best_score_:
                self.best_score_ = gs.best_score_
                self.best_estimator_ = gs.best_estimator_

            # Evaluate on the time-based test data
            test_score = gs.best_estimator_.score(X_test, y_test)
            print(f"{pipeline_name} Inner Test Score: {test_score:.3f}")

            results_timebased.append({
                "pipeline_name": pipeline_name,
                "model_type": pipeline_name.split("_")[0],
                "with_person_static": ("with_PS" in pipeline_name),
                "best_cv_score": gs.best_score_,
                "inner_test_score": test_score,
                "best_estimator": gs.best_estimator_
            })

        if self.cfg.SAVE_MODELS and self.best_estimator_ is not None:
            dump(self.best_estimator_, "best_model.joblib")

        return results_timebased

    # -------------------------------
    # HOLDOUT (USER-BASED) EVALUATION
    # -------------------------------
    def evaluate_holdout_all(self, timebased_results):
        """
        Evaluate user-based holdout (scenarios a + b) for each pipeline's best estimator.
        Returns a separate list of holdout results.
        """
        results_holdout = []

        for res in timebased_results:
            pipeline_name = res["pipeline_name"]
            best_estimator = res["best_estimator"]

            print(f"\n[Holdout Eval] Pipeline: {pipeline_name}")
            holdout_score_a, holdout_score_b = self._evaluate_holdout_single(pipeline_name, best_estimator)

            results_holdout.append({
                "pipeline_name": pipeline_name,
                "holdout_score_a": holdout_score_a,
                "holdout_score_b": holdout_score_b
            })

        return results_holdout

    def _evaluate_holdout_single(self, pipeline_name, best_estimator):
        """
        (a) Direct evaluation for the last X% of each holdout user
        (b) Adaptation-based scenario
        """
        if self.df_holdout is None or len(self.df_holdout) == 0:
            print("No holdout data available.")
            return None, None

        holdout_score_a = None
        feature_cols = self._get_feature_cols(pipeline_name)

        # Scenario (a) direct evaluation
        holdout_eval_dfs = []
        for user in self.df_holdout[self.cfg.USER_COL].unique():
            user_data = self.df_holdout[self.df_holdout[self.cfg.USER_COL] == user].sort_values(by=self.cfg.TIME_COL)
            if len(user_data) < 1:
                continue
            eval_start = int(np.floor(len(user_data) * (1 - self.cfg.HOLDOUT_EVAL_RATIO)))
            evaluation_data = user_data.iloc[eval_start:]
            holdout_eval_dfs.append(evaluation_data)

        if holdout_eval_dfs:
            holdout_eval_df = pd.concat(holdout_eval_dfs, axis=0).reset_index(drop=True)
            X_holdout_eval = holdout_eval_df[feature_cols].values
            y_holdout_eval = holdout_eval_df[self.cfg.LABEL_COL].values

            holdout_score_a = best_estimator.score(X_holdout_eval, y_holdout_eval)
            print(f"{pipeline_name} Holdout Score Scenario (a): {holdout_score_a:.3f}")

        # Scenario (b) adaptation
        holdout_score_b = None
        holdout_adapt_split = self.build_holdout_adaptation_split()
        adapted_scores = []

        for user, (adapt_df, eval_df) in holdout_adapt_split.items():
            if len(adapt_df) < 1 or len(eval_df) < 1:
                continue
            X_adapt = adapt_df[feature_cols].values
            y_adapt = adapt_df[self.cfg.LABEL_COL].values
            X_eval = eval_df[feature_cols].values
            y_eval = eval_df[self.cfg.LABEL_COL].values

            for strategy in self.cfg.HOLDOUT_ADAPT_STRATEGIES:
                adapted_model = self.build_model(adaptation_strategy=strategy,
                                                 adaptation_data=(X_adapt, y_adapt))
                if adapted_model is not None:
                    score = adapted_model.score(X_eval, y_eval)
                    adapted_scores.append(score)
                    print(f"{pipeline_name} - User {user} - Strategy {strategy}: {score:.3f}")

        if adapted_scores:
            holdout_score_b = np.mean(adapted_scores)
            print(f"{pipeline_name} Average Adaptation Score Scenario (b): {holdout_score_b:.3f}")

        return holdout_score_a, holdout_score_b

    def build_holdout_adaptation_split(self):
        """
        For each holdout user, split their data into adaptation + evaluation sets
        according to HOLDOUT_ADAPT_RATIO.
        """
        adaptation_evaluation_split = {}
        if self.df_holdout is None or len(self.df_holdout) == 0:
            return adaptation_evaluation_split

        for user in self.df_holdout[self.cfg.USER_COL].unique():
            user_data = self.df_holdout[self.df_holdout[self.cfg.USER_COL] == user].sort_values(by=self.cfg.TIME_COL)
            n = len(user_data)
            if n < 2:
                continue
            adaptation_end = int(np.floor(n * self.cfg.HOLDOUT_ADAPT_RATIO))
            adaptation_data = user_data.iloc[:adaptation_end]
            evaluation_data = user_data.iloc[adaptation_end:]
            adaptation_evaluation_split[user] = (adaptation_data, evaluation_data)

        return adaptation_evaluation_split

    def build_model(self, adaptation_strategy=None, adaptation_data=None):
        """
        Stub for personalized adaptation. If adaptation_strategy is set,
        we update the best_estimator or apply partial fit. 
        (Currently placeholders for MERF or NN embeddings).
        """
        if adaptation_strategy is None:
            return self.best_estimator_
        if adaptation_strategy == "MERF":
            print("MERF adaptation not implemented yet.")
            return None
        elif adaptation_strategy == "NN_embeddings":
            print("NN embeddings adaptation not implemented yet.")
            return None
        else:
            raise ValueError(f"Unknown adaptation strategy: {adaptation_strategy}")

