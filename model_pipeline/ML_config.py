#################################
# ML_Config.py
#################################

from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from merf.merf import MERF
from custom_models import MERFWrapperEmbed, GroupwiseStandardizingRegressor  # Import the wrapper

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#################################
# Define Custom Transformers
#################################



Regression_model_settings = {
    "LR_without_PS": (
        Pipeline([
            ("model_LR", TransformedTargetRegressor(
                regressor=LinearRegression(),
                transformer=StandardScaler()  # Standardize y
            ))
        ]),
        {
            "model_LR__regressor__fit_intercept": [True, False]  # ✅ Use correct step name
        }
    ),
    "LR_with_PS": (
        Pipeline([
            ("model_LRPS", TransformedTargetRegressor(
                regressor=LinearRegression(),
                transformer=StandardScaler()
            ))
        ]),
        {"model_LRPS__regressor__fit_intercept": [True, False]}
    ),

    # Random Forest WITHOUT PS (Population-based) - Currently commented out
#    "RF_without_PS": (
#        Pipeline([
#            ("varth", VarianceThreshold()),
#            ("scale_features", StandardScaler()),
#            ("model_TTR", TransformedTargetRegressor(
#                regressor=RandomForestRegressor(random_state=42),
#                transformer=StandardScaler()
#            ))
#        ]),
#        {
#            "model_TTR__regressor__n_estimators": [100, 200, 300],
#            "model_TTR__regressor__max_depth": [10, 20, None],
#            "model_TTR__regressor__min_samples_split": [2, 5, 10],
#            "model_TTR__regressor__max_features": ['sqrt', 'log2'],
#        }
#    ),

    # Updated hyperparameters for RF_with_PS - Currently commented out
#    "RF_with_PS": (
#        Pipeline([
#            ("varth", VarianceThreshold()),
#            ("scale_features", StandardScaler()),
#            ("model_TTR", TransformedTargetRegressor(
#                regressor=RandomForestRegressor(random_state=42),
#                transformer=StandardScaler()
#            ))
#        ]),
#        {
#            "model_TTR__regressor__n_estimators": [100, 200, 300],
#            "model_TTR__regressor__max_depth": [10, 20, None],
#            "model_TTR__regressor__min_samples_split": [2, 5, 10],
#            "model_TTR__regressor__max_features": ['sqrt', 'log2'],
#        }
#    ),

    "MERF_without_PS": (
        Pipeline([
            ("model_MERF", MERFWrapperEmbed(
                    gll_early_stop_threshold=0.01,
                    max_iterations=10,
                    rf__n_estimators=100,
                ))
        ]),
        {
            "model_MERF__max_iterations": [10, 15], 
            "model_MERF__rf__n_estimators": [50, 100]
        }
    ),
    
    "MERF_with_PS": (
        Pipeline([
            ("model_MERF", MERFWrapperEmbed(
                    gll_early_stop_threshold=0.01,
                    max_iterations=10,
                    rf__n_estimators=100,
                )
            )
        ]),
        {
            "model_MERF__max_iterations": [10, 15], 
            "model_MERF__rf__n_estimators": [50, 100]
        }

    ),
    "FFNN_without_PS": (
        Pipeline([
            # Wrap MLPRegressor in TransformedTargetRegressor so y is standardized
            ("model_TTR", TransformedTargetRegressor(
                regressor=MLPRegressor(random_state=42, max_iter=500),
                transformer=StandardScaler()  # standardize outcome
            ))
        ]),
        {
            # Note the param references "model_TTR__regressor__<param>"
            "model_TTR__regressor__hidden_layer_sizes": [(50,), (100,)],
            "model_TTR__regressor__alpha": [0.0001, 0.001]
        }
    ),

    "FFNN_with_PS": (
        Pipeline([
            ("model_TTR", TransformedTargetRegressor(
                regressor=MLPRegressor(random_state=42, max_iter=500),
                transformer=StandardScaler()
            ))
        ]),
        {
            "model_TTR__regressor__hidden_layer_sizes": [(50,), (100,)],
            "model_TTR__regressor__alpha": [0.0001, 0.001]
        }
    )
}

#################################
# Define Config Class
#################################

class Config:
    """
    Configuration Settings for a regression-based pipeline.

    This config defines:
      1) Data files (.pkl) to use.
      2) Column definitions for user ID, timestamp, sensor features, label, and person‑static features.
      3) Feature groupings and data types.
      4) Imputation and scaling choices.
      5) Regression model settings: a dictionary of model pipelines with descriptive names.
      6) Splitting parameters for outer (user-based) and inner (time-based) splits.
      7) Holdout evaluation parameters for adaptation scenarios.
      8) Outcome standardization: whether to z‑standardize the outcome per individual.
      9) Execution settings.
    """

    # 1) Data Loading
    USE_PKL_FILES = False  # Set to False if loading DataFrame directly
    PKL_FILES = [
        "data/df_ema_passive.pkl"
    ]

    # 2) Column Names
    USER_COL = "customer"
    TIME_COL = "sensor_block_end"
    LABEL_COL = "mean_na"
    STRATIFY_COL = "n_quest"

    # 3) Skewed Features
    SKEWED_FEATURES = [
        'age', 'hr_mean', 'hr_min', 'hr_max', 'hr_std', 'hr_zone_resting',
        'hr_zone_moderate', 'hr_zone_vigorous', 'n_steps', 'n_GPS',
        'total_distance_km', 'at_home_minute', 'time_in_transition_minutes',
        'time_stationary_minutes', 'activity_102_minutes', 'activity_103_minutes',
        'activity_104_minutes', 'activity_105_minutes', 'activity_106_minutes',
        'activity_107_minutes', 'sunshine_duration', 'precipitation_hours'
    ]

    # 4) Feature Groups
    numeric_features = SKEWED_FEATURES + ['apparent_temperature_mean']
    binary_features = ['somatic_problems', 'psychotropic', 'ema_smartphone', 'weekend']
    categorical_features = [
        'weekday', 'prior_treatment_description_simple', 'quest_create_hour', 'season', 'time_of_day', 'employability_description_simple'
    ]
    person_static_features = [
        'age', 'somatic_problems', 'psychotropic','employability_description_simple',
        'prior_treatment_description_simple', 'ema_smartphone'
    ]
    merf_cols = ["customer","intercept"]

    # 5) Feature Types (used during preprocessing)
    FEATURE_TYPES = {
        **{col: "continuous" for col in numeric_features},
        **{col: "categorical" for col in categorical_features + binary_features}
    }

    # 6) Preprocessing Settings
    IMPUTE_STRATEGY = "knn"
    SCALER_STRATEGY = "minmax"

    # 7) Regression Model Settings
    ANALYSIS = {
        "neg_affect_regression": {
            "TASK_TYPE": "regression",
            "LABEL": LABEL_COL,
            "MODEL_PIPEGRIDS": {},  # To be defined below
            "METRICS": {
                "r2": "r2",
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error"
            },
            "REFIT": "mae"  # Use the key 'mae' for refitting
        }
    }

    # Assign Regression_model_settings to MODEL_PIPEGRIDS in ANALYSIS
    ANALYSIS["neg_affect_regression"]["MODEL_PIPEGRIDS"] = Regression_model_settings

    # 8) Splitting Parameters
    HOLDOUT_RATIO = 0.1
    TIME_RATIO = 0.8
    N_INNER_CV = 5
    CV_METHOD = "forwardchaining"

    # 9) Holdout Evaluation
    HOLDOUT_EVAL_RATIO = 0.2
    HOLDOUT_ADAPT_RATIO = 0.8
    HOLDOUT_ADAPT_STRATEGIES = ["MERF", "NN_embeddings"]

    # 10) Execution
    N_JOBS = 1
    PARALLELIZE = True
    SAVE_MODELS = True
    DEBUG = False
