#################################
# ML_Config.py
#################################

from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

import numpy as np
import pandas as pd
import random

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor, ColumnTransformer
from merf.merf import MERF
from custom_models import MERFWrapperEmbed, GlobalInterceptModel, PerUserInterceptModel, PerUserLabelScaler
from custom_models import PerUserTransformedTargetRegressor, SplitFeaturesTransformer, PerUserFeatureScaler,LMERWrapper
# Enable experimental features in scikit-learn
from sklearn.experimental import enable_iterative_imputer  # ✅ Must be imported first
from sklearn.impute import IterativeImputer  # ✅ Now you can import it

from custom_models import KerasFFNNRegressor  # your new custom class
from tensorflow.keras.callbacks import EarlyStopping

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    STRATIFY_COL = "n_quest_stratify"

    
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
    categorical_features_categories = {
    'weekday': ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday', 'Wednesday'],
    'prior_treatment_description_simple': ['no prior treatment', 'prior inpatient', 'prior psychotherapy'],
    'quest_create_hour': [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
    'season': ['Fall', 'Spring', 'Summer', 'Winter'],
    'time_of_day': ['Afternoon', 'Early Morning', 'Evening', 'Morning', 'Night'],
    'employability_description_simple': ['no', 'yes']
}

    person_static_features = [
        'age', 'somatic_problems', 'psychotropic','employability_description_simple',
        'prior_treatment_description_simple', 'ema_smartphone'
    ]
    merf_cols = ["customer","intercept"]
    passive_cols = ['hr_mean', 'hr_min', 'hr_max', 'hr_std', 'hr_zone_resting',
        'hr_zone_moderate', 'hr_zone_vigorous', 'n_steps', 'n_GPS',
        'total_distance_km', 'at_home_minute', 'time_in_transition_minutes',
        'time_stationary_minutes', 'activity_102_minutes', 'activity_103_minutes',
        'activity_104_minutes', 'activity_105_minutes', 'activity_106_minutes',
        'activity_107_minutes', 'sunshine_duration', 'precipitation_hours', 'apparent_temperature_mean', 'weekend', 'weekday','quest_create_hour', 'season', 'time_of_day']

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


    # 8) Splitting Parameters
    HOLDOUT_RATIO = 0.1
    TIME_RATIO = 0.8
    N_INNER_CV = 5
    CV_METHOD = "forwardchaining"

    # 9) Holdout Evaluation
    HOLDOUT_EVAL_RATIO = 0.2
    HOLDOUT_ADAPT_RATIO = 0.8
    HOLDOUT_DATA_USAGE = "last_20"


    # 10) Execution
    N_JOBS = 1
    PARALLELIZE = True
    SAVE_MODELS = True
    DEBUG = False


embedding_model = KerasFFNNRegressor(
            sensor_feature_dim=None,  # Let the model infer: input will be (n_features, ) + user col.
            num_users=158,
            embedding_dim=32,
            hidden_units=(64, 32),
            epochs=20,
            batch_size=32,
            use_embedding=True,
            verbose=0
)
plain_ffnn_model = KerasFFNNRegressor(
            sensor_feature_dim=None,  # Will be inferred from X (all columns are sensor features)
            num_users=158,            # Irrelevant when use_embedding=False
            embedding_dim=32,         # Ignored when use_embedding is False
            hidden_units=(64, 32),
            epochs=15,
            batch_size=32,
            use_embedding=False,
            verbose=0
)


inner_ffnn_pipeline = Pipeline([                  
    ("split_features", SplitFeaturesTransformer(user_col=Config.USER_COL)), 
    ("keras_model", embedding_model)
])


#################################
# Define Custom Transformers
#################################

Regression_model_settings = {
    #################################
    # 1. BASELINE MODELS
    #################################
    "Global_Intercept": (
        Pipeline([
            ("basemodel_global", GlobalInterceptModel())
        ]),
        {}
    ),
    "PerUser_Intercept": (
        Pipeline([
            ("basemodel_PerUser", PerUserInterceptModel())
        ]),
        {}
    ),


    #################################
    # 2. NON-SCALED MODELS (NO FEATURE SCALING) – Global Version
    #################################
    
    "FFNN_with_Embeddings": (
        Pipeline([
            ("split_features", SplitFeaturesTransformer(user_col=Config.USER_COL)),
            ("keras_model", embedding_model)
        ]),
        {
            "keras_model__embedding_dim": [8, 16, 32],
            "keras_model__hidden_units": [(64, 32),(128, 64), (128, 64, 32)],
            "keras_model__batch_size": [64],
        })
        }
    
"""    
    "LR": (
        Pipeline([
            ("model_LR", LinearRegression())
        ]),
        {"model_LR__fit_intercept": [True, False]}
    ),
    "LR_with_PS": (
        Pipeline([
            ("model_LRPS", LinearRegression())
        ]),
        {"model_LRPS__fit_intercept": [True, False]}
    ),

    "RF": (
        Pipeline([
            ("model_TTR", TransformedTargetRegressor(
                regressor=RandomForestRegressor(random_state=42)
            ))
        ]),
        {
            "model_TTR__regressor__n_estimators": [100, 200, 300],
            "model_TTR__regressor__max_depth": [5,10, None],
            "model_TTR__regressor__min_samples_split": [2, 5, 10],
            "model_TTR__regressor__max_features": ['sqrt', 'log2']
        }
    ),
    "RF_with_PS": (
        Pipeline([
            ("model_TTR", TransformedTargetRegressor(
                regressor=RandomForestRegressor(random_state=42)
            ))
        ]),
        {
            "model_TTR__regressor__n_estimators": [100, 200, 300],
            "model_TTR__regressor__max_depth": [5,10, None],
            "model_TTR__regressor__min_samples_split": [2, 5, 10],
            "model_TTR__regressor__max_features": ['sqrt', 'log2']
        }
    ),
    "FFNN": (
        Pipeline([
            ("keras_model", TransformedTargetRegressor(
                regressor=plain_ffnn_model
            ))
        ]),
        {
            "keras_model__regressor__hidden_units": [(64, 32), (128, 64), (128, 64, 32)],
            "keras_model__regressor__batch_size": [32, 64],
            "keras_model__regressor__learning_rate": [1e-3, 1e-4],
            "keras_model__regressor__dropout_rate": [0.25, 0.5],
        }
    ),
    "FFNN_with_PS": (
        Pipeline([
            ("keras_model", TransformedTargetRegressor(
                regressor=plain_ffnn_model
            ))
        ]),
        {
            "keras_model__regressor__hidden_units": [(64, 32), (128, 64), (128, 64, 32)],
            "keras_model__regressor__batch_size": [32, 64],
            "keras_model__regressor__learning_rate": [1e-3, 1e-4],
            "keras_model__regressor__dropout_rate": [0.25, 0.5],
        }
    ),

    "MERF": (
        Pipeline([
            ("model_MERF", TransformedTargetRegressor(
                regressor=MERFWrapperEmbed(
                    gll_early_stop_threshold=0.025,
                    max_iterations=10,
                    rf__n_estimators=100
                )
            ))
        ]),
        {
            "model_MERF__regressor__max_iterations": [10,15],
            "model_MERF__regressor__rf__n_estimators": [50, 100]
        }
    ),
    "MERF_with_PS": (
        Pipeline([
            ("model_MERF", TransformedTargetRegressor(
                regressor=MERFWrapperEmbed(
                    gll_early_stop_threshold=0.025,
                    max_iterations=10,
                    rf__n_estimators=100
                )
            ))
        ]),
        {
            "model_MERF__regressor__max_iterations": [10,15],
            "model_MERF__regressor__rf__n_estimators": [50, 100]
        }
    ),"""


# Assign model settings to config
Config.ANALYSIS["neg_affect_regression"]["MODEL_PIPEGRIDS"] = Regression_model_settings
