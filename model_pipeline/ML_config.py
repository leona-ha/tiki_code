#################################
# ML_Config.py
#################################

from imblearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from merf.merf import MERF
from model_pipeline.custom_models import MERFWrapper  # Import the wrapper


# Define regression model settings in a dictionary.
# Each key is a descriptive pipeline name.
# Define regression model settings in a dictionary.
# Each key is a descriptive pipeline name.
Regression_model_settings = {
    "LR_without_PS": (
        Pipeline([
            ("varth", VarianceThreshold()),
            ("scale_features", StandardScaler()),
            ("model_TTR", TransformedTargetRegressor(
                regressor=LinearRegression(),
                transformer=StandardScaler()  # standardize y
            ))
        ]),
        {"model_TTR__regressor__fit_intercept": [True]}
    ),
    "LR_with_PS": (
        Pipeline([
            ("varth", VarianceThreshold()),
            ("scale_features", StandardScaler()),
            ("model_TTR", TransformedTargetRegressor(
                regressor=LinearRegression(),
                transformer=StandardScaler()
            ))
        ]),
        {"model_TTR__regressor__fit_intercept": [True]}
    ),


    # Random Forest WITHOUT PS (Population-based)
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
#        "model_TTR__regressor__n_estimators": [100, 200, 300],
#        "model_TTR__regressor__max_depth": [10, 20, None],
#        "model_TTR__regressor__min_samples_split": [2, 5, 10],
#        "model_TTR__regressor__max_features": ['sqrt', 'log2'],
#        }
#    ),

    # Updated hyperparameters for RF_with_PS
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


    # MERF WITHOUT PS (Personalized) => We do NOT wrap with TTR
    # => outcome is NOT standardized
    # MERF WITHOUT PS (Personalized)
    "MERF_without_PS": (
        Pipeline([
            ("varth", VarianceThreshold()),
            ("scale_features", StandardScaler()),
            ("model_MERF", MERFWrapper(gll_early_stop_threshold=0.01, max_iterations=20))
        ]),
        {
            "model_MERF__max_iterations": [10, 20],
            "model_MERF__rf__n_estimators": [10, 50,100],          # Random Forest parameters
        }
    ),

    # MERF WITH PS (Personalized)
    "MERF_with_PS": (
        Pipeline([
            ("varth", VarianceThreshold()),
            ("scale_features", StandardScaler()),
            ("model_MERF", MERFWrapper(gll_early_stop_threshold=0.01, max_iterations=20))
        ]),
        {
            "model_MERF__max_iterations": [10, 20],
            "model_MERF__rf__n_estimators": [10, 50,100],  
        }
    ),

    "FFNN_without_PS": (
        Pipeline([
            ("varth", VarianceThreshold()),
            ("scale", StandardScaler()),
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
            ("varth", VarianceThreshold()),
            ("scale", StandardScaler()),
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
        'weekday', 'prior_treatment_description_simple', 'quest_create_hour', 'season', 'time_of_day'
    ]
    person_static_features = [
        'age', 'somatic_problems', 'psychotropic',
        'prior_treatment_description_simple', 'ema_smartphone'
    ]

    # 5) Feature Types (used during preprocessing)
    FEATURE_TYPES = {
        **{col: "continuous" for col in numeric_features},
        **{col: "categorical" for col in categorical_features + binary_features}
    }

    # 6) Preprocessing Settings
    IMPUTE_STRATEGY = "knn"
    SCALER_STRATEGY = "minmax"

    ANALYSIS = {
        "neg_affect_regression": {
            "TASK_TYPE": "regression",
            "LABEL": LABEL_COL,
            "MODEL_PIPEGRIDS": Regression_model_settings,
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

    # 9) Holdout Evaluation
    HOLDOUT_EVAL_RATIO = 0.2
    HOLDOUT_ADAPT_RATIO = 0.8
    HOLDOUT_ADAPT_STRATEGIES = ["MERF", "NN_embeddings"]

    # 10) Execution
    N_JOBS = 2
    PARALLELIZE = True
    SAVE_MODELS = True
    DEBUG = False
