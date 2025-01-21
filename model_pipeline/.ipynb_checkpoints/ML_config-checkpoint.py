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

# Define regression model settings in a dictionary.
# Each key is a descriptive pipeline name.
Regression_model_settings = {
    # Linear Regression WITHOUT PS (Population-based)
    # => outcome is standardized
    "LR_without_PS": (
        Pipeline([
            ("varth", VarianceThreshold()),
            ("scale_features", StandardScaler()),
            # TTR for target standardization
            ("model_TTR", TransformedTargetRegressor(
                regressor=LinearRegression(),
                transformer=StandardScaler()  # standardize y
            ))
        ]),
        # Param grid must reference the "model_TTR__regressor__" prefix
        {"model_TTR__regressor__fit_intercept": [True]}
    ),

    # Linear Regression WITH PS (still population-based)
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
    "RF_without_PS": (
        Pipeline([
            ("varth", VarianceThreshold()),
            ("scale_features", StandardScaler()),
            ("model_TTR", TransformedTargetRegressor(
                regressor=RandomForestRegressor(random_state=42),
                transformer=StandardScaler()
            ))
        ]),
        {
         "model_TTR__regressor__n_estimators": [100, 200],
         "model_TTR__regressor__max_depth": [None, 10, 20]
        }
    ),

    # Random Forest WITH PS (Population-based)
    "RF_with_PS": (
        Pipeline([
            ("varth", VarianceThreshold()),
            ("scale_features", StandardScaler()),
            ("model_TTR", TransformedTargetRegressor(
                regressor=RandomForestRegressor(random_state=42),
                transformer=StandardScaler()
            ))
        ]),
        {
         "model_TTR__regressor__n_estimators": [100, 200],
         "model_TTR__regressor__max_depth": [None, 10, 20]
        }
    ),

    # MERF WITHOUT PS (Personalized) => We do NOT wrap with TTR
    # => outcome is NOT standardized
    "MERF_without_PS": (
        Pipeline([
            ("varth", VarianceThreshold()),
            ("scale_features", StandardScaler()),
            ("model_MERF", MERF())
        ]),
        {"model_MERF__n_estimators": [100],
         "model_MERF__max_depth": [None, 10]}
    ),

    # MERF WITH PS (Personalized) => also no TTR
    "MERF_with_PS": (
        Pipeline([
            ("varth", VarianceThreshold()),
            ("scale_features", StandardScaler()),
            ("model_MERF", MERF())
        ]),
        {"model_MERF__n_estimators": [100],
         "model_MERF__max_depth": [None, 10]}
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

    # 1) Data Files
    PKL_FILES = [
        "data/df_ema_passive.pkl"  # Replace with your actual file path(s)
    ]

    # 2) Column Names
    USER_COL = "customer"
    TIME_COL = "sensor_block_end"
    LABEL_COL = "mean_na"
    STRATIFY_COL = "n_quest"
    
    SKEWED_FEATURES =  ['age', 'hr_mean', 'hr_min', 'hr_max', 'hr_std', 'hr_zone_resting', 'hr_zone_moderate', 'hr_zone_vigorous', 'n_steps', 'n_GPS','total_distance_km', 'at_home_minute', 'time_in_transition_minutes', 'time_stationary_minutes', 'prop_time_moving', 'prop_time_stationary', 'activity_102_minutes', 'activity_103_minutes', 'activity_104_minutes', 'activity_105_minutes', 'activity_106_minutes', 'activity_107_minutes', 'sunshine_duration', 'precipitation_hours']

    # 3) Feature Groups
    numeric_features = [
        'hr_mean', 'hr_min', 'hr_max', 'hr_std', 'hr_zone_resting', 'hr_zone_moderate',
        'hr_zone_vigorous', 'n_steps',  'n_GPS',
        'total_distance_km', 'at_home_minute', 'time_in_transition_minutes',
        'time_stationary_minutes',
        'activity_102_minutes', 'activity_103_minutes', 'activity_104_minutes',
        'activity_105_minutes', 'activity_106_minutes', 'activity_107_minutes',
        'apparent_temperature_mean', 'sunshine_duration', 'precipitation_hours'
    ]
    binary_features = [
        'somatic_problems', 'psychotropic',
        'employability_description_simple', 'ema_smartphone', 'weekend'
    ]
    categorical_features = [
        'weekday', 'prior_treatment_description_simple', 'quest_create_hour', 'season', 'time_of_day'
    ]
    person_static_features = [
        'age', 'somatic_problems', 'psychotropic','prior_treatment_description_simple'
        'employability_description_simple', 'ema_smartphone',
    ]


    # 4) Feature Types (used during preprocessing)
    FEATURE_TYPES = {
        # Numeric (continuous)
        'hr_mean': "continuous",
        'hr_min': "continuous",
        'hr_max': "continuous",
        'hr_std': "continuous",
        'hr_zone_resting': "continuous",
        'hr_zone_moderate': "continuous",
        'hr_zone_vigorous': "continuous",
        'n_steps': "continuous",
        'n_GPS': "continuous",
        'total_distance_km': "continuous",
        'at_home_minute': "continuous",
        'time_in_transition_minutes': "continuous",
        'time_stationary_minutes': "continuous",
        'activity_102_minutes': "continuous",
        'activity_103_minutes': "continuous",
        'activity_104_minutes': "continuous",
        'activity_105_minutes': "continuous",
        'activity_106_minutes': "continuous",
        'activity_107_minutes': "continuous",
        'apparent_temperature_mean': "continuous",
        'sunshine_duration': "continuous",
        'precipitation_hours': "continuous",

        # Categorical
        'somatic_description': "categorical",
        'psychotropic_description': "categorical",
        'employability_description_simple': "categorical",
        'smartphone_type': "categorical",
        'weekend': "categorical",
        'weekday': "categorical",
        'weekday': "categorical",
        'season': "categorical",
        'time_of_day': "categorical",
        'quest_create_hour':"categorical"

    }

    
    # 5) Imputation and Scaling Settings
    # Options for IMPUTE_STRATEGY: "knn", "iterative", "mean", "median", "most_frequent"
    IMPUTE_STRATEGY = "iterative"
    # Options for SCALER_STRATEGY: "standard", "minmax", or None
    SCALER_STRATEGY = "minmax"

    # 6) Regression Analysis Settings
    ANALYSIS = {
        "neg_affect_regression": {
            "TASK_TYPE": "regression",
            "LABEL": LABEL_COL,
            "MODEL_PIPEGRIDS": Regression_model_settings,
            "METRICS": "r2"
        }
    }

    # 7) Outer & Inner Split Parameters
    HOLDOUT_RATIO = 0.1  # ~10% of users held out as new users
    TIME_RATIO = 0.8     # For known users: first 80% for training, last 20% for testing
    N_INNER_CV = 5       # Number of folds for inner cross-validation

    # 8) Holdout Evaluation Parameters
    HOLDOUT_EVAL_RATIO = 0.2   # 20% of each holdout user's data for direct evaluation (scenario a)
    HOLDOUT_ADAPT_RATIO = 0.8  # 80% of each holdout user's data for adaptation (scenario b)
    # Define adaptation strategies to use for holdout users.
    HOLDOUT_ADAPT_STRATEGIES = ["MERF", "NN_embeddings"]


    # 10) Execution Settings
    N_JOBS = 2
    PARALLELIZE = True
    SAVE_MODELS = False
    DEBUG = False
