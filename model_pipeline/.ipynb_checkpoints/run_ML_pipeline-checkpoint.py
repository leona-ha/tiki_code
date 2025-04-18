#################################
# run_ML_pipeline.py
#################################

import os
import sys
import argparse
import importlib
from os.path import join
from datetime import datetime
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Local imports
from ML_pipeline import MLpipeline

def main():
    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run ML Pipeline with Configuration')
    parser.add_argument('config', type=str, help='Name of the configuration module (without .py)')
    parser.add_argument('-d', '--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()
    config_name = args.config
    runDEBUG = args.debug

    # 2. Import config dynamically
    try:
        config_module = importlib.import_module(f'ML_Config')  # Adjust if config is in a subfolder
    except ImportError:
        print(f"Error: Could not import config module 'ML_Config'")
        sys.exit(1)
    cfg = config_module.Config
    cfg_path = os.path.abspath(sys.modules[cfg.__module__].__file__)

    # 3. Set output directory based on config path
    OUTPUT_DIR = cfg_path.replace('ML_Config.py', 'results')
    if runDEBUG:
        print("=" * 40 + "\nRunning DEBUG MODE\n" + "=" * 40)
        cfg.N_INNER_CV = 2
        cfg.HOLDOUT_RATIO = 0.1
        cfg.TIME_RATIO = 0.8
        cfg.N_JOBS = 1
        cfg.PARALLELIZE = False
        cfg.SAVE_MODELS = False
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 4. Iterate through each PKL file defined in config
    for pkl_file in cfg.PKL_FILES:
        start_time = datetime.now()
        pkl_name = os.path.basename(pkl_file).replace(".pkl", "")
        SAVE_DIR = join(OUTPUT_DIR, pkl_name, start_time.strftime("%Y%m%d-%H%M"))
        os.makedirs(SAVE_DIR, exist_ok=True)
        # Save the config file in the results folder for reproducibility
        cpy_cfg = join(SAVE_DIR, f'ML_Config_{start_time.strftime("%Y%m%d_%H%M")}.py')
        os.system(f"cp {cfg_path} {cpy_cfg}")

        # 5. Create an instance of MLpipeline and load data
        mlp = MLpipeline(config=cfg, random_state=42)

        # Load the data from .pkl and set to pipeline
        try:
            with open(pkl_file, 'rb') as f:
                df = pickle.load(f)
            mlp.set_data(df)
        except Exception as e:
            print(f"Error loading data from {pkl_file}: {e}")
            continue

        # 6. Perform outer and inner splits
        mlp.outer_user_split()
        mlp.inner_time_split()

        # 7. Select the model pipeline/grid combos (for regression in this example)
        pipeline_grid_tuples = cfg.ANALYSIS["neg_affect_regression"]["MODEL_PIPEGRIDS"]
        scoring = cfg.ANALYSIS["neg_affect_regression"]["METRICS"]
        task_type = cfg.ANALYSIS["neg_affect_regression"]["TASK_TYPE"]

        # 8. Run the pipeline on inner splits
        try:
            results_timebased = mlp.run(pipeline_grid_dict=pipeline_grid_tuples,
                                        task_type=task_type, scoring=scoring)
        except Exception as e:
            print(f"Error running pipelines on inner splits: {e}")
            continue

        # 9. Evaluate on holdout set
        try:
            results_holdout = mlp.evaluate_holdout_all(results_timebased)
        except Exception as e:
            print(f"Error during holdout evaluation: {e}")
            results_holdout = []

        # 10. Save or consolidate results (e.g., into CSV)
        try:
            df_results_timebased = pd.DataFrame(results_timebased)
            df_results_timebased.to_csv(join(SAVE_DIR, "timebased_results.csv"), index=False)

            if results_holdout:
                df_results_holdout = pd.DataFrame(results_holdout)
                df_results_holdout.to_csv(join(SAVE_DIR, "holdout_results.csv"), index=False)
        except Exception as e:
            print(f"Error saving results: {e}")

        total_runtime = str(datetime.now() - start_time).split(".")[0]
        print(f"TOTAL RUNTIME: {total_runtime} secs")

if __name__ == "__main__":
    sys.exit(main())

