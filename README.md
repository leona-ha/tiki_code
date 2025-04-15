# PREACT-digital

This repository contains the codebase for the **PREACT-digital** project, associated with the paper:

> **Towards JITAI – Moment-to-moment Prediction of Negative Affect in Internalizing Disorders using Digital Phenotyping**

## Repository Structure

- **model_pipeline/**  
  Contains all model definitions, training scripts, and evaluation code related to the predictive modeling of negative affect.

- **src/**  
  Houses preprocessing functions and utilities required to transform raw data into a format suitable for model training and evaluation.

- **notebooks/**  
  - **04_Passive_Preprocess.ipynb**  
    - Notebook for executing data preprocessing steps. It relies on functions from `src/` to clean and prepare raw data.
  - **X1_Short_Term_Prediction.ipynb**  
    - Notebook for running the model pipeline. It utilizes modules in `model_pipeline/` and processes the output from the preprocessing step.
