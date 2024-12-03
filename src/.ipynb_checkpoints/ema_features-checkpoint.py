import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def binary_na_outcome(df, emotion_columns):
    """
    Analyzes EMA data by calculating the mode of emotional scores for each participant,
    standardizing emotional scores relative to this mode, and classifying changes.

    Parameters:
        df (pd.DataFrame): DataFrame containing EMA data with columns for participant IDs,
                           timestamps, and emotional scores.
        emotion_columns (list): List of column names representing the emotional scores.

    Returns:
        pd.DataFrame: DataFrame with standardized emotional scores and change classification.
    """
    # Calculate mode for each emotion per participant
    def calculate_mode(group):
        return group.mode().iloc[0]

    modes = df.groupby('participant_id').agg({col: calculate_mode for col in emotion_columns}).reset_index()
    modes.columns = ['participant_id'] + [f'{col}_mode' for col in emotion_columns]

    # Merge the modes back into the original DataFrame
    df = df.merge(modes, on='participant_id')

    # Standardize the scores by subtracting the mode
    for col in emotion_columns:
        df[f'{col}_standardized'] = df[col] - df[f'{col}_mode']

    # Define improvement or deterioration based on standardized scores
    def classify_change(score):
        if score > 0:
            return 'Improvement'
        elif score < 0:
            return 'Deterioration'
        else:
            return 'No Change'

    # Classify changes for each emotion
    for col in emotion_columns:
        standardized_col = f'{col}_standardized'
        df[f'{col}_change'] = df[standardized_col].apply(classify_change)

    return df







