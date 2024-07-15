import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import statistics
import math
from scipy import stats
from heapq import nlargest
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt
import string
from sklearn.impute import SimpleImputer
import pickle 


def age_mapping(x):
    """
    Map age to categories

    :param x: <int>, the age

    :return <str>: the age category
    """
    if x < 25:
        return '18 to 25'
    elif x < 35:
        return '25 to 34'
    elif x < 45:
        return '35 to 44'
    elif x < 55:
        return '45 to 54'
    elif x < 65:
        return '55 to 64'
    elif x < 75:
        return '65 to 74'
    elif x < 85:
        return '75 to 84'
    elif x >= 85:
        return 'Greater than 85'
    else:
        return 'Not answered'


def calculate_outcome_measures(df_redcap):
    # Step 1: Calculate the change in BSI-GSI score from baseline to T20
    df_redcap['bsi_gsi_change'] = df_redcap['bsi_gsi_t20'] - df_redcap['bsi_gsi_base']

    # Step 2: Calculate the Reliable Change Index (RCI)
    std_dev = df_redcap['bsi_gsi_base'].std()  # Standard deviation of BSI-GSI scores

    df_redcap['rci'] = (df_redcap['bsi_gsi_change'] * (2 ** 0.5)) / std_dev

    # Step 3: Determine clinically significant change
    df_redcap['clinically_significant'] = np.where(df_redcap['bsi_gsi_t20'] < 0.56, True, False)

    # Step 4: Determine if change is reliable
    df_redcap['reliable_change'] = abs(df_redcap['rci']) >= 1.96  # Assuming critical value for 95% confidence interval

    # Step 5: Create binary column for both clinically significant and reliable change
    df_redcap['clinically_reliable_change'] = np.where((df_redcap['clinically_significant']) & (df_redcap['reliable_change']), 1, 0)

    return df_redcap







