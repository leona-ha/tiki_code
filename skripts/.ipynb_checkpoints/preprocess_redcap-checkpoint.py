"""
The script imports the different data sources, namely:
- EMA and passive data
- Redcap data 
- Monitoring data 

then it does basic preprocessing, like datetime transformations 
@author: Leona Hammelrath
"""

import os
import glob
import pickle
import sys

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from redcap_features import age_mapping, calculate_outcome_measures
import pandas as pd
import datetime as dt
from datetime import date, datetime
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from config import datapath, proj_sheet

# set paths for data import
today = date.today().strftime("%d%m%Y")
today_day = pd.to_datetime('today').normalize()

