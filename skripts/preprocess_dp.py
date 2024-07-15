"""
The script imports  different data sources, namely:
- EMA and passive data
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

datapath_tiki = datapath + f"raw/export_tiki_{today}/"
file_pattern = os.path.join(datapath_tiki, "epoch_part*.csv")

# files are stored in multiple .csv files. add them together
file_list = glob.glob(file_pattern)
file_list.sort()


df_complete = pd.concat((pd.read_csv(f, encoding="latin-1", low_memory=False) for f in file_list), ignore_index=True)
df_redcap_zert = pd.read_csv(datapath + "ZERTIFIZIERUNGFOR518_DATA_2024-05-21_0911.csv", low_memory=False)
df_redcap = pd.read_csv(datapath + f"FOR5187_DATA_2024-05-21_0910.csv", low_memory=False)
df_monitoring = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{proj_sheet}/export?format=csv")
