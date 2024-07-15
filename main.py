"""
Created in July 2024
by author Leona Hammelrath
"""

import os
import glob
import pickle
import sys

import pandas as pd
import datetime as dt
from datetime import date, datetime
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from config import datapath, proj_sheet

# ===========================
# Import data
# ===========================

# set paths for data import
TODAY = date.today().strftime("%d%m%Y")
TODAY_DAY = pd.to_datetime('today').normalize()
TODAY = "08072024"

datapath_tiki = datapath + f"raw/export_tiki_{TODAY}/"
file_pattern = os.path.join(datapath_tiki, "epoch_part*.csv")

# files are stored in multiple .csv files. add them together
file_list = glob.glob(file_pattern)
file_list.sort()

df_complete = pd.concat((pd.read_csv(f, encoding="latin-1", low_memory=False) for f in file_list), ignore_index=True)
df_monitoring = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{proj_sheet}/export?format=csv")






if __name__ == "__main__":
    main()