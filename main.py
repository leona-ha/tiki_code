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
# backup passive data
backup_file_pattern = os.path.join(datapath, 'raw/tiki_backup_files/tiki_backup_*.csv')  # Adjust the path and extension if needed
backup_files = glob.glob(backup_file_pattern)
# files are stored in multiple .csv files. add them together
file_list = glob.glob(file_pattern)
file_list.sort()

df_complete = pd.concat((pd.read_csv(f, encoding="latin-1", low_memory=False) for f in file_list), ignore_index=True)
df_monitoring = pd.read_csv(f"https://docs.google.com/spreadsheets/d/{proj_sheet}/export?format=csv")

# read in data that are stored in backup
dataframes = []
for file in backup_files:
    df_backup = pd.read_csv(file, encoding="latin-1", low_memory=False)  # Adjust read_csv parameters as needed
    # Extract the date from the filename
    filename_parts = os.path.basename(file).split('_')
    date_str = filename_parts[2]
    time_suffix = int(filename_parts[3].split('.')[0])  # Convert the suffix to an integer
    date = pd.to_datetime(date_str)
    df_backup['file_date'] = date
    df_backup['time_suffix'] = time_suffix
    dataframes.append(df_backup)

df_backup_complete = pd.concat(dataframes, ignore_index=True)
df_backup_complete = df_backup_complete.sort_values(by=['file_date', 'time_suffix'])




if __name__ == "__main__":
    main()