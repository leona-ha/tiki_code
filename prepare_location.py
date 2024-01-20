"""
Python Adaptation of the AMPPS tutorial 

Müller, S. R., Bayer, J. B., Ross, M. Q., Mount, J., Stachl, C., Harari, G. M., Chang, Y.-J., Le, H. (2022). 
Analyzing GPS Data for Psychological Research: A Tutorial. Advances in Methods and Practices in Psychological Science, 5(2), 
https://doi.org/10.1177/25152459221082680
 
@author: Leona Hammelrath
"""

# intitialization
# -----------------------------------------------------------------------------
# import packages

import pandas as pd
import datetime as dt
from datetime import date

# set global variables
today = date.today()
week_ago = today - dt.timedelta(days=7)
today = today.strftime("%d%m%Y")
week_ago = week_ago.strftime("%Y-%m-%d")

# path setup
filepath = f"/Users/leonahammelrath/FU_Psychoinformatik/Github/tiki_code/data/export_tiki_{today}/"
filepath_1 = filepath + "epoch_part0001.csv"
filepath_2 = filepath + "epoch_part0002.csv"
filepath_3 = filepath + "epoch_part0003.csv"

# data import
# -----------------------------------------------------------------------------
# Passive data on epoch level

df_1 = pd.read_csv(filepath_1, encoding= "latin-1")
df_2 = pd.read_csv(filepath_2, encoding= "latin-1")
df_3 = pd.read_csv(filepath_3, encoding= "latin-1")

df_complete = pd.concat([df_1, df_2, df_3])
df_complete["customer"] = df_complete.customer.str.split("@").str.get(0)
df_complete["customer"] = df_complete["customer"].str[:4]

# Keep only location data 

df_loc = df_complete[df_complete.type.isin(["Latitude", "Longitude"])]
df_loc = df_loc[["customer", "startTimestamp", "type", "doubleValue","timezoneOffset"]]


# Change UTC-0 to local timezone  

df_loc["timezoneOffset"] = df_loc["timezoneOffset"] * 60000
df_loc["date"] = df_loc["startTimestamp"] + df_loc["timezoneOffset"]
df_loc["date"] = (pd.to_datetime(df_loc["timezone"],unit='ms'))
df_loc["startTimestamp"] = (pd.to_datetime(df_loc["startTimestamp"],unit='ms'))

# Convert df to wide format  

df_wide = df_loc.pivot(index=["customer", "date"],columns="type",values="doubleValue")
df_wide = df_wide.rename_axis(None, axis=1).reset_index()

# Drop duplicates and missings   

#df_wide = df_wide.sort_values(by=["customer", "date"]).drop_duplicates(subset=["date"], keep="last") 
df_wide.dropna(subset = ['Latitude', 'Longitude', 'date'], inplace=True)

# Grab hour and day from timestamp to check for valid entries

df_wide["hour"] = df_wide.startTimestamp.dt.hour
df_wide["day"] = df_wide.startTimestamp.dt.strftime('%Y/%m/%d')

df_wide["n_hours"] = df_wide.groupby(["customer", "day"])["hour"].transform("nunique")
df_wide["n_days"] = df_wide.groupby("customer")["day"].transform("nunique")

# Valid entries have at least 15 hours with at least one measure per day
# Valid entries have at least 3 valid days

df_valid = df_wide.loc[(df_wide.n_hours >= 8) & (df_wide.n_days >= 7)]

# Create unique hourly ID per customer

df_valid.sort_values(by=['customer', 'day'], inplace=True)

# Calculate 'day_index' and 'hourID' by checking for changes in 'date.split' within each 'userID' group
def assign_day_index(user_group):
    user_group['day_index'] = (user_group["day"] != user_group["day"].shift()).cumsum()
    return user_group

df_valid = df_valid.groupby('customer').apply(assign_day_index)

df_valid['hourID'] = df_valid['customer'].astype(str) + '0' + \
                    df_valid['day_index'].astype(str) + '0' + \
                    df_valid['hours'].astype(str)





