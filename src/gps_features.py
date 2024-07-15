import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.impute import SimpleImputer


import statistics
import math
from scipy import stats
from heapq import nlargest
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt, log
import string
import pickle 
import pingouin as pg

# Adjusted from Mueller et al., (2021); https://doi.org/ 10.1038/s41598-021-93087-x

# Haversine formula to calculate distance between two lat/lon points in meters
def haversine(lon1, lat1, lon2, lat2):
    R = 6371000  # Radius of Earth in meters
    phi_1 = np.radians(lat1)
    phi_2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = np.sin(delta_phi/2.0)**2 + np.cos(phi_1) * np.cos(phi_2) * np.sin(delta_lambda/2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    meters = R * c  # Output distance in meters
    return meters

def apply_clustering(df, epsilon, min_samples):
    def db2(x):
        clustering_model = DBSCAN(eps=epsilon, min_samples=min_samples, metric="haversine")
        cluster_labels = clustering_model.fit_predict(x[['Longitude', 'Latitude']].apply(np.radians))
        return pd.DataFrame({'cluster_100m': cluster_labels})
    
    # Group by 'customer' and apply clustering function
    geodata_cluster_df = df.groupby('customer').apply(lambda x: db2(x)).reset_index()
    return geodata_cluster_df

def adaptive_kmeans(x, Dkmeans=500, max_k=50):
    """
    Perform K-Means clustering with an adaptive number of clusters.

    Parameters:
    x (pd.DataFrame): The data to cluster, with 'Longitude' and 'Latitude' columns.
    Dkmeans (float): The maximum allowed distance from any point to its cluster center (in meters).
    max_k (int): The maximum number of clusters to test.

    Returns:
    pd.Series: Cluster labels for each point in the data.
    """
    X = x[['Longitude', 'Latitude']].apply(np.radians)
    
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(X)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_

        # Convert centers back to degrees for distance calculation
        centers_degrees = np.degrees(centers)

        # Calculate the maximum distance of any point to its cluster center
        max_distance = 0
        for i in range(k):
            cluster_points = x[labels == i]
            if not cluster_points.empty:
                distances = cluster_points.apply(
                    lambda row: haversine(row['Longitude'], row['Latitude'], centers_degrees[i][0], centers_degrees[i][1]), axis=1
                )
                if distances.max() > max_distance:
                    max_distance = distances.max()

        if max_distance <= Dkmeans:
            print(f"Number of clusters selected: {k}")
            return pd.Series(labels, index=x.index)

    print(f"Maximum number of clusters reached: {max_k}")
    return pd.Series(labels, index=x.index)


def identify_home(df):
    df['home'] = df.groupby('customer')['clusterID'].transform(lambda x: statistics.mode(x))
    return df

def isclose(loc1, loc2, threshold=30):
    """
    Determine if two locations are within a given distance of each other.

    Parameters:
    - loc1, loc2: Tuples representing (latitude, longitude) of two locations.
    - threshold: Distance in meters below which locations are considered "close". Default is 30 meters.

    Returns:
    - True if the distance between loc1 and loc2 is less than the threshold; otherwise, False.
    """
    return haversine(loc1[1], loc1[0], loc2[1], loc2[0]) < threshold


def cal_entropy(sig_locs):
    """
    Calculate entropy and normalized entropy.
    
    :param sig_locs: tuple of tuple of lat, lon, start, end, dur
    :return: entropy, normalized_entropy
    """
    if len(sig_locs) <= 1:
        return np.nan, np.nan
    # Calculate the total duration for each cluster
    clusters = {}
    for lat, lon, _, _, dur, _ in sig_locs:
        if (lat, lon) in clusters:
            clusters[(lat, lon)] += dur
        else:
            clusters[(lat, lon)] = dur
    values = np.array(list(clusters.values())).astype(np.float64)  # Change np.float to np.float64
    probs = values / values.sum()
    probs[probs == 0] = 1e-10  # Avoid log(0) by setting zero probabilities to a very small positive number
    ent = -probs.dot(np.log(probs))

    # normalized entropy ranges from 0 to 1. If N clusters are defined, we normalize the max entropy
    # (when every case is distributed equally as 1/N)
    norm_ent = ent / (log(len(clusters)) + 1e-09)
    return ent, norm_ent

def calculate_metrics(df, group_by=['customer']):
    """
    Calculate raw entropy, entropy, normalized entropy, total distance traveled, percentage of time at home,
    number of unique clusters visited, number of non-unique clusters visited for each group defined by `group_by`.

    Parameters:
    df (pd.DataFrame): DataFrame with 'customer', 'clusterID', 'home', 'Latitude', 'Longitude', 'startTimestamp', and optional 'day' columns.
    group_by (list): List of column names to group by. Default is ['customer'].

    Returns:
    pd.DataFrame: DataFrame with group levels, raw entropy, entropy, normalized entropy, total distance,
                  percentage time at home, number of unique clusters, total clusters visited.
    """
    results = []

    for group_keys, group in df.groupby(group_by):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)

        group_results = {key: val for key, val in zip(group_by, group_keys)}

        group = group.sort_values(by='startTimestamp')
        group['time_spent'] = group['startTimestamp'].diff().shift(-1).fillna(pd.Timedelta(seconds=0)).dt.total_seconds()

        valid_labels = group['clusterID'][group['clusterID'] != -1]
        if len(valid_labels) == 0:
            group_results.update({
                'raw_entropy': np.nan,
                'entropy': np.nan,
                'normalized_entropy': np.nan,
                'total_distance': 0,
                'percentage_time_at_home': np.nan,
                'num_unique_clusters': 0,
                'num_total_clusters': 0
            })
            results.append(group_results)
            continue

        # Calculate clusters entropy
        sig_locs = [(row['Latitude'], row['Longitude'], row['startTimestamp'], row['startTimestamp'] + pd.Timedelta(seconds=row['time_spent']), row['time_spent'], row['clusterID']) for index, row in group.iterrows()]
        entropy, normalized_entropy = cal_entropy(sig_locs)
        
        group_results['entropy'] = entropy
        group_results['normalized_entropy'] = normalized_entropy

        # Calculate the total distance using the Haversine formula
        latitudes = group['Latitude'].to_numpy()
        longitudes = group['Longitude'].to_numpy()
        total_distance = sum(haversine(lon1, lat1, lon2, lat2) for lon1, lat1, lon2, lat2 in zip(longitudes[:-1], latitudes[:-1], longitudes[1:], latitudes[1:]))

        # Calculate percentage of time spent at home if home is not NaN
        total_duration = group['time_spent'].sum()
        if pd.notna(group['home'].iloc[0]):
            home_cluster = group['home'].iloc[0]
            time_at_home = group.loc[group['clusterID'] == home_cluster, 'time_spent'].sum()
            percentage_time_at_home = (time_at_home / total_duration) * 100 if total_duration > 0 else 0
        else:
            percentage_time_at_home = np.nan

        # Number of unique clusters
        num_unique_clusters = len(set(valid_labels))

        # Number of total clusters
        num_total_clusters = len(group['clusterID'].unique())

        # Raw location entropy based on non-clustered GPS coordinates
        lat_bins = pd.cut(group['Latitude'], bins=10)
        lon_bins = pd.cut(group['Longitude'], bins=10)
        raw_entropy = 0
        bin_counts = group.groupby([lat_bins, lon_bins]).size()
        total_points = bin_counts.sum()

        for count in bin_counts:
            p_bin = count / total_points
            if p_bin > 0:
                raw_entropy -= p_bin * np.log(p_bin)

        group_results.update({
            'total_distance': total_distance,
            'percentage_time_at_home': percentage_time_at_home,
            'num_unique_clusters': num_unique_clusters,
            'num_total_clusters': num_total_clusters,
            'raw_entropy': raw_entropy
        })

        results.append(group_results)

    return pd.DataFrame(results)




def calculate_transition_time(df, group_by=['customer']):
    """
    Calculate the transition time percentage for the raw, unfiltered GPS data for each group defined by `group_by`.

    Parameters:
    df (pd.DataFrame): DataFrame with 'customer', 'Latitude', 'Longitude', 'startTimestamp', and optional 'day' columns.
    group_by (list): List of column names to group by. Default is ['customer'].

    Returns:
    pd.DataFrame: DataFrame with group levels and transition time percentage.
    """
    results = []

    for group_keys, group in df.groupby(group_by):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)

        group_results = {key: val for key, val in zip(group_by, group_keys)}

        # Compute the transition time
        if len(group) > 1:
            group = group.sort_values(by='startTimestamp')
            latitudes = group['Latitude'].to_numpy()
            longitudes = group['Longitude'].to_numpy()
            times = group['startTimestamp'].to_numpy()

            distances = np.array([
                haversine(lon1, lat1, lon2, lat2)
                for lon1, lat1, lon2, lat2 in zip(longitudes[:-1], latitudes[:-1], longitudes[1:], latitudes[1:])
            ])

            time_deltas = np.diff(times) / np.timedelta64(1, 's')  # Convert time deltas to seconds
            speeds = distances / time_deltas
            moving_time = np.sum(time_deltas[speeds >= 1.4])
            total_time = np.sum(time_deltas)

            transition_time_percentage = (moving_time / total_time) * 100 if total_time > 0 else 0
        else:
            transition_time_percentage = 0.0

        group_results['transition_time'] = transition_time_percentage
        results.append(group_results)

    return pd.DataFrame(results)

def normalize_data(data):
    """
    Normalize skewed data using a log transformation.
    
    Parameters:
    data (pd.Series): The data to transform.
    
    Returns:
    pd.Series: The normalized data.
    """
    # Add a small constant to ensure all values are positive
    shift_value = abs(data.min()) + 1 if data.min() <= 0 else 0
    normalized_data = np.log(data + shift_value)
    
    return normalized_data

def calculate_retest_reliability(geodata_cluster_merged, features):
    # Initialize a dictionary to store the correlation results
    correlations = {}
    
    # Calculate the correlation for each feature
    for feature in features:
        feature_first = feature + '_first'
        feature_second = feature + '_second'
        
        if feature_first in geodata_cluster_merged.columns and feature_second in geodata_cluster_merged.columns:
            # Drop rows with NaN values in the relevant columns
            clean_data = geodata_cluster_merged[[feature_first, feature_second]].dropna()
            
            if not clean_data.empty:
                # Normalize the data if it is skewed
                clean_data[feature_first] = normalize_data(clean_data[feature_first])
                clean_data[feature_second] = normalize_data(clean_data[feature_second])
                
                correlation = clean_data[feature_first].corr(clean_data[feature_second])
                correlations[feature] = correlation
            else:
                correlations[feature] = np.nan
        else:
            print(f"Feature columns for {feature} are missing in the dataframe.")
    
    return correlations


def calculate_intraclass_coefficient(geodata_cluster_merged, features):
    # Initialize a dictionary to store the ICC results
    iccs = {}

    # Calculate the ICC for each feature
    for feature in features:
        feature_first = feature + '_first'
        feature_second = feature + '_second'
        
        if feature_first in geodata_cluster_merged.columns and feature_second in geodata_cluster_merged.columns:
            # Drop rows with NaN values in the relevant columns
            clean_data = geodata_cluster_merged[[feature_first, feature_second]].dropna()
            
            if len(clean_data) >= 5:
                # Print the clean data for debugging
                print(f"Clean data for feature {feature}:")
                print(clean_data.head())

                # Reshape data for ICC calculation
                long_data = pd.DataFrame({
                    'subject': np.repeat(clean_data.index, 2),
                    'measurement': pd.concat([clean_data[feature_first], clean_data[feature_second]]).values,
                    'time': ['first'] * len(clean_data) + ['second'] * len(clean_data)
                })

                # Print the reshaped data for debugging
                print(f"Long data for feature {feature}:")
                print(long_data.head())

                # Check if data is balanced and has at least 5 subjects with both time points
                count_per_subject = long_data.groupby(['subject', 'time']).size().unstack(fill_value=0)
                if (count_per_subject.min(axis=1) >= 1).sum() >= 5:
                    try:
                        # Calculate ICC(2,1) for single measurements
                        icc = pg.intraclass_corr(data=long_data, targets='subject', raters='time', ratings='measurement', nan_policy='omit')
                        icc_value = icc[icc['Type'] == 'ICC2,1']['ICC'].values[0]
                        iccs[feature] = icc_value
                    except Exception as e:
                        print(f"Error calculating ICC for feature {feature}: {e}")
                        iccs[feature] = np.nan
                else:
                    print(f"Data for feature {feature} is unbalanced or insufficient data points.")
                    iccs[feature] = np.nan
            else:
                print(f"Not enough non-missing values for feature {feature}.")
                iccs[feature] = np.nan
        else:
            print(f"Feature columns for {feature} are missing in the dataframe.")
    
    return iccs







