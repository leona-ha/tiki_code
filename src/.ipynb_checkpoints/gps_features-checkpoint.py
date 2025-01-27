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

#import hdbscan
from scipy.stats import mode
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)


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

######################################################################

# Home Cluster Extractor Class from Short-term Prediction Paper

######################################################################
class HomeClusterExtractor:
    def __init__(
        self, df, speed_limit, max_distance, epsilon, min_samples, min_nights_obs, min_f_home,
        clustering_method='dbscan', normalize_min_samples=False, min_data_points=10
    ):
        self.df = df.copy()
        # Parameter validation
        if speed_limit <= 0:
            raise ValueError("speed_limit must be positive.")
        if max_distance <= 0:
            raise ValueError("max_distance must be positive.")
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1.")
        if min_nights_obs < 1:
            raise ValueError("min_nights_obs must be at least 1.")
        if not 0 <= min_f_home <= 1:
            raise ValueError("min_f_home must be between 0 and 1.")
        if clustering_method not in ['dbscan', 'hdbscan']:
            raise ValueError("clustering_method must be 'dbscan' or 'hdbscan'.")
        if min_data_points < 1:
            raise ValueError("min_data_points must be at least 1.")

        self.speed_limit = speed_limit
        self.max_distance = max_distance
        self.epsilon = epsilon
        self.min_samples = min_samples
        self.min_nights_obs = min_nights_obs
        self.min_f_home = min_f_home
        self.clustering_method = clustering_method
        self.normalize_min_samples = normalize_min_samples
        self.min_data_points = min_data_points  # Minimum data points threshold

        # Ensure 'startTimestamp' is datetime
        self.df['startTimestamp'] = pd.to_datetime(self.df['startTimestamp'], errors='coerce')
        # Ensure 'Latitude' and 'Longitude' are floats
        self.df['Latitude'] = self.df['Latitude'].astype(float)
        self.df['Longitude'] = self.df['Longitude'].astype(float)

        # Extract hour and day from 'startTimestamp'
        self.df['hour_gps'] = self.df['startTimestamp'].dt.hour
        self.df['day_gps'] = self.df['startTimestamp'].dt.date

    def data_quality_check(self):
        """Filter out customers with insufficient data points."""
        customer_counts = self.df.groupby('customer').size().reset_index(name='point_count')
        valid_customers = customer_counts[customer_counts['point_count'] >= self.min_data_points]['customer']
        self.df = self.df[self.df['customer'].isin(valid_customers)]
        logging.info(f"Data quality check: {len(valid_customers)} customers with sufficient data retained.")

    def calculate_distances_and_speeds(self):
        """Calculate distances and speeds for each customer."""
        self.df['distance'], self.df['time_diff'], self.df['speed'] = np.nan, np.nan, np.nan

        for customer in self.df['customer'].unique():
            mask = self.df['customer'] == customer
            customer_data = self.df.loc[mask].sort_values('startTimestamp')

            distances = self._calculate_distances(customer_data)
            time_diffs = customer_data['startTimestamp'].diff().dt.total_seconds().fillna(0)
            speeds = distances / time_diffs.replace(0, np.nan)

            # Assign calculated values using customer_data.index to ensure alignment
            self.df.loc[customer_data.index, 'distance'] = distances
            self.df.loc[customer_data.index, 'time_diff'] = time_diffs
            self.df.loc[customer_data.index, 'speed'] = speeds

    def calculate_stationary_and_transition(self):
        """Determine stationary points and transition status based on speed and distance."""
        # Filter out points with speed > 220 km/h (converted to m/s)
        self.df = self.df[self.df['speed'] <= 220 * 1000 / 3600]
        self.df['stationary'] = (self.df['speed'] < self.speed_limit) & (self.df['distance'] < self.max_distance)
        self.df['transition'] = np.where(self.df['stationary'], 0, 1)
        return self.df

    def _calculate_distances(self, df):
        """Helper method to calculate distances using haversine formula."""
        coords = df[['Latitude', 'Longitude']].values
        distances = np.array([
            self._haversine(coords[i - 1][1], coords[i - 1][0], coords[i][1], coords[i][0])
            for i in range(1, len(coords))
        ])
        return np.append(distances, 0)

    def _haversine(self, lon1, lat1, lon2, lat2):
        """Haversine formula to calculate distance between two lat/lon points in meters."""
        R = 6371000  # Earth radius in meters
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    def apply_clustering(self, df):
        """Apply clustering based on the selected method."""
        df_cleaned = df.dropna(subset=['Longitude', 'Latitude'])

        clusters = []
        for customer_id, group_df in df_cleaned.groupby('customer'):
            # Ensure 'customer' column is in the DataFrame
            group_df = group_df.copy()
            group_df['customer'] = customer_id
            cluster_result = self._apply_clustering_method(group_df)
            clusters.append(cluster_result)

        # Concatenate the results
        cluster_results = pd.concat(clusters, ignore_index=True)
        return cluster_results

    def _apply_clustering_method(self, df):
        """Helper method to apply the chosen clustering method."""
        customer_point_count = len(df)
        customer_id = df['customer'].iloc[0]

        # Skip clustering for customers with too few points
        if customer_point_count < self.min_data_points:
            logging.info(f"Customer {customer_id} has too few data points ({customer_point_count}). Skipping clustering.")
            df['cluster'] = -1
            return df

        # Use normalized min_samples or a default value
        if self.normalize_min_samples:
            min_samples = max(2, int(customer_point_count * 0.03))  # 3% of points, with a minimum of 2
        else:
            min_samples = self.min_samples

        if self.clustering_method == 'dbscan':
            clustering_model = DBSCAN(eps=self.epsilon, min_samples=min_samples, metric="haversine")
            cluster_labels = clustering_model.fit_predict(df[['Longitude', 'Latitude']].apply(np.radians))
            cluster_labels = cluster_labels.astype(int)
            df['cluster'] = cluster_labels
            return df
        elif self.clustering_method == 'hdbscan':
            min_cluster_size = max(2, min(min_samples, customer_point_count))
            clustering_model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='haversine')
            cluster_labels = clustering_model.fit_predict(df[['Longitude', 'Latitude']].apply(np.radians))
            cluster_labels = cluster_labels.astype(int)
            df['cluster'] = cluster_labels
            return df
        else:
            raise ValueError(f"Invalid clustering method: {self.clustering_method}")

    def find_home_cluster(self, geodata_clusters):
        """Identify the home cluster based on nighttime data, with fallback to the largest cluster."""
        # Filter for night hours
        geodata_night = geodata_clusters.loc[
            (geodata_clusters['hour_gps'] >= 20) | (geodata_clusters['hour_gps'] <= 6)
        ].copy()

        # Initialize the 'home' column to None
        geodata_clusters['home'] = np.nan

        # Time-based home cluster assignment: most frequent cluster at night
        if not geodata_night.empty:
            # Exclude noise from home assignment during night hours
            valid_clusters_night = geodata_night[geodata_night['cluster'] != -1].copy()

            if not valid_clusters_night.empty:
                # Function to compute mode safely
                def safe_mode(x):
                    modes = x.mode()
                    if len(modes) == 1:
                        return modes.iloc[0]
                    elif len(modes) > 1:
                        return modes.min()  # Choose the smallest cluster label in case of multiple modes
                    else:
                        return np.nan

                # Calculate the most frequent cluster (mode) per customer at night
                valid_clusters_night['home'] = valid_clusters_night.groupby('customer')['cluster'].transform(safe_mode)

                # Calculate the number of unique nights with observations for each customer
                valid_clusters_night['nights_with_obs'] = valid_clusters_night.groupby('customer')['day_gps'].transform('nunique')

                # Count the number of points in the identified home cluster per customer
                valid_clusters_night['n_home'] = valid_clusters_night.groupby(['customer', 'home'])['day_gps'].transform('size')

                # Calculate the total number of night-time points for each customer
                valid_clusters_night['night_obs'] = valid_clusters_night.groupby('customer')['day_gps'].transform('size')

                # Calculate the fraction of night-time points spent at home
                valid_clusters_night['f_home'] = valid_clusters_night['n_home'] / valid_clusters_night['night_obs']

                # Apply both conditions: Minimum nights observed and minimum fraction of time spent at home
                valid_clusters_night['home'] = valid_clusters_night.apply(
                    lambda x: x['home'] if (x['nights_with_obs'] >= self.min_nights_obs) and (x['f_home'] >= self.min_f_home) else np.nan, axis=1
                )

                # Merge the time-based home assignment back into the main dataframe
                home_mapping = valid_clusters_night[['customer', 'home']].drop_duplicates(subset=['customer'])
                geodata_clusters = geodata_clusters.merge(home_mapping, on='customer', how='left', suffixes=('', '_temp'))
                # Use 'home_temp' where 'home' is NaN
                geodata_clusters['home'] = geodata_clusters['home'].fillna(geodata_clusters['home_temp'])
                geodata_clusters.drop(columns=['home_temp'], inplace=True)

        # Fallback: Assign the largest cluster per customer from all data points if no home is found
        no_home_customers = geodata_clusters.loc[geodata_clusters['home'].isna(), 'customer'].unique()
        logging.info(f"Customers with no home after time-based method: {len(no_home_customers)}")

        if len(no_home_customers) > 0:
            # Consider all points (not just night-time) for customers with no home cluster
            fallback_home_clusters = (
                geodata_clusters[geodata_clusters['customer'].isin(no_home_customers) & (geodata_clusters['cluster'] != -1)]
                .groupby(['customer', 'cluster'])
                .size()
                .reset_index(name='cluster_size')
            )

            if not fallback_home_clusters.empty:
                # Take the largest cluster per customer based on all data points
                fallback_home_clusters = fallback_home_clusters.loc[
                    fallback_home_clusters.groupby('customer')['cluster_size'].idxmax()
                ]

                # Assign the fallback home clusters
                fallback_home_clusters['home'] = fallback_home_clusters['cluster']

                # Merge fallback home clusters back to the main dataset
                fallback_home_mapping = fallback_home_clusters[['customer', 'home']].drop_duplicates()
                geodata_clusters = geodata_clusters.merge(fallback_home_mapping, on='customer', how='left', suffixes=('', '_fallback'))

                # Fill any remaining NaNs in the 'home' column with the fallback cluster
                geodata_clusters['home'] = geodata_clusters['home'].fillna(geodata_clusters['home_fallback'])
                geodata_clusters.drop(columns=['home_fallback'], inplace=True)
                logging.info(f"Fallback home clusters assigned: {len(fallback_home_clusters)}")

        # For customers that still have no home cluster after the fallback
        final_no_home = geodata_clusters.loc[geodata_clusters['home'].isna(), 'customer'].unique()
        logging.warning(f"{len(final_no_home)} customers still do not have a home cluster.")

        # Create homeID by combining customer and home cluster
        geodata_clusters['homeID'] = geodata_clusters.apply(
            lambda x: f"{x['customer']}00{int(x['home'])}" if pd.notna(x['home']) else None, axis=1
        )

        return geodata_clusters

    def determine_if_at_home(self, df):
        """Determine if a person is at home, handling unclustered points (-1) properly."""
        # Convert cluster to integer before creating the clusterID and homeID
        df['cluster'] = df['cluster'].astype(int)
        df['home'] = df['home'].astype(float)
        df['home'] = df['home'].astype(int, errors='ignore')  # Handle NaNs gracefully

        # Create clusterID and homeID, ensuring no decimal points
        df['clusterID'] = df.apply(lambda x: f"{x['customer']}00{int(x['cluster'])}" if x['cluster'] != -1 else None, axis=1)
        df['homeID'] = df.apply(lambda x: f"{x['customer']}00{int(x['home'])}" if pd.notna(x['home']) else None, axis=1)

        # Check if a person is at home (-1 if no valid cluster/home)
        df['at_home'] = df.apply(
            lambda x: -1 if x['cluster'] == -1 else (1 if x['clusterID'] == x['homeID'] else 0), axis=1
        )
        return df

    def run(self):
        """Run the full extraction process."""
        self.data_quality_check()
        self.calculate_distances_and_speeds()
        self.df = self.calculate_stationary_and_transition()

        # Apply clustering based on all data (not just stationary points)
        geodata_cluster_df = self.apply_clustering(self.df)

        # Merge clustering results back to the original dataframe, including transition points
        geodata_clusters = geodata_cluster_df.copy()
        # Fill NaNs in 'cluster' with -1
        geodata_clusters['cluster'] = geodata_clusters['cluster'].fillna(-1)

        # Find home cluster with fallback
        geodata_clusters = self.find_home_cluster(geodata_clusters)

        # Determine if the person is at home
        geodata_clusters = self.determine_if_at_home(geodata_clusters)

        return geodata_clusters







