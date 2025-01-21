import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from intervaltree import IntervalTree
from tqdm import tqdm
import psutil 

#####################################################################################################################

class EMAMapper:
    """
    A class to coordinate the mapping of sensor data to EMA blocks.
    """
    def __init__(self, df_ema, df_data, df_home_clusters=None):
        """
        Initialize the EMAMapper.

        Parameters:
            df_ema (pd.DataFrame): EMA data.
            df_data (pd.DataFrame): Sensor data.
            df_home_clusters (pd.DataFrame, optional): Home clusters data.
        """
        self.df_ema = df_ema.copy()
        self.df_data = df_data.copy()
        self.df_home_clusters = df_home_clusters
        self.sensor_mapper = SensorDataMapper(self.df_ema, self.df_data, self.df_home_clusters)

    def run_mappings(self):
        """
        Run all the mapping methods to enrich the EMA DataFrame with sensor data features.
        """
        self.sensor_mapper.map_heart_rate_to_ema()
        self.sensor_mapper.map_steps_and_metrics_to_ema()
        self.sensor_mapper.map_gps_and_transition_to_ema()
        self.sensor_mapper.map_activity_types_to_ema()
        
        # Update the EMA DataFrame with enriched data
        self.df_ema = self.sensor_mapper.df_ema

    def get_result(self):
        """
        Get the final EMA DataFrame with all mapped features.

        Returns:
            pd.DataFrame: EMA DataFrame enriched with sensor data features.
        """
        return self.df_ema


#####################################################################################################################
#####################################################################################################################

class DataCleaner:
    def __init__(self):
        
        # Initialize counters for entries removed
        self.hr_entries_removed = 0
        self.step_entries_removed = 0

    def clean_heart_rate_data(self, df_hr):
        """
        Clean heart rate data by removing unrealistic values and artifacts.
        """
        df_hr = df_hr.copy()
        
        # List all columns you want to keep. If your raw data has 'endTimestamp',
        # include it here:
        columns_to_keep = ['startTimestamp', 'endTimestamp', 'longValue', 'customer']
        
        # If your raw data might not always have 'endTimestamp', do a set intersection:
        existing_cols = list(set(df_hr.columns).intersection(columns_to_keep))
        df_hr = df_hr[existing_cols].copy()
        
        initial_count = len(df_hr)
        
        # Step 1: Convert to numeric and drop NaNs for 'longValue'
        df_hr['longValue'] = pd.to_numeric(df_hr['longValue'], errors='coerce')
        df_hr = df_hr.dropna(subset=['longValue'])
        after_numeric_conversion = len(df_hr)
        na_removed = initial_count - after_numeric_conversion
        
        # Step 2: Apply physiological thresholds
        min_hr_threshold = 30
        max_hr_threshold = 220
        df_hr = df_hr[
            (df_hr['longValue'] >= min_hr_threshold) &
            (df_hr['longValue'] <= max_hr_threshold)
        ]
        after_thresholds = len(df_hr)
        thresholds_removed = after_numeric_conversion - after_thresholds
        
        # Step 3: Sort by 'startTimestamp' (and optionally 'endTimestamp')
        df_hr = df_hr.sort_values('startTimestamp')
        
        # Summarize
        final_count = len(df_hr)
        total_removed = initial_count - final_count
        print(f"Heart Rate Data Cleaning Summary:")
        print(f"Initial entries: {initial_count}")
        print(f"Removed due to non-numeric values: {na_removed}")
        print(f"Removed due to thresholds: {thresholds_removed}")
        print(f"Total entries removed: {total_removed}")
        print(f"Remaining entries: {final_count}")
        
        return df_hr


    def clean_step_data(self, df_steps):
        """
        Clean steps data by removing unrealistic values.

        Parameters:
            df_steps (pd.DataFrame): Raw steps data.

        Returns:
            pd.DataFrame: Cleaned steps data.
        """
        df_steps = df_steps.copy()
        initial_count = len(df_steps)

        # Ensure 'doubleValue' is numeric
        df_steps['doubleValue'] = pd.to_numeric(df_steps['doubleValue'], errors='coerce')
        df_steps = df_steps.dropna(subset=['doubleValue'])
        after_numeric_conversion = len(df_steps)
        numeric_conversion_removed = initial_count - after_numeric_conversion

        # Remove negative step counts
        df_steps = df_steps[df_steps['doubleValue'] >= 0]
        after_negative_removal = len(df_steps)
        negative_removed = after_numeric_conversion - after_negative_removal

        # Convert timestamps to datetime
        df_steps['startTimestamp'] = pd.to_datetime(df_steps['startTimestamp'], errors='coerce')
        df_steps['endTimestamp'] = pd.to_datetime(df_steps['endTimestamp'], errors='coerce')

        # Calculate duration of each step entry in minutes
        df_steps['duration_min'] = (df_steps['endTimestamp'] - df_steps['startTimestamp']).dt.total_seconds() / 60

        # Remove entries with zero or negative duration
        df_steps = df_steps[df_steps['duration_min'] > 0]
        after_duration_removal = len(df_steps)
        duration_removed = after_negative_removal - after_duration_removal

        # Calculate steps per minute
        df_steps['steps_per_min'] = df_steps['doubleValue'] / df_steps['duration_min']

        # Define physiological threshold for steps per minute
        max_steps_per_minute = 200  # steps per minute

        # Remove entries with steps per minute exceeding the maximum threshold
        df_steps = df_steps[df_steps['steps_per_min'] <= max_steps_per_minute]
        final_count = len(df_steps)
        steps_per_min_removed = after_duration_removal - final_count

        # Drop helper columns
        df_steps.drop(['duration_min', 'steps_per_min'], axis=1, inplace=True, errors='ignore')

        # Calculate total entries removed
        total_removed = initial_count - final_count

        # Print detailed information
        print(f"Step Data Cleaning Summary:")
        print(f"Initial entries: {initial_count}")
        print(f"Removed due to non-numeric values: {numeric_conversion_removed}")
        print(f"Removed due to negative steps: {negative_removed}")
        print(f"Removed due to zero or negative duration: {duration_removed}")
        print(f"Removed due to exceeding steps per minute threshold: {steps_per_min_removed}")
        print(f"Total entries removed: {total_removed}")
        print(f"Remaining entries: {final_count}")

        # Update cumulative counter
        self.step_entries_removed += total_removed

        return df_steps

    def clean_calories_and_distance_data(self, df_data, metric_type):
        """
        Clean the ActiveBurnedCalories and CoveredDistance data by removing unrealistic values.
    
        Parameters:
            df_data (pd.DataFrame): Raw data for the metric.
            metric_type (str): Type of metric ('ActiveBurnedCalories' or 'CoveredDistance').
    
        Returns:
            pd.DataFrame: Cleaned data for the metric.
        """
        initial_count = len(df_data)
    
        # Step 1: Convert 'doubleValue' to numeric
        df_cleaned = df_data.copy()
        df_cleaned['doubleValue'] = pd.to_numeric(df_cleaned['doubleValue'], errors='coerce')
    
        # Step 2: Drop NaNs in 'doubleValue'
        df_cleaned = df_cleaned.dropna(subset=['doubleValue'])
    
        # Step 3: Remove negative values
        df_cleaned = df_cleaned[df_cleaned['doubleValue'] >= 0]
    
        # Step 4: Calculate remaining entries
        final_count = len(df_cleaned)
        entries_removed = initial_count - final_count
    
        # Print cleaning summary
        print(f"{metric_type} Data Cleaning Summary:")
        print(f"Initial entries: {initial_count}")
        print(f"Remaining entries: {final_count}")
        print(f"Entries removed: {entries_removed}")
    
        return df_cleaned


    


#####################################################################################################################
#####################################################################################################################

class SensorDataMapper:
    def __init__(self, df_ema, df_data, df_home_clusters=None):
        """
        Initialize the SensorDataMapper with EMA data, sensor data, and optional home clusters data.

        Parameters:
            df_ema (pd.DataFrame): EMA data.
            df_data (pd.DataFrame): Sensor data.
            df_home_clusters (pd.DataFrame, optional): Home clusters data.
        """
        self.df_ema = df_ema.copy()
        self.df_data = df_data.copy()
        self.df_home_clusters = df_home_clusters
        self.data_cleaner = DataCleaner()


    def map_steps_to_ema(self):
        """
        Map steps data to EMA blocks using explicit filtering for overlaps.
        """
        # Prepare data
        df_steps = self.df_data[self.df_data['type'] == 'Steps'].copy()
        df_steps = self.data_cleaner.clean_step_data(df_steps)
        
        df_steps['startTimestamp'] = pd.to_datetime(df_steps['startTimestamp'])
        df_steps['endTimestamp'] = pd.to_datetime(df_steps['endTimestamp'])
        self.df_ema['sensor_block_start'] = pd.to_datetime(self.df_ema['sensor_block_start'])
        self.df_ema['sensor_block_end'] = pd.to_datetime(self.df_ema['sensor_block_end'])

        # Ensure 'customer' is of type string
        df_steps['customer'] = df_steps['customer'].astype(str)
        self.df_ema['customer'] = self.df_ema['customer'].astype(str)
    
        # Cartesian join on 'customer'
        df_steps['key'] = 0
        self.df_ema['key'] = 0
        df_joined = pd.merge(df_steps, self.df_ema, on=['key', 'customer']).drop(columns=['key'])
    
        # Filter based on overlap conditions
        df_joined = df_joined[
            (df_joined['startTimestamp'] <= df_joined['sensor_block_end']) &
            (df_joined['endTimestamp'] >= df_joined['sensor_block_start'])
        ]
    
        # Calculate overlap duration and weighted steps
        df_joined['overlap_start'] = df_joined[['startTimestamp', 'sensor_block_start']].max(axis=1)
        df_joined['overlap_end'] = df_joined[['endTimestamp', 'sensor_block_end']].min(axis=1)
        df_joined['overlap_duration'] = (df_joined['overlap_end'] - df_joined['overlap_start']).dt.total_seconds()
        df_joined['step_duration'] = (df_joined['endTimestamp'] - df_joined['startTimestamp']).dt.total_seconds()
        df_joined['proportion'] = df_joined['overlap_duration'] / df_joined['step_duration']
        df_joined['weighted_steps'] = df_joined['proportion'] * df_joined['doubleValue']
    
        # Aggregate steps by unique_blocks
        df_steps_summary = df_joined.groupby('unique_blocks')['weighted_steps'].sum().reset_index()
        df_steps_summary.rename(columns={'weighted_steps': 'n_steps'}, inplace=True)
    
        # Merge aggregated results back into EMA
        self.df_ema = pd.merge(self.df_ema, df_steps_summary, on='unique_blocks', how='left')
        self.df_ema['n_steps'] = self.df_ema['n_steps'].fillna(-1).astype(int)
    
        return self.df_ema

    def map_activity_times_to_ema(self):
        """
        Maps activity times (walking, active, biking, running) to EMA blocks and sums up
        the total time for each activity based on overlaps. Assigns -1 for blocks without any data.
        
        Returns:
            pd.DataFrame: The updated EMA DataFrame with time spent on each activity added.
        """
    
        # Define activity types
        activity_types = ['WalkBinary', 'ActiveBinary', 'BikeBinary', 'RunBinary']

    
        # Filter activity data
        df_activity = self.df_data[self.df_data['type'].isin(activity_types)].copy()
    
        # Ensure timestamps are in datetime format
        df_activity['startTimestamp'] = pd.to_datetime(df_activity['startTimestamp'])
        df_activity['endTimestamp'] = pd.to_datetime(df_activity['endTimestamp'])
        self.df_ema['sensor_block_start'] = pd.to_datetime(self.df_ema['sensor_block_start'])
        self.df_ema['sensor_block_end'] = pd.to_datetime(self.df_ema['sensor_block_end'])
    
        # Ensure 'customer' is string type
        df_activity['customer'] = df_activity['customer'].astype(str)
        self.df_ema['customer'] = self.df_ema['customer'].astype(str)
    
        # Filter out rows where 'booleanValue' is False or NaN
        df_activity = df_activity[df_activity['booleanValue'] == True]

    
        # Cartesian join on 'customer'
        df_activity['key'] = 0
        self.df_ema['key'] = 0
        df_joined = pd.merge(df_activity, self.df_ema, on=['key', 'customer']).drop(columns=['key'])
    
        # Filter based on overlaps with EMA blocks
        df_joined = df_joined[
            (df_joined['startTimestamp'] <= df_joined['sensor_block_end']) &
            (df_joined['endTimestamp'] >= df_joined['sensor_block_start'])
        ]
    
        # Calculate overlaps
        df_joined['overlap_start'] = df_joined[['startTimestamp', 'sensor_block_start']].max(axis=1)
        df_joined['overlap_end'] = df_joined[['endTimestamp', 'sensor_block_end']].min(axis=1)
        df_joined['overlap_duration'] = (df_joined['overlap_end'] - df_joined['overlap_start']).dt.total_seconds()
    
        # Calculate time spent in each activity
        activity_times = {}
    
        for activity in activity_types:
            df_activity_filtered = df_joined[df_joined['type'] == activity]
            activity_time = df_activity_filtered.groupby('unique_blocks')['overlap_duration'].sum().reset_index()
            col_name = f"time_{activity.split('Binary')[0].lower()}_minutes"
            activity_time.rename(columns={'overlap_duration': col_name}, inplace=True)
            activity_time[col_name] /= 60  # Convert to minutes
            activity_times[activity] = activity_time
    
        # Merge activity times back into EMA DataFrame
        for activity, activity_time in activity_times.items():
            col_name = f"time_{activity.split('Binary')[0].lower()}_minutes"
            self.df_ema = pd.merge(self.df_ema, activity_time, on='unique_blocks', how='left')
            self.df_ema[col_name] = self.df_ema[col_name].fillna(-1)
    
        return self.df_ema
    

    def map_heart_rate_to_ema(self, compute_stats=True, batch_size=1000):
        """
        Map heart rate data to EMA blocks and compute statistics using batch processing.
    
        Parameters:
            compute_stats (bool): If True, compute detailed heart rate statistics and additional features.
                                  If False, only compute the average heart rate.
            batch_size (int): Number of EMA blocks to process per batch.
    
        Returns:
            pd.DataFrame: The updated EMA DataFrame with heart rate metrics.
        """
    
        # 1. Clean / filter heart rate data
        df_hr_cleaned = self.data_cleaner.clean_heart_rate_data(
            self.df_data[self.df_data['type'] == 'HeartRate']
        )
        
        # If you have both 'startTimestamp' and 'endTimestamp' for heart-rate intervals,
        # ensure they are datetimes:
        df_hr_cleaned['startTimestamp'] = pd.to_datetime(df_hr_cleaned['startTimestamp'])
        df_hr_cleaned['endTimestamp'] = pd.to_datetime(df_hr_cleaned['endTimestamp'])
    
        # Filter HR data by the overall time range of EMA blocks
        min_time = self.df_ema['sensor_block_start'].min()
        max_time = self.df_ema['sensor_block_end'].max()
        df_hr_cleaned = df_hr_cleaned[
            (df_hr_cleaned['startTimestamp'] <= max_time) &
            (df_hr_cleaned['endTimestamp'] >= min_time)
        ]
        # Note: We filter by overlapping intervals rather than just startTimestamp
    
        # 2. Break EMA blocks into batches
        num_batches = (len(self.df_ema) // batch_size) + 1
        results = []
    
        for i in range(num_batches):
            ema_batch = self.df_ema.iloc[i * batch_size : (i + 1) * batch_size]
    
            # Merge on participant; 'inner' keeps only matching customers
            df_joined = ema_batch.merge(df_hr_cleaned, on='customer', how='inner')
    
            # Now filter for any possible overlap between HR intervals and the EMA block
            # Because the HR intervals have start/end, we say:
            # (end of HR interval >= start of block) AND (start of HR interval <= end of block)
            df_joined = df_joined[
                (df_joined['endTimestamp'] >= df_joined['sensor_block_start']) &
                (df_joined['startTimestamp'] <= df_joined['sensor_block_end'])
            ]
            
            if not df_joined.empty:
                if compute_stats:
                    # 
                    def calculate_features(group):
                        """Compute statistical and zone-based features for each EMA block."""
    
                        # 1) Basic stats on the entire set of HR values in this group
                        values = group['longValue'].values
                        features = {
                            'hr_mean': values.mean(),
                            'hr_min': values.min(),
                            'hr_max': values.max(),
                            'hr_std': values.std(),
                            'hr_median': np.median(values),
                            'range_heartrate': values.max() - values.min(),
                            'iqr_heartrate': np.percentile(values, 75) - np.percentile(values, 25),
                            'skewness_heartrate': pd.Series(values).skew(),
                            'kurtosis_heartrate': pd.Series(values).kurtosis(),
                            'hr_peak_counts': np.sum(values > 100),  # Example threshold
                        }
    
                        # 2) Compute the overlap-based zone durations
                        block_start = group['sensor_block_start'].iloc[0]
                        block_end   = group['sensor_block_end'].iloc[0]
                        block_duration = (block_end - block_start).total_seconds()
    
                        # Initialize zone counters (in seconds)
                        resting_sec = 0.0
                        moderate_sec = 0.0
                        vigorous_sec = 0.0
    
                        for idx, row in group.iterrows():
                            # Heart-rate interval
                            hr_start = row['startTimestamp']
                            hr_end   = row['endTimestamp']
    
                            # Overlap with EMA block
                            overlap_start = max(hr_start, block_start)
                            overlap_end   = min(hr_end, block_end)
                            overlap_duration = (overlap_end - overlap_start).total_seconds()
    
                            # If there's a valid overlap
                            if overlap_duration > 0:
                                # Classify by HR value
                                hr_val = row['longValue']
                                if hr_val < 60:
                                    resting_sec += overlap_duration
                                elif hr_val < 100:
                                    moderate_sec += overlap_duration
                                else:
                                    vigorous_sec += overlap_duration
    
                        # 3) Convert total durations to proportions of the entire block
                        if block_duration > 0:
                            features.update({
                                'hr_zone_resting': resting_sec / block_duration,
                                'hr_zone_moderate': moderate_sec / block_duration,
                                'hr_zone_vigorous': vigorous_sec / block_duration,
                            })
                        else:
                            # If block_duration is zero (unlikely), set zones to 0
                            features.update({
                                'hr_zone_resting': 0,
                                'hr_zone_moderate': 0,
                                'hr_zone_vigorous': 0,
                            })
    
                        return pd.Series(features)
    
                    hr_features = df_joined.groupby('unique_blocks').apply(calculate_features).reset_index()
    
                else:
                    # ---------- SIMPLE AVERAGE HEART RATE ----------
                    # For the simpler path, just compute the average 'longValue' in each block
                    # ignoring intervals (or adapt if you want interval weighting)
                    hr_features = (
                        df_joined
                        .groupby('unique_blocks')['longValue']
                        .mean()
                        .reset_index()
                    )
                    hr_features.rename(columns={'longValue': 'avg_heartrate'}, inplace=True)
    
                results.append(hr_features)
    
        # 3. Combine results from all batches
        if results:
            final_features = pd.concat(results, ignore_index=True)
            self.df_ema = pd.merge(self.df_ema, final_features, on='unique_blocks', how='left')
        else:
            # If no heart-rate data matched any EMA block, fill columns with -1
            if compute_stats:
                self.df_ema[
                    [
                        'hr_mean', 'hr_min', 'hr_max', 'hr_std', 'hr_median',
                        'range_heartrate', 'iqr_heartrate',
                        'skewness_heartrate', 'kurtosis_heartrate',
                        'hr_peak_counts', 'hr_zone_resting',
                        'hr_zone_moderate', 'hr_zone_vigorous'
                    ]
                ] = -1
            else:
                self.df_ema['avg_heartrate'] = -1
    
        return self.df_ema

        return self.df_ema


    def map_gps_and_transition_to_ema(self, batch_size=1000):
        """
        Map GPS and transition data to EMA blocks using batch processing,
        including `n_GPS`, `at_home_minute`, and `total_distance_km`.
    
        Parameters:
            batch_size (int): Number of EMA blocks to process per batch.
    
        Returns:
            pd.DataFrame: EMA DataFrame with GPS and transition features added.
        """
    
        if self.df_home_clusters is None:
            raise ValueError("df_home_clusters is not provided.")
    
        # Prepare data
        self.df_home_clusters['customer'] = self.df_home_clusters['customer'].astype(str)
        self.df_ema['customer'] = self.df_ema['customer'].astype(str)
        self.df_home_clusters['startTimestamp'] = pd.to_datetime(self.df_home_clusters['startTimestamp'])
        self.df_ema['sensor_block_start'] = pd.to_datetime(self.df_ema['sensor_block_start'])
        self.df_ema['sensor_block_end'] = pd.to_datetime(self.df_ema['sensor_block_end'])
    
        # Filter GPS data by the overall time range of EMA blocks
        min_time = self.df_ema['sensor_block_start'].min()
        max_time = self.df_ema['sensor_block_end'].max()
        df_gps_filtered = self.df_home_clusters[
            (self.df_home_clusters['startTimestamp'] >= min_time) &
            (self.df_home_clusters['startTimestamp'] <= max_time)
        ]
    
        # Break EMA blocks into batches
        num_batches = (len(self.df_ema) // batch_size) + 1
        results = []
    
        for i in range(num_batches):
            ema_batch = self.df_ema.iloc[i * batch_size: (i + 1) * batch_size]
    
            # Merge EMA with GPS data
            df_joined = ema_batch.merge(df_gps_filtered, on='customer', how='inner')
    
            # Filter for overlaps
            df_joined = df_joined[
                (df_joined['startTimestamp'] >= df_joined['sensor_block_start']) &
                (df_joined['startTimestamp'] <= df_joined['sensor_block_end'])
            ]
    
            if not df_joined.empty:
                def calculate_features(group):
                    # Calculate n_GPS
                    n_GPS = group.shape[0]
    
                    # Calculate total distance in kilometers
                    total_distance_km = group['distance'].sum() / 1000  # Convert meters to kilometers
    
                    # Time-based calculations
                    if group['at_home'].eq(-1).all():
                        at_home_minutes = -1  # No valid home cluster information
                    else:
                        # Calculate durations
                        group['next_startTimestamp'] = group['startTimestamp'].shift(-1).fillna(group['sensor_block_end'])
                        group['duration'] = (group['next_startTimestamp'] - group['startTimestamp']).dt.total_seconds()
                    
                        # Calculate at_home minutes only for valid data
                        at_home_minutes = group[group['at_home'] == 1]['duration'].sum() / 60  # Convert seconds to minutes

    
                    # Moving and stationary durations
                    moving_duration = group[group['transition'] == 1]['time_diff'].sum()
                    stationary_duration = group[group['transition'] == 0]['time_diff'].sum()
                    ema_duration = (group['sensor_block_end'].iloc[0] - group['sensor_block_start'].iloc[0]).total_seconds()
    
                    return pd.Series({
                        'n_GPS': n_GPS,
                        'total_distance_km': total_distance_km,
                        'at_home_minute': at_home_minutes,
                        'time_in_transition_minutes': moving_duration / 60,
                        'time_stationary_minutes': stationary_duration / 60,
                        'prop_time_moving': moving_duration / ema_duration if ema_duration > 0 else 0,
                        'prop_time_stationary': stationary_duration / ema_duration if ema_duration > 0 else 0
                    })
    
                # Compute features for this batch
                gps_features = df_joined.groupby('unique_blocks').apply(calculate_features).reset_index()
                results.append(gps_features)
    
        # Combine results from all batches
        if results:
            final_features = pd.concat(results, ignore_index=True)
            self.df_ema = pd.merge(self.df_ema, final_features, on='unique_blocks', how='left')
    
            # Fill missing values with defaults
            self.df_ema[['n_GPS', 'total_distance_km', 'at_home_minute', 'time_in_transition_minutes',
                         'time_stationary_minutes', 'prop_time_moving', 'prop_time_stationary']] = \
                self.df_ema[['n_GPS', 'total_distance_km', 'at_home_minute', 'time_in_transition_minutes',
                             'time_stationary_minutes', 'prop_time_moving', 'prop_time_stationary']].fillna(-1)
    
        else:
            self.df_ema[['n_GPS', 'total_distance_km', 'at_home_minute', 'time_in_transition_minutes',
                         'time_stationary_minutes', 'prop_time_moving', 'prop_time_stationary']] = -1
    
        return self.df_ema


    def map_steps_and_metrics_to_ema(self, batch_size=1000):
        """
        Map steps, calories, and distance data to EMA blocks using batch processing.
    
        Parameters:
            batch_size (int): Number of EMA blocks to process in each batch.
        """
    
        metric_types = ['Steps', 'ActiveBurnedCalories', 'CoveredDistance']
        metric_results = {metric: [] for metric in metric_types}  # Store results for each metric
    
        for metric in metric_types:
    
            # Prepare and clean data
            df_metric = self.df_data[self.df_data['type'] == metric].copy()
            if metric == 'Steps':
                df_metric = self.data_cleaner.clean_step_data(df_metric)
            elif metric in ['ActiveBurnedCalories', 'CoveredDistance']:
                df_metric = self.data_cleaner.clean_calories_and_distance_data(df_metric, metric)
    
            if df_metric.empty:
                col_name = f"total_{metric.lower()}" if metric != 'Steps' else 'n_steps'
                self.df_ema[col_name] = -1
                continue
    
            # Ensure timestamps and customer formatting
            df_metric['startTimestamp'] = pd.to_datetime(df_metric['startTimestamp'])
            df_metric['endTimestamp'] = pd.to_datetime(df_metric['endTimestamp'])
            df_metric['customer'] = df_metric['customer'].astype(str)
            self.df_ema['sensor_block_start'] = pd.to_datetime(self.df_ema['sensor_block_start'])
            self.df_ema['sensor_block_end'] = pd.to_datetime(self.df_ema['sensor_block_end'])
            self.df_ema['customer'] = self.df_ema['customer'].astype(str)
    
            # Filter data within the overall EMA time range
            min_time = self.df_ema['sensor_block_start'].min()
            max_time = self.df_ema['sensor_block_end'].max()
            df_metric = df_metric[
                (df_metric['startTimestamp'] <= max_time) &
                (df_metric['endTimestamp'] >= min_time)
            ]
    
            # Batch processing for EMA blocks
            num_batches = (len(self.df_ema) // batch_size) + 1
            for i in range(num_batches):
                # Process batch of EMA blocks
                ema_batch = self.df_ema.iloc[i * batch_size:(i + 1) * batch_size]
                ema_batch['key'] = 0
                df_metric['key'] = 0
                df_joined = pd.merge(df_metric, ema_batch, on=['key', 'customer']).drop(columns=['key'])
    
                # Filter based on overlap conditions
                df_joined = df_joined[
                    (df_joined['startTimestamp'] <= df_joined['sensor_block_end']) &
                    (df_joined['endTimestamp'] >= df_joined['sensor_block_start'])
                ]
    
                if not df_joined.empty:
                    # Calculate overlap duration and weighted values
                    df_joined['overlap_start'] = df_joined[['startTimestamp', 'sensor_block_start']].max(axis=1)
                    df_joined['overlap_end'] = df_joined[['endTimestamp', 'sensor_block_end']].min(axis=1)
                    df_joined['overlap_duration'] = (df_joined['overlap_end'] - df_joined['overlap_start']).dt.total_seconds()
                    df_joined['interval_duration'] = (df_joined['endTimestamp'] - df_joined['startTimestamp']).dt.total_seconds()
                    df_joined['weight'] = df_joined['overlap_duration'] / df_joined['interval_duration']
                    df_joined['weighted_value'] = df_joined['doubleValue'] * df_joined['weight']
    
                    # Aggregate by unique_blocks
                    col_name = f"total_{metric.lower()}" if metric != 'Steps' else 'n_steps'
                    metric_summary = df_joined.groupby('unique_blocks')['weighted_value'].sum().reset_index()
                    metric_summary.rename(columns={'weighted_value': col_name}, inplace=True)
                    metric_results[metric].append(metric_summary)
    
        # Combine results for each metric and merge into EMA
        for metric, summaries in metric_results.items():
            if summaries:
                combined_summary = pd.concat(summaries, ignore_index=True)
                col_name = f"total_{metric.lower()}" if metric != 'Steps' else 'n_steps'
                self.df_ema = pd.merge(self.df_ema, combined_summary, on='unique_blocks', how='left')
                self.df_ema[col_name] = self.df_ema[col_name].fillna(-1).astype(int if metric == 'Steps' else float)
    
        return self.df_ema

            
    def map_activity_types_to_ema(self, batch_size=1000):
        """
        Map ActivityType occurrences and durations (in minutes) to EMA blocks.
        """
        # Check if necessary columns are present
        if 'type' not in self.df_data.columns or 'longValue' not in self.df_data.columns:
            print("Error: Required columns 'ActivityType' or 'longValue' are missing from the data.")
            return self.df_ema
    
        print("Available columns in df_data:", self.df_data.columns.tolist())
        print("Available types in 'type' column:", self.df_data['type'].unique())
    
        # Filter for ActivityType data
        df_activities = self.df_data[self.df_data['type'] == 'ActivityType'].copy()
        if df_activities.empty:
            print("Warning: No ActivityType data available.")
            for col in range(102, 108):
                self.df_ema[f'activity_{col}_minutes'] = -1
            return self.df_ema
    
        print("Filtered ActivityType data shape:", df_activities.shape)
    
        # Ensure timestamps and customer columns are properly formatted
        df_activities['startTimestamp'] = pd.to_datetime(df_activities['startTimestamp'])
        df_activities['endTimestamp'] = pd.to_datetime(df_activities['endTimestamp'])
        df_activities['customer'] = df_activities['customer'].astype(str)
        self.df_ema['sensor_block_start'] = pd.to_datetime(self.df_ema['sensor_block_start'])
        self.df_ema['sensor_block_end'] = pd.to_datetime(self.df_ema['sensor_block_end'])
        self.df_ema['customer'] = self.df_ema['customer'].astype(str)
    
        # Filter activities to the overall EMA time range
        min_time = self.df_ema['sensor_block_start'].min()
        max_time = self.df_ema['sensor_block_end'].max()
        df_activities = df_activities[
            (df_activities['startTimestamp'] <= max_time) &
            (df_activities['endTimestamp'] >= min_time)
        ]
        print("ActivityType data after filtering by time range:", df_activities.shape)
    
        # Process EMA blocks in batches
        results = []
        num_batches = (len(self.df_ema) // batch_size) + 1
        for i in range(num_batches):
            print(f"Processing batch {i + 1}/{num_batches}...")
    
            ema_batch = self.df_ema.iloc[i * batch_size:(i + 1) * batch_size]
            ema_batch['key'] = 0
            df_activities['key'] = 0
    
            df_joined = pd.merge(df_activities, ema_batch, on=['key', 'customer']).drop(columns=['key'])
            print(f"Batch {i + 1} joined shape: {df_joined.shape}")
    
            # Filter for overlaps
            df_joined = df_joined[
                (df_joined['startTimestamp'] <= df_joined['sensor_block_end']) &
                (df_joined['endTimestamp'] >= df_joined['sensor_block_start'])
            ]
            print(f"Batch {i + 1} after overlap filtering: {df_joined.shape}")
    
            if not df_joined.empty:
                # Calculate overlap durations
                df_joined['overlap_start'] = df_joined[['startTimestamp', 'sensor_block_start']].max(axis=1)
                df_joined['overlap_end'] = df_joined[['endTimestamp', 'sensor_block_end']].min(axis=1)
                df_joined['overlap_duration'] = (df_joined['overlap_end'] - df_joined['overlap_start']).dt.total_seconds()
    
                # Convert duration to minutes
                df_joined['overlap_minutes'] = df_joined['overlap_duration'] / 60
    
                # Aggregate durations per activity type within each block
                activity_summary = df_joined.groupby(['unique_blocks', 'longValue'])['overlap_minutes'].sum().unstack(fill_value=0)
    
                # Rename columns to reflect activity types
                activity_summary.columns = [f"activity_{int(col)}_minutes" for col in activity_summary.columns]
    
                # Reset index for merging
                activity_summary.reset_index(inplace=True)
                results.append(activity_summary)
    
        if results:
            final_features = pd.concat(results, ignore_index=True)
            print("Before merging results with EMA DataFrame:")
            print(final_features.columns)
            self.df_ema = pd.merge(self.df_ema, final_features, on='unique_blocks', how='left')
    
            # Ensure all activity columns are present
            for col in range(102, 108):
                col_name = f'activity_{col}_minutes'
                if col_name not in self.df_ema.columns:
                    self.df_ema[col_name] = 0
    
            # Assign -1 for blocks with no activity data
            self.df_ema.loc[self.df_ema['unique_blocks'].isin(final_features['unique_blocks']) == False, [f'activity_{col}_minutes' for col in range(102, 108)]] = -1
    
            print("Final EMA DataFrame shape after activity mapping:", self.df_ema.shape)
        else:
            for col in range(102, 108):
                self.df_ema[f'activity_{col}_minutes'] = -1
    
            print("No valid ActivityType overlaps found. Assigned -1 to all activity columns.")
    
        return self.df_ema