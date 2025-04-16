import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
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
        self.sensor_mapper.map_steps_to_ema()
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
        Clean steps data by removing unrealistic values and ensuring proper formatting.
    
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
    
        # Print detailed cleaning summary
        print(f"Step Data Cleaning Summary:")
        print(f"Initial entries: {initial_count}")
        print(f"Removed due to non-numeric values: {numeric_conversion_removed}")
        print(f"Removed due to negative steps: {negative_removed}")
        print(f"Removed due to zero or negative duration: {duration_removed}")
        print(f"Removed due to exceeding steps per minute threshold: {steps_per_min_removed}")
        print(f"Remaining entries: {final_count}")
    
        return df_steps
        


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

    
    def map_steps_to_ema(self, batch_size=20):
        """
        Map steps data to EMA blocks using batch processing by subsets of users.
    
        Parameters:
            batch_size (int): Number of users to process in each batch.
        """
        # Prepare data
        df_steps = self.df_data[self.df_data['type'] == 'Steps'].copy()
        df_steps = self.data_cleaner.clean_step_data(df_steps)
    
        if df_steps.empty:
            print("Warning: No Steps data available.")
            self.df_ema['n_steps'] = -1
            return self.df_ema
    
        # Ensure proper formatting for timestamps and customer column
        df_steps['startTimestamp'] = pd.to_datetime(df_steps['startTimestamp'])
        df_steps['endTimestamp'] = pd.to_datetime(df_steps['endTimestamp'])
        self.df_ema['sensor_block_start'] = pd.to_datetime(self.df_ema['sensor_block_start'])
        self.df_ema['sensor_block_end'] = pd.to_datetime(self.df_ema['sensor_block_end'])
        self.df_ema['customer'] = self.df_ema['customer'].astype(str)
        df_steps['customer'] = df_steps['customer'].astype(str)
    
        # Filter steps data to match the overall EMA time range
        min_time = self.df_ema['sensor_block_start'].min()
        max_time = self.df_ema['sensor_block_end'].max()
        df_steps = df_steps[
            (df_steps['startTimestamp'] <= max_time) &
            (df_steps['endTimestamp'] >= min_time)
        ]
    
        # Get unique users for batch processing
        unique_users = self.df_ema['customer'].unique()
        num_batches = (len(unique_users) // batch_size) + 1
    
        results = []
    
        # Process data in batches of users
        for i in range(num_batches):
            user_batch = unique_users[i * batch_size:(i + 1) * batch_size]
            print(f"Processing user batch {i + 1}/{num_batches}...")
    
            # Filter data for the current batch of users
            ema_batch = self.df_ema[self.df_ema['customer'].isin(user_batch)]
            steps_batch = df_steps[df_steps['customer'].isin(user_batch)]
    
            # Merge EMA blocks with steps data
            ema_batch['key'] = 0
            steps_batch['key'] = 0
            df_joined = pd.merge(steps_batch, ema_batch, on=['key', 'customer']).drop(columns=['key'])
    
            # Filter for overlaps
            df_joined = df_joined[
                (df_joined['startTimestamp'] <= df_joined['sensor_block_end']) &
                (df_joined['endTimestamp'] >= df_joined['sensor_block_start'])
            ]
    
            if not df_joined.empty:
                # Calculate overlap duration and weighted steps
                df_joined['overlap_start'] = df_joined[['startTimestamp', 'sensor_block_start']].max(axis=1)
                df_joined['overlap_end'] = df_joined[['endTimestamp', 'sensor_block_end']].min(axis=1)
                df_joined['overlap_duration'] = (df_joined['overlap_end'] - df_joined['overlap_start']).dt.total_seconds()
                df_joined['step_duration'] = (df_joined['endTimestamp'] - df_joined['startTimestamp']).dt.total_seconds()
                df_joined['proportion'] = df_joined['overlap_duration'] / df_joined['step_duration']
                df_joined['weighted_steps'] = df_joined['proportion'] * df_joined['doubleValue']
    
                # Aggregate steps by unique_blocks
                steps_summary = df_joined.groupby('unique_blocks')['weighted_steps'].sum().reset_index()
                steps_summary.rename(columns={'weighted_steps': 'n_steps'}, inplace=True)
                results.append(steps_summary)
    
        # Combine results from all batches
        if results:
            final_features = pd.concat(results, ignore_index=True)
            self.df_ema = pd.merge(self.df_ema, final_features, on='unique_blocks', how='left')
            self.df_ema['n_steps'] = self.df_ema['n_steps'].fillna(-1).astype(int)
        else:
            # Assign default values if no steps data matched
            self.df_ema['n_steps'] = -1
    
        return self.df_ema


    def map_activity_types_to_ema(self, batch_size=20):
        """
        Map ActivityType occurrences and durations (in minutes) to EMA blocks,
        using batch processing by subsets of users.
    
        Parameters:
            batch_size (int): Number of users to process in each batch.
        """
        # Check if necessary columns are present
        if 'type' not in self.df_data.columns or 'longValue' not in self.df_data.columns:
            print("Error: Required columns 'ActivityType' or 'longValue' are missing from the data.")
            return self.df_ema

    
        # Filter for ActivityType data
        df_activities = self.df_data[self.df_data['type'] == 'ActivityType'].copy()
        if df_activities.empty:
            print("Warning: No ActivityType data available.")
            for col in range(102, 108):
                self.df_ema[f'activity_{col}_minutes'] = -1
            return self.df_ema
    
    
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
    
        # Get unique users for batch processing
        unique_users = self.df_ema['customer'].unique()
        total_users = len(unique_users)
        num_batches = (total_users // batch_size) + 1
    
        results = []
    
        for i in range(num_batches):
            user_batch = unique_users[i * batch_size:(i + 1) * batch_size]
            print(f"Processing user batch {i + 1}/{num_batches}...")
    
            # Filter data for the current batch of users
            ema_batch = self.df_ema[self.df_ema['customer'].isin(user_batch)]
            activities_batch = df_activities[df_activities['customer'].isin(user_batch)]
    
            # Cartesian join
            ema_batch['key'] = 0
            activities_batch['key'] = 0
            df_joined = pd.merge(activities_batch, ema_batch, on=['key', 'customer']).drop(columns=['key'])
            print(f"Batch {i + 1} joined shape: {df_joined.shape}")
    
            # Filter for overlaps
            df_joined = df_joined[
                (df_joined['startTimestamp'] <= df_joined['sensor_block_end']) &
                (df_joined['endTimestamp'] >= df_joined['sensor_block_start'])
            ]
    
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
    
        # Combine all batch results and merge into EMA DataFrame
        if results:
            final_features = pd.concat(results, ignore_index=True)

            self.df_ema = pd.merge(self.df_ema, final_features, on='unique_blocks', how='left')
    
            # Ensure all activity columns are present
            for col in range(102, 108):
                col_name = f'activity_{col}_minutes'
                if col_name not in self.df_ema.columns:
                    self.df_ema[col_name] = 0
    
            # Assign -1 for blocks with no activity data
            self.df_ema.loc[self.df_ema['unique_blocks'].isin(final_features['unique_blocks']) == False, [f'activity_{col}_minutes' for col in range(102, 108)]] = -1
    
        else:
            for col in range(102, 108):
                self.df_ema[f'activity_{col}_minutes'] = -1
    
            print("No valid ActivityType overlaps found. Assigned -1 to all activity columns.")
    
        return self.df_ema

    
    def map_heart_rate_to_ema(self, compute_stats=True, batch_size=20):
        """
        Map heart rate data to EMA blocks and compute statistics using batch processing by subsets of users.
        
        Parameters:
            compute_stats (bool): If True, compute detailed heart rate statistics and additional features.
                                  If False, only compute the average heart rate.
            batch_size (int): Number of users to process in each batch.
        
        Returns:
            pd.DataFrame: The updated EMA DataFrame with heart rate metrics.
        """
        # 1. Clean / filter heart rate data
        df_hr_cleaned = self.data_cleaner.clean_heart_rate_data(
            self.df_data[self.df_data['type'] == 'HeartRate']
        )
        
        if df_hr_cleaned.empty:
            print("Warning: No HeartRate data available.")
            # Assign default values if no HR data is present
            if compute_stats:
                hr_columns = [
                    'hr_mean', 'hr_min', 'hr_max', 'hr_std', 
                    'hr_zone_resting', 'hr_zone_moderate', 'hr_zone_vigorous'
                ]
                self.df_ema[hr_columns] = -1
            else:
                self.df_ema['avg_heartrate'] = -1
            return self.df_ema

        
        # Ensure timestamps and customer columns are properly formatted
        df_hr_cleaned['startTimestamp'] = pd.to_datetime(df_hr_cleaned['startTimestamp'])
        df_hr_cleaned['endTimestamp'] = pd.to_datetime(df_hr_cleaned['endTimestamp'])
        df_hr_cleaned['customer'] = df_hr_cleaned['customer'].astype(str)
        self.df_ema['sensor_block_start'] = pd.to_datetime(self.df_ema['sensor_block_start'])
        self.df_ema['sensor_block_end'] = pd.to_datetime(self.df_ema['sensor_block_end'])
        self.df_ema['customer'] = self.df_ema['customer'].astype(str)
        
        # Filter HR data to the overall EMA time range
        min_time = self.df_ema['sensor_block_start'].min()
        max_time = self.df_ema['sensor_block_end'].max()
        df_hr_cleaned = df_hr_cleaned[
            (df_hr_cleaned['startTimestamp'] <= max_time) &
            (df_hr_cleaned['endTimestamp'] >= min_time)
        ]
        
        # Get unique users for batch processing
        unique_users = self.df_ema['customer'].unique()
        total_users = len(unique_users)
        num_batches = (total_users // batch_size) + 1
        
        results = []
        
        # Process data in batches of users
        for i in range(num_batches):
            user_batch = unique_users[i * batch_size:(i + 1) * batch_size]
            print(f"Processing user batch {i + 1}/{num_batches}...")
            
            # Filter data for the current batch of users
            ema_batch = self.df_ema[self.df_ema['customer'].isin(user_batch)]
            hr_batch = df_hr_cleaned[df_hr_cleaned['customer'].isin(user_batch)]
            
            # Merge on customer
            df_joined = ema_batch.merge(hr_batch, on='customer', how='inner')
            
            # Filter for overlaps
            df_joined = df_joined[
                (df_joined['endTimestamp'] >= df_joined['sensor_block_start']) &
                (df_joined['startTimestamp'] <= df_joined['sensor_block_end'])
            ]
            
            if not df_joined.empty:
                if compute_stats:
                    # Compute statistical and zone-based features for each EMA block
                    def calculate_features(group):
                        values = group['longValue'].values
                        features = {
                            'hr_mean': values.mean(),
                            'hr_min': values.min(),
                            'hr_max': values.max(),
                            'hr_std': values.std(),
                        }
                        
                        # Time spent in HR zones (in minutes)
                        resting_minutes, moderate_minutes, vigorous_minutes = 0.0, 0.0, 0.0
                        for _, row in group.iterrows():
                            overlap_start = max(row['startTimestamp'], group['sensor_block_start'].iloc[0])
                            overlap_end = min(row['endTimestamp'], group['sensor_block_end'].iloc[0])
                            overlap_duration = max((overlap_end - overlap_start).total_seconds(), 0)
                            
                            if overlap_duration > 0:
                                hr_val = row['longValue']
                                duration_minutes = overlap_duration / 60  # Convert seconds to minutes
                                if hr_val < 60:
                                    resting_minutes += duration_minutes
                                elif hr_val < 100:
                                    moderate_minutes += duration_minutes
                                else:
                                    vigorous_minutes += duration_minutes
                        
                        features.update({
                            'hr_zone_resting': resting_minutes,
                            'hr_zone_moderate': moderate_minutes,
                            'hr_zone_vigorous': vigorous_minutes,
                        })
                        
                        return pd.Series(features)
                    
                    hr_features = df_joined.groupby('unique_blocks').apply(calculate_features).reset_index()
                
                else:
                    # Compute average heart rate
                    hr_features = (
                        df_joined
                        .groupby('unique_blocks')['longValue']
                        .mean()
                        .reset_index()
                    )
                    hr_features.rename(columns={'longValue': 'avg_heartrate'}, inplace=True)
                
                results.append(hr_features)
        
        # Combine results from all batches
        if results:
            final_features = pd.concat(results, ignore_index=True)
            self.df_ema = pd.merge(self.df_ema, final_features, on='unique_blocks', how='left')
        
        # Fill NaN values with -1 for missing HR data
        self.df_ema = self.df_ema.fillna(-1)
        
        return self.df_ema


    def map_gps_and_transition_to_ema(self, batch_size=10):
        """
        Map GPS and transition data to EMA blocks using batch processing.
        Process data per subset of users to optimize memory usage.
        
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
    
        # Process in batches of users
        users = self.df_ema['customer'].unique()
        num_batches = (len(users) // batch_size) + 1
        results = []
    
        for i in range(num_batches):
            # Get a subset of users for the current batch
            user_batch = users[i * batch_size: (i + 1) * batch_size]
            ema_batch = self.df_ema[self.df_ema['customer'].isin(user_batch)]
            gps_batch = df_gps_filtered[df_gps_filtered['customer'].isin(user_batch)]
    
            if gps_batch.empty or ema_batch.empty:
                continue
    
            # Merge GPS data with EMA blocks
            df_joined = pd.merge(ema_batch, gps_batch, on='customer', how='inner')
    
            # Filter for overlaps
            df_joined = df_joined[
                (df_joined['startTimestamp'] >= df_joined['sensor_block_start']) &
                (df_joined['startTimestamp'] <= df_joined['sensor_block_end'])
            ]
    
            if not df_joined.empty:
                def calculate_features(group):
                    """Calculate GPS-related features for each EMA block."""
                    n_GPS = group.shape[0]
                    total_distance_km = group['distance'].sum() / 1000  # Convert meters to kilometers
    
                    # Handle at_home_minutes using shift
                    group = group.sort_values('startTimestamp')
                    group['next_startTimestamp'] = group['startTimestamp'].shift(-1, fill_value=group['sensor_block_end'].iloc[-1])
                    group['duration'] = (group['next_startTimestamp'] - group['startTimestamp']).dt.total_seconds()
    
                    at_home_minutes = group[group['at_home'] == 1]['duration'].sum() / 60  # Convert to minutes
    
                    moving_duration = group[group['transition'] == 1]['duration'].sum()
                    stationary_duration = group[group['transition'] == 0]['duration'].sum()
    
                    return pd.Series({
                        'n_GPS': n_GPS,
                        'total_distance_km': total_distance_km,
                        'at_home_minute': at_home_minutes,
                        'time_in_transition_minutes': moving_duration / 60,
                        'time_stationary_minutes': stationary_duration / 60,

                    })
    
                gps_features = df_joined.groupby('unique_blocks').apply(calculate_features).reset_index()
                results.append(gps_features)
    
        # Combine results and merge into EMA DataFrame
        if results:
            final_features = pd.concat(results, ignore_index=True)
            self.df_ema = pd.merge(self.df_ema, final_features, on='unique_blocks', how='left')
    
            # Fill missing values with defaults
            self.df_ema[['n_GPS', 'total_distance_km', 'at_home_minute', 'time_in_transition_minutes',
                         'time_stationary_minutes']] = \
                self.df_ema[['n_GPS', 'total_distance_km', 'at_home_minute', 'time_in_transition_minutes',
                             'time_stationary_minutes']].fillna(-1)
        else:
            # Fill with defaults if no GPS data
            self.df_ema[['n_GPS', 'total_distance_km', 'at_home_minute', 'time_in_transition_minutes',
                         'time_stationary_minutes']] = -1
    
        return self.df_ema
    