import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt


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
        Run the mapping methods to enrich the EMA DataFrame with sensor data features.
        """
        self.sensor_mapper.map_steps_to_ema()
        self.sensor_mapper.map_heartrate_to_ema()
        self.sensor_mapper.map_heart_rate_features_to_ema()
        # Add other mapping methods as needed

        # If GPS and transition data mapping is needed, call the mapping method
        if self.df_home_clusters is not None:
            self.sensor_mapper.map_gps_and_transition_to_ema()

        # Update the EMA DataFrame with the mapped data
        self.df_ema = self.sensor_mapper.df_ema

    def get_result(self):
        """
        Get the final EMA DataFrame with all mapped features.

        Returns:
            pd.DataFrame: EMA DataFrame with sensor data features.
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

        Parameters:
            df_hr (pd.DataFrame): Raw heart rate data.

        Returns:
            pd.DataFrame: Cleaned heart rate data.
        """
        df_hr = df_hr.copy()
        initial_count = len(df_hr)

        # Step 1: Convert to numeric and drop NaNs
        df_hr['longValue'] = pd.to_numeric(df_hr['longValue'], errors='coerce')
        df_hr = df_hr.dropna(subset=['longValue'])
        after_numeric_conversion = len(df_hr)
        na_removed = initial_count - after_numeric_conversion

        # Step 2: Apply physiological thresholds
        min_hr_threshold = 30  # Adjusted from 40
        max_hr_threshold = 230  # Adjusted from 220
        df_hr = df_hr[
            (df_hr['longValue'] >= min_hr_threshold) &
            (df_hr['longValue'] <= max_hr_threshold)
        ]
        after_thresholds = len(df_hr)
        thresholds_removed = after_numeric_conversion - after_thresholds

        # Step 3: Sort by timestamp
        df_hr.sort_values('startTimestamp', inplace=True)

        # Step 4: Apply smoothing before calculating change rate
        window_size = 5  # Adjust as needed
        df_hr['longValue_smoothed'] = df_hr['longValue'].rolling(window=window_size, min_periods=1).mean()

        # Step 5: Calculate change rate with smoothed data
        df_hr['time_diff'] = df_hr['startTimestamp'].diff().dt.total_seconds().fillna(0)
        df_hr['hr_diff'] = df_hr['longValue_smoothed'].diff().fillna(0)
        df_hr['hr_change_rate'] = df_hr['hr_diff'] / df_hr['time_diff'].replace(0, np.nan)

        # Step 6: Adjust maximum change rate threshold
        max_change_rate = 30  # Adjusted from 20
        df_hr = df_hr[
            df_hr['hr_change_rate'].abs() <= max_change_rate
        ]
        final_count = len(df_hr)
        change_rate_removed = after_thresholds - final_count

        # Summarize entries removed
        total_removed = initial_count - final_count
        print(f"Heart Rate Data Cleaning Summary:")
        print(f"Initial entries: {initial_count}")
        print(f"Removed due to non-numeric values: {na_removed}")
        print(f"Removed due to thresholds: {thresholds_removed}")
        print(f"Removed due to exceeding change rate threshold: {change_rate_removed}")
        print(f"Total entries removed: {total_removed}")
        print(f"Remaining entries: {final_count}")

        # Update cumulative counter
        self.hr_entries_removed += total_removed

        # Proceed with resampling and further steps
        df_hr.set_index('startTimestamp', inplace=True)
        df_hr_numeric = df_hr[['longValue_smoothed']].copy()
        df_hr_numeric = df_hr_numeric.resample('1s').mean()

        # If 'customer' is needed
        if 'customer' in df_hr.columns:
            df_hr_numeric['customer'] = df_hr['customer'].iloc[0]

        df_hr_numeric.reset_index(inplace=True)
        df_hr_numeric.rename(columns={'longValue_smoothed': 'longValue'}, inplace=True)

        # Drop helper columns if they exist
        df_hr_numeric.drop(['time_diff', 'hr_diff', 'hr_change_rate'], axis=1, inplace=True, errors='ignore')

        return df_hr_numeric


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
        self.data_cleaner = DataCleaner()#
        
    def map_steps_to_ema(self):
        # Existing code...

        n_steps_values = []

        # Filter for steps data
        df_steps = self.df_data[self.df_data['type'] == 'Steps'].copy()

        # Clean steps data
        df_steps_cleaned = self.data_cleaner.clean_step_data(df_steps)

        # Iterate over EMA blocks
        for idx, ema_row in self.df_ema.iterrows():
            sensor_block_start = ema_row['sensor_block_start']
            sensor_block_end = ema_row['sensor_block_end']
            customer = ema_row['customer']

            # Calculate EMA block duration in minutes
            ema_duration_min = (sensor_block_end - sensor_block_start).total_seconds() / 60

            # Define maximum feasible steps for this EMA block
            max_steps_per_minute = 200  # steps per minute
            max_feasible_steps = max_steps_per_minute * ema_duration_min

            # Filter steps data that overlaps with the EMA block and for the same customer
            df_filtered = df_steps_cleaned[
                (df_steps_cleaned['customer'] == customer) &
                (df_steps_cleaned['startTimestamp'] < sensor_block_end) &
                (df_steps_cleaned['endTimestamp'] > sensor_block_start)
            ]

            if df_filtered.empty:
                n_steps_values.append(np.nan)  # Use NaN to indicate missing data
            else:
                # Clip the start and end times to the EMA block window
                overlap_start = df_filtered['startTimestamp'].clip(lower=sensor_block_start, upper=sensor_block_end)
                overlap_end = df_filtered['endTimestamp'].clip(lower=sensor_block_start, upper=sensor_block_end)
                overlap_duration = (overlap_end - overlap_start).dt.total_seconds()  # Convert to seconds

                # Calculate the duration of each steps interval
                step_duration = (df_filtered['endTimestamp'] - df_filtered['startTimestamp']).dt.total_seconds()

                # Calculate the proportion of each steps interval that overlaps with the EMA block
                proportion = overlap_duration / step_duration

                # Weight the steps by the overlap proportion
                weighted_value = proportion * df_filtered['doubleValue']

                # Sum the weighted steps values
                n_steps = weighted_value.sum()

                # Check if n_steps exceeds the maximum feasible steps for the EMA block
                if n_steps <= max_feasible_steps:
                    n_steps_values.append(round(n_steps))
                else:
                    # Handle entries exceeding the maximum feasible steps
                    print(f"EMA Block {idx} for customer {customer} exceeds maximum feasible steps.")
                    n_steps_values.append(np.nan)  # Or handle as appropriate (e.g., set to max_feasible_steps)

        # Add the 'n_steps' column to the EMA DataFrame
        self.df_ema['n_steps'] = n_steps_values
        return self.df_ema

    def map_heartrate_to_ema(self):
        """
        Map heart rate data to EMA blocks.
        """
        avg_hr_values = []
    
        # Filter for HeartRate data
        df_hr = self.df_data[self.df_data['type'] == 'HeartRate'].copy()
    
        # Clean heart rate data
        df_hr_cleaned = self.data_cleaner.clean_heart_rate_data(df_hr)
    
        # Ensure timestamps are in datetime format
        df_hr_cleaned['startTimestamp'] = pd.to_datetime(df_hr_cleaned['startTimestamp'])
        self.df_ema['sensor_block_start'] = pd.to_datetime(self.df_ema['sensor_block_start'])
        self.df_ema['sensor_block_end'] = pd.to_datetime(self.df_ema['sensor_block_end'])
    
        # Iterate over EMA blocks
        for idx, ema_row in self.df_ema.iterrows():
            sensor_block_start = ema_row['sensor_block_start']
            sensor_block_end = ema_row['sensor_block_end']
            customer = ema_row['customer']
    
            # Filter heart rate data within the EMA block for the same customer
            df_filtered = df_hr_cleaned[
                (df_hr_cleaned['customer'] == customer) &
                (df_hr_cleaned['startTimestamp'] >= sensor_block_start) &
                (df_hr_cleaned['startTimestamp'] <= sensor_block_end)
            ]
    
            if df_filtered.empty or df_filtered['longValue'].isnull().all():
                avg_hr_values.append(np.nan)  # Indicate missing data with NaN
            else:
                # Remove NaN values before calculation
                hr_values = df_filtered['longValue'].dropna().values
    
                # Calculate the average heart rate
                if len(hr_values) > 0:
                    avg_hr = np.mean(hr_values)
                    avg_hr_values.append(avg_hr)
                else:
                    avg_hr_values.append(np.nan)
    
        # Add the 'avg_heartrate' column to the EMA DataFrame
        self.df_ema['avg_heartrate'] = avg_hr_values
        return self.df_ema


    def map_heart_rate_features_to_ema(self):
        """
        Map additional heart rate features to EMA blocks.

        Returns:
            pd.DataFrame: EMA DataFrame with new heart rate feature columns added.
        """
        # Ensure required columns exist in df_data
        required_hr_columns = ['startTimestamp', 'endTimestamp', 'longValue', 'type', 'customer']
        missing_hr_columns = [col for col in required_hr_columns if col not in self.df_data.columns]
        if missing_hr_columns:
            raise ValueError(f"df_data is missing required columns for HeartRate data: {missing_hr_columns}")

        # Initialize lists to hold feature values
        min_hr_values = []
        max_hr_values = []
        range_hr_values = []
        std_hr_values = []
        median_hr_values = []
        iqr_hr_values = []
        skewness_hr_values = []
        kurtosis_hr_values = []
        hr_zone_resting = []
        hr_zone_moderate = []
        hr_zone_vigorous = []
        hr_peak_counts = []
        hr_slope_values = []

        # Filter for HeartRate data
        df_hr = self.df_data[self.df_data['type'] == 'HeartRate'].copy()

        # Convert timestamps to datetime
        df_hr['startTimestamp'] = pd.to_datetime(df_hr['startTimestamp'], errors='raise')
        df_hr['endTimestamp'] = pd.to_datetime(df_hr['endTimestamp'], errors='raise')

        # Ensure 'customer' is string type
        df_hr['customer'] = df_hr['customer'].astype(str)

        # Clean heart rate data
        df_hr_cleaned = self.data_cleaner.clean_heart_rate_data(df_hr)

        # Iterate over EMA blocks
        for idx, ema_row in self.df_ema.iterrows():
            sensor_block_start = ema_row['sensor_block_start']
            sensor_block_end = ema_row['sensor_block_end']
            customer = ema_row['customer']

            # Filter HeartRate data that overlaps with the EMA block and for the same customer
            df_filtered = df_hr_cleaned[
                (df_hr_cleaned['customer'] == customer) &
                (df_hr_cleaned['startTimestamp'] < sensor_block_end) &
                (df_hr_cleaned['endTimestamp'] > sensor_block_start)
            ].copy()

            if df_filtered.empty:
                # Append NaN to feature lists
                min_hr_values.append(np.nan)
                max_hr_values.append(np.nan)
                range_hr_values.append(np.nan)
                std_hr_values.append(np.nan)
                median_hr_values.append(np.nan)
                iqr_hr_values.append(np.nan)
                skewness_hr_values.append(np.nan)
                kurtosis_hr_values.append(np.nan)
                hr_zone_resting.append(np.nan)
                hr_zone_moderate.append(np.nan)
                hr_zone_vigorous.append(np.nan)
                hr_peak_counts.append(np.nan)
                hr_slope_values.append(np.nan)
            else:
                # Extract heart rate values
                hr_values = df_filtered['longValue'].values

                # Calculate features
                min_hr = hr_values.min()
                max_hr = hr_values.max()
                range_hr = max_hr - min_hr
                std_hr = hr_values.std()
                median_hr = np.median(hr_values)
                q25, q75 = np.percentile(hr_values, [25, 75])
                iqr_hr = q75 - q25
                skewness_hr = pd.Series(hr_values).skew()
                kurtosis_hr = pd.Series(hr_values).kurtosis()

                # Time spent in heart rate zones
                df_filtered['duration'] = (df_filtered['endTimestamp'] - df_filtered['startTimestamp']).dt.total_seconds()
                total_duration = df_filtered['duration'].sum()

                # Calculate durations in each zone
                resting_duration = df_filtered[df_filtered['longValue'] < 60]['duration'].sum()
                moderate_duration = df_filtered[(df_filtered['longValue'] >= 60) & (df_filtered['longValue'] < 100)]['duration'].sum()
                vigorous_duration = df_filtered[df_filtered['longValue'] >= 100]['duration'].sum()

                # Calculate proportions
                hr_zone_resting.append(resting_duration / total_duration if total_duration > 0 else np.nan)
                hr_zone_moderate.append(moderate_duration / total_duration if total_duration > 0 else np.nan)
                hr_zone_vigorous.append(vigorous_duration / total_duration if total_duration > 0 else np.nan)

                # Number of heart rate peaks (above 100 bpm as an example)
                hr_peak_count = np.sum(hr_values > 100)
                hr_peak_counts.append(hr_peak_count)

                # Trend analysis (slope)
                timestamps = df_filtered['startTimestamp'].astype(np.int64) / 1e9  # Convert to seconds
                if len(hr_values) > 1:
                    slope = np.polyfit(timestamps, hr_values, 1)[0]
                else:
                    slope = 0
                hr_slope_values.append(slope)

                # Append calculated features
                min_hr_values.append(min_hr)
                max_hr_values.append(max_hr)
                range_hr_values.append(range_hr)
                std_hr_values.append(std_hr)
                median_hr_values.append(median_hr)
                iqr_hr_values.append(iqr_hr)
                skewness_hr_values.append(skewness_hr)
                kurtosis_hr_values.append(kurtosis_hr)

        # Add new columns to the EMA DataFrame
        self.df_ema['min_heartrate'] = min_hr_values
        self.df_ema['max_heartrate'] = max_hr_values
        self.df_ema['range_heartrate'] = range_hr_values
        self.df_ema['std_heartrate'] = std_hr_values
        self.df_ema['median_heartrate'] = median_hr_values
        self.df_ema['iqr_heartrate'] = iqr_hr_values
        self.df_ema['skewness_heartrate'] = skewness_hr_values
        self.df_ema['kurtosis_heartrate'] = kurtosis_hr_values
        self.df_ema['hr_zone_resting'] = hr_zone_resting
        self.df_ema['hr_zone_moderate'] = hr_zone_moderate
        self.df_ema['hr_zone_vigorous'] = hr_zone_vigorous
        self.df_ema['hr_peak_counts'] = hr_peak_counts
        self.df_ema['hr_slope'] = hr_slope_values

    def map_gps_and_transition_to_ema(self):
        """
        Map GPS and transition data to EMA blocks.
        """
        if self.df_home_clusters is None:
            raise ValueError("df_home_clusters is not provided.")

        # Ensure required columns exist in df_home_clusters
        required_home_columns = ['startTimestamp', 'distance', 'transition', 'at_home', 'customer']
        missing_home_columns = [col for col in required_home_columns if col not in self.df_home_clusters.columns]
        if missing_home_columns:
            raise ValueError(f"df_home_clusters is missing required columns: {missing_home_columns}")

        gps_counts = []
        total_distances = []
        transition_values = []
        transition_minute_values = []
        at_home_minute_values = []
        at_home_binary_values = []

        # Ensure that 'distance' is numeric
        self.df_home_clusters['distance'] = pd.to_numeric(self.df_home_clusters['distance'], errors='coerce')
        self.df_home_clusters = self.df_home_clusters.dropna(subset=['distance'])

        # Ensure 'transition' and 'at_home' are numeric (0 or 1)
        self.df_home_clusters['transition'] = pd.to_numeric(self.df_home_clusters['transition'], errors='coerce')
        self.df_home_clusters['at_home'] = pd.to_numeric(self.df_home_clusters['at_home'], errors='coerce')

        # Ensure 'customer' is string type
        self.df_home_clusters['customer'] = self.df_home_clusters['customer'].astype(str)

        # Iterate over EMA blocks
        for idx, ema_row in self.df_ema.iterrows():
            sensor_block_start = ema_row['sensor_block_start']
            sensor_block_end = ema_row['sensor_block_end']
            customer = ema_row['customer']

            # Filter the home clusters for the current customer and time window
            df_filtered = self.df_home_clusters[
                (self.df_home_clusters['customer'] == customer) &
                (self.df_home_clusters['startTimestamp'] >= sensor_block_start) &
                (self.df_home_clusters['startTimestamp'] <= sensor_block_end)
            ].copy()

            if df_filtered.empty:
                # No data for the block, set to default values
                gps_counts.append(0)  # Assign 0 to gps_counts
                total_distances.append(np.nan)
                transition_values.append(np.nan)
                transition_minute_values.append(np.nan)
                at_home_minute_values.append(np.nan)
                at_home_binary_values.append(np.nan)
            else:
                # Count GPS points
                gps_count = df_filtered.shape[0]
                gps_counts.append(gps_count)

                # Calculate total distance (sum of distances)
                total_distance = df_filtered['distance'].sum() / 1000  # Convert to kilometers
                total_distances.append(total_distance)

                # Sort data by timestamp
                df_filtered.sort_values('startTimestamp', inplace=True)

                # Calculate time differences between consecutive GPS points within the block
                df_filtered['time_diff'] = df_filtered['startTimestamp'].diff().dt.total_seconds().fillna(0)

                # Calculate transition minutes and at_home minutes
                transition_minutes = df_filtered[df_filtered['transition'] == 1]['time_diff'].sum() / 60  # Convert seconds to minutes
                at_home_minutes = df_filtered[df_filtered['at_home'] == 1]['time_diff'].sum() / 60  # Convert seconds to minutes

                transition_minute_values.append(transition_minutes)
                at_home_minute_values.append(at_home_minutes)

                # Calculate at_home_binary
                if df_filtered['at_home'].eq(1).any():
                    # At least one GPS point at home
                    at_home_binary_values.append(1)
                else:
                    # There is data, but no points at home
                    at_home_binary_values.append(0)

                # Transition status
                if transition_minutes > 0:
                    transition_status = 1  # Some transition occurred
                else:
                    transition_status = 0  # No transition occurred
                transition_values.append(transition_status)

        # Add new columns to the EMA DataFrame
        self.df_ema['n_GPS'] = gps_counts
        self.df_ema['total_distance_km'] = total_distances
        self.df_ema['transition'] = transition_values
        self.df_ema['transition_minutes'] = transition_minute_values
        self.df_ema['at_home_minute'] = at_home_minute_values
        self.df_ema['at_home_binary'] = at_home_binary_values

        return self.df_ema
