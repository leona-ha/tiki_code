clean_heart_rate_data <- function(df_hr) {
  # Load required libraries
  library(dplyr)
  library(zoo)
  
  # Make a copy of the input data frame (not strictly necessary in R since we're not mutating in place)
  df_hr <- df_hr %>% as.data.frame()

  # Define which columns to keep
  columns_to_keep <- c("startTimestamp", "longValue", "customer")
  
  # Select only the desired columns
  df_hr <- df_hr %>% select(all_of(columns_to_keep))
  
  # Store the initial number of entries
  initial_count <- nrow(df_hr)
  
  #------------------------------------------------------
  # Step 1: Convert 'longValue' to numeric and drop NAs
  #------------------------------------------------------
  
  # Convert 'longValue' to numeric, coercing any non-numeric entries to NA
  df_hr <- df_hr %>%
    mutate(longValue = as.numeric(longValue))
  
  # Drop rows where 'longValue' is NA
  df_hr <- df_hr %>% drop_na(longValue)
  
  # Count how many rows remain after numeric conversion and NA removal
  after_numeric_conversion <- nrow(df_hr)
  
  # Calculate how many were removed due to non-numeric values
  na_removed <- initial_count - after_numeric_conversion
  
  #------------------------------------------------------
  # Step 2: Apply physiological thresholds
  # Keep only heart rates between 30 and 220
  #------------------------------------------------------
  
  min_hr_threshold <- 30
  max_hr_threshold <- 220
  
  df_hr <- df_hr %>%
    filter(longValue >= min_hr_threshold & longValue <= max_hr_threshold)
  
  # Count how many remain after threshold filtering
  after_thresholds <- nrow(df_hr)
  
  # Calculate how many were removed due to thresholds
  thresholds_removed <- after_numeric_conversion - after_thresholds
  
  #------------------------------------------------------
  # Step 3: Sort by timestamp
  #------------------------------------------------------
  
  df_hr <- df_hr %>%
    arrange(startTimestamp)
  
  #------------------------------------------------------
  # Step 4: Apply smoothing before calculating change rate
  # We'll use a rolling mean with a window size of 5
  #------------------------------------------------------
  
  window_size <- 5
  
  # Use zoo::rollmean with partial=TRUE to mimic min_periods=1
  df_hr <- df_hr %>%
    mutate(longValue_smoothed = zoo::rollmean(longValue, k = window_size, fill = NA, align = "right"))
  
  # For the first rows where the window isn't full, rollmean returns NA by default, 
  # so we can replace those with a progressively computed mean:
  # Recalculate with partial window: 
  df_hr$longValue_smoothed <- zoo::rollapply(df_hr$longValue, width = window_size, 
                                             FUN = mean, partial = TRUE, align = "right")
  
  #------------------------------------------------------
  # Step 5: Calculate change rate with smoothed data
  # time_diff = difference in seconds between consecutive entries
  # hr_diff = difference in the smoothed heart rate between consecutive entries
  # hr_change_rate = hr_diff / time_diff
  #------------------------------------------------------
  
  # Calculate time differences in seconds
  # We assume startTimestamp is POSIXct. If not, convert it first.
  df_hr <- df_hr %>%
    mutate(time_diff = as.numeric(difftime(startTimestamp, lag(startTimestamp), units = "secs")),
           time_diff = if_else(is.na(time_diff), 0, time_diff))
  
  # Calculate heart rate differences (smoothed)
  df_hr <- df_hr %>%
    mutate(hr_diff = longValue_smoothed - lag(longValue_smoothed, default = longValue_smoothed[1]))
  
  # Compute hr_change_rate = hr_diff / time_diff
  # Replace division by zero with NA
  df_hr <- df_hr %>%
    mutate(hr_change_rate = if_else(time_diff == 0, NA_real_, hr_diff / time_diff))
  
  #------------------------------------------------------
  # Step 6: Adjust maximum change rate threshold
  # Filter out entries where abs(hr_change_rate) > 30
  #------------------------------------------------------
  
  max_change_rate <- 30
  
  # Before filtering, record how many rows we had
  before_change_rate_filter <- nrow(df_hr)
  
  df_hr <- df_hr %>%
    filter(abs(hr_change_rate) <= max_change_rate | is.na(hr_change_rate))
  
  # Count how many remain after change rate filtering
  final_count <- nrow(df_hr)
  
  # Calculate how many were removed due to change rate
  change_rate_removed <- before_change_rate_filter - final_count
  
  #------------------------------------------------------
  # Summarize entries removed
  #------------------------------------------------------
  
  total_removed <- initial_count - final_count
  
  message("Heart Rate Data Cleaning Summary:")
  message(sprintf("Initial entries: %d", initial_count))
  message(sprintf("Removed due to non-numeric values: %d", na_removed))
  message(sprintf("Removed due to thresholds: %d", thresholds_removed))
  message(sprintf("Removed due to exceeding change rate threshold: %d", change_rate_removed))
  message(sprintf("Total entries removed: %d", total_removed))
  message(sprintf("Remaining entries: %d", final_count))
  
  # Return the cleaned data frame
  return(df_hr)
}

map_heart_rate_to_ema <- function(self, compute_stats=TRUE, batch_size=5000) {
  library(dplyr)
  library(moments)
  
  # Clean heart rate data
  df_hr_cleaned <- self$data_cleaner$clean_heart_rate_data(
    self$df_data %>% filter(type == "HeartRate")
  )
  
  # Filter HR data by the overall time range of EMA blocks
  min_time <- min(self$df_ema$sensor_block_start, na.rm=TRUE)
  max_time <- max(self$df_ema$sensor_block_end, na.rm=TRUE)
  
  df_hr_cleaned <- df_hr_cleaned %>%
    filter(startTimestamp >= min_time & startTimestamp <= max_time)
  
  # Calculate number of batches
  num_batches <- (nrow(self$df_ema) %/% batch_size) + 1
  
  results <- list()
  
  # Define a function to calculate features if compute_stats = TRUE
  if (compute_stats) {
    calculate_features <- function(group) {
      # 'group' is a subset of df_joined corresponding to one 'unique_blocks'
      values <- group$longValue
      timestamps <- as.numeric(group$startTimestamp) # seconds since epoch
      
      hr_mean <- mean(values)
      hr_min <- min(values)
      hr_max <- max(values)
      hr_std <- sd(values)
      hr_median <- median(values)
      range_heartrate <- hr_max - hr_min
      iqr_heartrate <- diff(quantile(values, c(0.25, 0.75)))
      skewness_heartrate <- skewness(values)
      kurtosis_heartrate <- kurtosis(values)
      hr_peak_counts <- sum(values > 100)
      
      # Compute slope using linear regression if more than one data point
      hr_slope <- if (length(values) > 1) {
        coef(lm(values ~ timestamps))[2]
      } else {
        0
      }
      
      # Compute durations
      # c(0, diff(...)) simulates the 'diff' in Python, with the first entry as 0
      group$duration <- c(0, diff(as.numeric(group$startTimestamp)))
      total_duration <- sum(group$duration, na.rm=TRUE)
      
      # Compute HR zones (if total_duration > 0)
      hr_zone_resting <- if (total_duration > 0) {
        sum(group$duration[group$longValue < 60], na.rm=TRUE) / total_duration
      } else {
        -1
      }
      
      hr_zone_moderate <- if (total_duration > 0) {
        sum(group$duration[group$longValue >= 60 & group$longValue < 100], na.rm=TRUE) / total_duration
      } else {
        -1
      }
      
      hr_zone_vigorous <- if (total_duration > 0) {
        sum(group$duration[group$longValue >= 100], na.rm=TRUE) / total_duration
      } else {
        -1
      }
      
      # Return a data frame (one row) with all features
      data.frame(
        unique_blocks = unique(group$unique_blocks),
        hr_mean = hr_mean,
        hr_min = hr_min,
        hr_max = hr_max,
        hr_std = hr_std,
        hr_median = hr_median,
        range_heartrate = range_heartrate,
        iqr_heartrate = iqr_heartrate,
        skewness_heartrate = skewness_heartrate,
        kurtosis_heartrate = kurtosis_heartrate,
        hr_peak_counts = hr_peak_counts,
        hr_zone_resting = hr_zone_resting,
        hr_zone_moderate = hr_zone_moderate,
        hr_zone_vigorous = hr_zone_vigorous,
        hr_slope = hr_slope,
        stringsAsFactors = FALSE
      )
    }
  }
  
  # Process EMA blocks in batches
  for (i in seq_len(num_batches)) {
    start_idx <- (i - 1) * batch_size + 1
    end_idx <- min(i * batch_size, nrow(self$df_ema))
    
    # If start_idx > nrow(self$df_ema), break the loop (no more data)
    if (start_idx > nrow(self$df_ema)) break
    
    ema_batch <- self$df_ema[start_idx:end_idx, ]
    
    # Merge EMA batch with HR data by 'customer'
    df_joined <- inner_join(ema_batch, df_hr_cleaned, by = "customer")
    
    # Filter for overlaps in time
    df_joined <- df_joined %>%
      filter(startTimestamp >= sensor_block_start & startTimestamp <= sensor_block_end)
    
    if (nrow(df_joined) > 0) {
      if (compute_stats) {
        # Group by unique_blocks and apply calculate_features
        hr_features <- df_joined %>%
          group_by(unique_blocks) %>%
          group_modify(~calculate_features(.x)) %>%
          ungroup()
      } else {
        # Compute average HR only
        hr_features <- df_joined %>%
          group_by(unique_blocks) %>%
          summarise(avg_heartrate = mean(longValue, na.rm=TRUE)) %>%
          ungroup()
      }
      
      results[[length(results) + 1]] <- hr_features
    }
  }
  
  # Combine results from all batches
  if (length(results) > 0) {
    final_features <- bind_rows(results)
    self$df_ema <- left_join(self$df_ema, final_features, by="unique_blocks")
  } else {
    # If no results, fill columns with -1
    if (compute_stats) {
      self$df_ema <- self$df_ema %>%
        mutate(
          hr_mean = -1, hr_min = -1, hr_max = -1, hr_std = -1, hr_median = -1,
          range_heartrate = -1, iqr_heartrate = -1, skewness_heartrate = -1,
          kurtosis_heartrate = -1, hr_peak_counts = -1,
          hr_zone_resting = -1, hr_zone_moderate = -1, hr_zone_vigorous = -1,
          hr_slope = -1
        )
    } else {
      self$df_ema <- self$df_ema %>%
        mutate(avg_heartrate = -1)
    }
  }
  
  return(self$df_ema)
}
