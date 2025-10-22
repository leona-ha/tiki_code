# %%
import logging

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


def midnight_timestamp_to_berlin_tzoffset(ts):
    # it's dst aware
    if isinstance(ts, pd.Series):
        return (ts + pd.Timedelta(hours=12)).dt.tz_convert("Europe/Berlin").map(
            lambda x: x.utcoffset()
        ).dt.total_seconds() / 60
    elif isinstance(ts, pd.Timestamp):
        return (ts + pd.Timedelta(hours=12)).tz_convert(
            "Europe/Berlin"
        ).utcoffset().total_seconds() / 60


# %%
def create_utcday_tzoffset_df(df: pd.DataFrame) -> pd.DataFrame:
    """Create a DataFrame with inferred timezone offsets for each customer and day.
    It assumes the specific structure of the input DataFrame."""
    # TODO might utilize something like ffill if the current implementation is too slow
    # %%
    # df = df.drop(columns=["stringValue", "doubleValue", "longValue", "booleanValue"]).copy()
    df = df[
        ["customer", "startTimestamp", "timezoneOffset", "type", "createdAt"]
    ].copy()

    df["startTimestamp_day"] = df["startTimestamp"].dt.floor("D")
    df["createdAt_day"] = df["createdAt"].dt.floor("D")

    df_tz = (
        df[["customer", "startTimestamp_day"]].drop_duplicates().reset_index(drop=True)
    )
    df_tz = df_tz.sort_values(["customer", "startTimestamp_day"]).reset_index(drop=True)

    # %%
    # Berlin DST changes
    # https://en.wikipedia.org/wiki/Daylight_saving_time_in_Europe
    # Create date range from min to max year in df, string make it interpreted as year
    date_range = pd.date_range(
        f"{df['startTimestamp_day'].min().year}",
        f"{df['startTimestamp_day'].max().year + 1}",
        freq="h",
        tz="Europe/Berlin",
    )

    berlin_offsets = pd.Series(
        date_range.map(lambda x: x.utcoffset()), index=date_range
    )

    # Find where timezone offsets change
    offset_changes = berlin_offsets != berlin_offsets.shift(1)
    dst_changes = date_range[offset_changes]
    dst_changes = dst_changes[1:]
    # remove the first value, which is always different than shift (nan)

    # %%
    df_tz["dst_berlin_change_day"] = df_tz.startTimestamp_day.dt.date.isin(
        dst_changes.date
    )

    # %% [markdown]
    # ### df tz gps

    # %%
    df_tz_gps = (
        df[df["type"].isin(["Latitude", "Longitude"])]
        .groupby(["customer", "startTimestamp_day"], observed=True)["timezoneOffset"]
        .apply(lambda x: x.unique())
        .reset_index(name="tzs_gps")
    )

    # %%
    df_tz = df_tz.merge(df_tz_gps, on=["customer", "startTimestamp_day"], how="left")
    df_tz.head(10)

    # %%
    df_tz["gps_exists"] = ~df_tz["tzs_gps"].isna()

    # %% [markdown]
    # ### df tz activitydetail createdAt

    # %%
    df_tz_activitydetailcreatedat = (
        df[df["type"].isin(["ActivityTypeDetail1", "ActivityTypeDetail2"])]
        .groupby(
            ["customer", "createdAt_day"],
            observed=True,
        )["timezoneOffset"]
        .apply(lambda x: x.unique())
        .reset_index(name="tzs_activitydetailcreatedat")
    )

    # %%
    df_tz = df_tz.merge(
        df_tz_activitydetailcreatedat.rename(
            columns={"createdAt_day": "startTimestamp_day"}
        ),
        on=["customer", "startTimestamp_day"],
        how="left",
    )

    # %% [markdown]
    # # return_tz

    # %%
    df_tz["return_tz"] = pd.NA  # we want to return this
    df_tz["return_source"] = pd.NA  # where did the tz come from

    # %%
    mask_gps = df_tz.gps_exists & (df_tz.tzs_gps.dropna().apply(len) == 1)
    df_tz.loc[mask_gps, "return_tz"] = df_tz.tzs_gps[mask_gps].apply(lambda x: x[0])
    df_tz.loc[mask_gps, "return_source"] = "gps_single"

    # %%
    mask_activitydetailcreatedat = (~mask_gps) & (
        df_tz.tzs_activitydetailcreatedat.dropna().apply(len) == 1
    )
    df_tz.loc[mask_activitydetailcreatedat, "return_tz"] = (
        df_tz.tzs_activitydetailcreatedat[mask_activitydetailcreatedat].apply(
            lambda x: x[0]
        )
    )
    df_tz.loc[mask_activitydetailcreatedat, "return_source"] = (
        "activitydetailcreatedat_single"
    )

    # %%

    def adjust_dst_changes(df_tz, update_source=False, inplace=False):
        """Adjust for DST changes in Berlin timezone."""
        # for the dst, assume there is the new timezone, in Berlin it's gonna be just one hour different
        # (1->2, and 2->1; instead of 2->3 & 3->2, as it shoud be) (since we convert at the 00:00 UTC )
        # If the primary metric would be the time from previous midnight (as in the sleep),
        # then it should be the other way round
        if not inplace:
            df_tz = df_tz.copy()

        change2summer = df_tz.dst_berlin_change_day & (
            df_tz.startTimestamp_day.dt.month == 3
        )
        change2winter = df_tz.dst_berlin_change_day & (
            df_tz.startTimestamp_day.dt.month == 10
        )
        dst_summer_mask = change2summer & df_tz.return_tz.isin([60, 120])
        dst_winter_mask = change2winter & df_tz.return_tz.isin([60, 120])
        df_tz.loc[dst_summer_mask, "return_tz"] = 120
        df_tz.loc[dst_winter_mask, "return_tz"] = 60

        if update_source:
            df_tz.loc[dst_summer_mask, "return_source"] += "_dst_adjusted"
            df_tz.loc[dst_winter_mask, "return_source"] += "_dst_adjusted"

        return df_tz

    # it's done here, to make gps and activitydetailcreatedat multiple better
    # don't update the source here, as it's gonna be adjusted later
    df_tz = adjust_dst_changes(df_tz, inplace=True)

    # %%
    def infer_timezone_from_neighbors(
        row, df_tz, potential_tzs=None, prefer_previous_than_distance=False
    ):
        """Infer timezone based on previous and next available data points."""
        # ! it assumes certain structure of row and df_tz, and that row is from df_tz

        # TODO add assertions

        # Get previous and next available data points
        prev_df = df_tz[
            (df_tz.customer == row.customer)
            & (df_tz.startTimestamp_day < row.startTimestamp_day)
            & (df_tz.return_tz.notna())
        ].sort_values("startTimestamp_day")
        prev_available = prev_df.iloc[-1] if not prev_df.empty else None

        next_df = df_tz[
            (df_tz.customer == row.customer)
            & (df_tz.startTimestamp_day > row.startTimestamp_day)
            & (df_tz.return_tz.notna())
        ].sort_values("startTimestamp_day")
        next_available = next_df.iloc[0] if not next_df.empty else None

        # Check which neighbors are valid (exist and match potential timezones)
        if potential_tzs is None:
            prev_valid = prev_available is not None
            next_valid = next_available is not None
        else:
            prev_valid = (
                prev_available is not None and prev_available.return_tz in potential_tzs
            )
            next_valid = (
                next_available is not None and next_available.return_tz in potential_tzs
            )

        # Handle cases based on validity
        if not prev_valid and not next_valid:
            if prev_available is None and next_available is None:
                # logging.warning(
                #     f"No neighbors for customer {row.customer} at {row.startTimestamp_day} with tzs {potential_tzs}"
                # )
                return None, "no_previous_no_next"
            else:
                return (
                    None,
                    "no_previous_conflict_with_next"
                    if prev_available is None
                    else (
                        "conflict_with_previous_no_next"
                        if next_available is None
                        else "conflict_with_both"
                    ),
                )

        # Choose the valid neighbor (or closer one if both valid)
        if prev_valid and not next_valid:
            return (
                prev_available.return_tz,
                "previous_is_fine_no_next"
                if next_available is None
                else "conflict_with_next_previous_is_fine",
            )

        if next_valid and not prev_valid:
            return (
                next_available.return_tz,
                "no_previous_next_is_fine"
                if prev_available is None
                else "conflict_with_previous_next_is_fine",
            )

        if prev_available.return_tz == next_available.return_tz:
            return prev_available.return_tz, "both_neighbors_agree"

        # Both valid - choose closer one
        days_to_prev = (row.startTimestamp_day - prev_available.startTimestamp_day).days
        days_to_next = (next_available.startTimestamp_day - row.startTimestamp_day).days

        if (
            days_to_prev <= days_to_next or prefer_previous_than_distance
        ):  # Prefer previous on tie
            return (
                prev_available.return_tz,
                f"inferred_from_previous_{days_to_prev}d"
                + ("_equal_dist" if days_to_prev == days_to_next else ""),
            )
        else:
            return next_available.return_tz, f"inferred_from_next_{days_to_next}d"

    # %%
    df_tz_before = df_tz.copy()
    for ind, row in df_tz[
        df_tz.tzs_gps.dropna().apply(lambda x: len(x) > 1) & df_tz.return_tz.isna()
    ].iterrows():
        inferred_tz, source = infer_timezone_from_neighbors(
            row, df_tz_before, row.tzs_gps
        )
        source = "gps_multiple_" + source
        df_tz.at[ind, "return_source"] = source
        if inferred_tz is not None:
            df_tz.at[ind, "return_tz"] = inferred_tz
        else:
            logging.info(
                f"[GPS] Could not infer tz for customer {row.customer} at {row.startTimestamp_day} with potential tzs {row.tzs_gps} - {source}"
            )

    df_tz = adjust_dst_changes(df_tz)

    # %%
    df_tz_before = df_tz.copy()
    for ind, row in df_tz[
        df_tz.tzs_activitydetailcreatedat.dropna().apply(lambda x: len(x) > 1)
        & df_tz.return_tz.isna()
    ].iterrows():
        potential_tzs = row.tzs_activitydetailcreatedat
        inferred_tz, source = infer_timezone_from_neighbors(
            row, df_tz_before, potential_tzs
        )
        source = "activitydetail_multiple_" + source
        df_tz.at[ind, "return_source"] = source
        if inferred_tz is not None:
            df_tz.at[ind, "return_tz"] = inferred_tz
        else:
            logging.info(
                f"[ActivityDetailCreatedAt] Could not infer tz for customer {row.customer} at {row.startTimestamp_day} with potential tzs {potential_tzs} - {source}"
            )

    df_tz = adjust_dst_changes(df_tz)

    # %%
    # last interpolation, without any potential tzs
    # prefer previous than distance to play better with dst changes
    df_tz_before = df_tz.copy()
    for ind, row in df_tz[df_tz.return_tz.isna()].iterrows():
        inferred_tz, source = infer_timezone_from_neighbors(
            row, df_tz_before, potential_tzs=None, prefer_previous_than_distance=True
        )
        source = "interpolate_" + source
        df_tz.at[ind, "return_source"] = source
        if inferred_tz is not None:
            df_tz.at[ind, "return_tz"] = inferred_tz
        else:
            logging.info(
                f"[Interpolate] Could not infer tz for customer {row.customer} at {row.startTimestamp_day} - {source}"
            )

    # %%
    # these customers still need adjustment
    # df[df.customer == '3oNs'] # only one day, no gps
    # df[df.customer == '9nSQ'] # only for one day, just one ECG recording
    # df[df.customer == 'INjr'] # only two days, two ECG sessions and that's it
    # df[df.customer == 'LEgz'] # only 10 days, mostly running sessions
    # df[df.customer == 'SssR'] # data for one month 2023-05 -- 2023-06, no gps activitytypedetails without createdAt
    # df[df.customer == 'Zv6E'] # data for two months 2023-09 -- 2023-11, no gps, activitytypedetails without createdAt

    # %%
    # fill the other with berlin timezone
    still_missing_mask = df_tz.return_tz.isna()
    if still_missing_mask.any():
        # add 12 hours to make the dst change day count towards the next offset
        # this make only one hour difference compared to true dst change (1->2 vs 2->3)
        # ! there might be some problems if the local time is read with tz
        df_tz.loc[still_missing_mask, "return_tz"] = (
            df_tz.loc[still_missing_mask, "startTimestamp_day"] + pd.Timedelta(hours=12)
        ).dt.tz_convert("Europe/Berlin").map(
            lambda x: x.utcoffset()
        ).dt.total_seconds() / 60
        df_tz.loc[still_missing_mask, "return_source"] = "assumed_berlin"

    # %%
    df_tz = adjust_dst_changes(df_tz, update_source=True)

    # %%
    df_tz["return_source"] = df_tz["return_source"].astype("category")

    # return df_tz

    df_tz = df_tz.rename(
        columns={
            "startTimestamp_day": "day",
            "return_tz": "inferred_tzoffset",
            "return_source": "inferred_tzoffset_source",
        }
    )
    df_tz = df_tz[["customer", "day", "inferred_tzoffset", "inferred_tzoffset_source"]]
    return df_tz


def merge_fill_tz(df, df_tz, day_col="quest_create_day", customer_col="customer"):
    """
    Fill missing timezone offsets by merging with timezone reference data,
    forward-filling within customer groups, correcting DST mismatches, and
    assuming Berlin timezone for remaining missing values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with customer and day columns that need timezone offset filling
    df_tz : pd.DataFrame
        Reference DataFrame with inferred timezone offsets per customer and day
        Must contain columns: ['customer', 'day', 'inferred_tzoffset', 'inferred_tzoffset_source']
    day_col : str, default="quest_create_day"
        Name of the day column in df to merge on
    customer_col : str, default="customer"
        Name of the customer column in df to merge on

    Returns
    -------
    pd.DataFrame
        DataFrame with filled timezone offsets and sources
    """
    assert (
        set(["inferred_tzoffset", "inferred_tzoffset_source"]) & set(df.columns)
        == set()
    ), (
        "Input df already contains 'inferred_tzoffset' or 'inferred_tzoffset_source' columns! "
        "Drop them before calling this function"
    )
    # Select only needed columns from df_tz and merge
    df_merged = pd.merge(
        df,
        df_tz,
        left_on=[customer_col, day_col],
        right_on=["customer", "day"],
        how="left",
        suffixes=("", ""),
    )

    df_merged = df_merged.drop(columns=["day"])

    # Track which rows were initially missing
    missing_tz = df_merged["inferred_tzoffset"].isna()
    if missing_tz.sum() == 0:
        logging.info("No missing timezone offsets to fill ✅.")
        df_merged["inferred_tzoffset"] = df_merged["inferred_tzoffset"].astype("int")
        df_merged["inferred_tzoffset_source"] = df_merged[
            "inferred_tzoffset_source"
        ].astype("category")
        return df_merged

    # Forward fill within customer groups
    df_merged["inferred_tzoffset"] = df_merged.groupby(customer_col)[
        "inferred_tzoffset"
    ].ffill()

    # Ensure source column is string type
    df_merged["inferred_tzoffset_source"] = df_merged[
        "inferred_tzoffset_source"
    ].astype("string")

    # Mark forward-filled entries
    ffilled = missing_tz & df_merged["inferred_tzoffset"].notna()
    df_merged.loc[ffilled, "inferred_tzoffset_source"] = "ffilled"

    logging.info(f"Forward-filled timezone offsets: {ffilled.sum()}")

    # Correct DST mismatches for forward-filled entries
    if ffilled.sum() > 0:
        # Determine expected timezone offset based on Berlin DST rules
        expected_berlin_offset = midnight_timestamp_to_berlin_tzoffset(
            df_merged.loc[ffilled, day_col]
        )

        # Identify mismatches
        is_summer_expected = expected_berlin_offset == 120
        is_winter_expected = expected_berlin_offset == 60

        needs_summer_correction = is_summer_expected & (
            df_merged.loc[ffilled, "inferred_tzoffset"] == 60
        )
        needs_winter_correction = is_winter_expected & (
            df_merged.loc[ffilled, "inferred_tzoffset"] == 120
        )

        # Apply corrections
        ffilled_idx = df_merged.index[ffilled]
        df_merged.loc[ffilled_idx[needs_summer_correction], "inferred_tzoffset"] = 120
        df_merged.loc[ffilled_idx[needs_winter_correction], "inferred_tzoffset"] = 60

        logging.info(f"Corrected to summer time (120 min): {needs_summer_correction.sum()}")
        logging.info(f"Corrected to winter time (60 min): {needs_winter_correction.sum()}")

    # Handle still-missing values by assuming Berlin timezone
    still_missing = df_merged["inferred_tzoffset"].isna()

    if still_missing.sum() > 0:
        logging.info(f"Assumed Berlin timezone offsets: {still_missing.sum()}")
        df_merged.loc[still_missing, "inferred_tzoffset"] = (
            midnight_timestamp_to_berlin_tzoffset(df_merged.loc[still_missing, day_col])
        )
        df_merged.loc[still_missing, "inferred_tzoffset_source"] = "assumed_berlin"

    df_merged["inferred_tzoffset"] = df_merged["inferred_tzoffset"].astype("int")
    df_merged["inferred_tzoffset_source"] = df_merged[
        "inferred_tzoffset_source"
    ].astype("category")
    # Final assertion
    assert df_merged["inferred_tzoffset"].isna().sum() == 0, (
        "Failed to fill all timezone offsets!"
    )

    return df_merged
