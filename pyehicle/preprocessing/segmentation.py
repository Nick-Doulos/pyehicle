"""
Trajectory segmentation module for pyehicle.

This module provides functions to split GPS trajectories into segments based on
temporal gaps. Useful for separating distinct trips or identifying breaks in
continuous tracking.
"""

import numpy as np
import pandas as pd
import polars as pl


def by_time(
    df: pd.DataFrame | pl.DataFrame,
    time_threshold: float = 30,
    length_threshold: int = 20,
    time_col: str = 'time'
) -> pd.DataFrame | pl.DataFrame | list:
    """
    Segment a trajectory into sub-trajectories based on time gaps between consecutive points.

    This function splits a trajectory whenever there is a time gap larger than the specified
    threshold. Only segments longer than the length threshold are kept. This is useful for:
    - Separating distinct trips recorded in a single file
    - Identifying breaks in continuous GPS tracking
    - Removing short, fragmented trajectory segments

    The function preserves all original columns and maintains the temporal order within segments.

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        The trajectory DataFrame to segment. Must contain a time column.
    time_threshold : float, default=30
        Maximum time gap in seconds. Gaps larger than this will cause a split.
        Common values:
        - 30s: For removing brief GPS signal losses
        - 300s (5 min): For separating distinct trips
        - 3600s (1 hour): For separating different days
    length_threshold : int, default=20
        Minimum number of points for a segment to be kept. Segments with fewer
        points are discarded. Helps filter out very short, potentially noisy segments.
    time_col : str, default='time'
        Name of the time column. Must be parseable by pandas.to_datetime().

    Returns
    -------
    pd.DataFrame, pl.DataFrame, or list
        The return type depends on the number of valid segments found:
        - If only 1 valid segment: Returns a single DataFrame (same type as input)
        - If multiple valid segments: Returns a list of DataFrames
        - If no valid segments (all too short): Returns the original DataFrame unchanged
        - If input has ≤ 1 row: Returns the original DataFrame unchanged

    Examples
    --------
    >>> import pandas as pd
    >>> import pyehicle as pye
    >>>
    >>> # Load a trajectory with multiple trips
    >>> df = pd.read_csv('full_day_gps.csv')
    >>> print(f"Total points: {len(df)}")
    >>>
    >>> # Split at 10-minute gaps, keep segments with 50+ points
    >>> segments = pye.preprocessing.by_time(
    ...     df,
    ...     time_threshold=600,      # 10 minutes
    ...     length_threshold=50
    ... )
    >>>
    >>> # Check results
    >>> if isinstance(segments, list):
    ...     print(f"Found {len(segments)} trips")
    ...     for i, seg in enumerate(segments):
    ...         print(f"Trip {i+1}: {len(seg)} points")
    ... else:
    ...     print(f"Single continuous trajectory: {len(segments)} points")
    >>>
    >>> # Process each segment separately
    >>> if isinstance(segments, list):
    ...     for i, segment in enumerate(segments):
    ...         # Apply preprocessing to each trip
    ...         compressed = pye.preprocessing.spatio_temporal_compress(segment)
    ...         matched = pye.preprocessing.leuven(compressed)
    ...         matched.to_csv(f'trip_{i+1}_matched.csv')

    Notes
    -----
    **Algorithm:**
    1. Calculate time differences between consecutive points
    2. Identify split points where time_diff > time_threshold
    3. Split trajectory at these points
    4. Filter out segments with length ≤ length_threshold
    5. Return results based on number of valid segments

    **Performance:**
    - Time complexity: O(n) where n is the number of points
    - Memory: O(n) for creating new DataFrames
    - Fast for all trajectory sizes

    **Edge Cases:**
    - Empty or single-point input: Returns original DataFrame
    - All segments too short: Returns original DataFrame
    - Time column not sorted: Results may be unexpected (assumes chronological order)

    **Use Cases:**
    - Preprocessing multi-day GPS logs
    - Separating work commutes from personal trips
    - Removing GPS signal loss periods
    - Batch processing distinct trajectory segments
    """
    # Handle edge cases: empty or very short trajectories
    if len(df) <= 1:
        return df

    if len(df) <= length_threshold:
        return df

    # Create pandas Timedelta object for threshold comparison
    td = pd.Timedelta(time_threshold, unit='s')

    # Process pandas DataFrame
    if isinstance(df, pd.DataFrame):
        # Extract timestamps for gap analysis
        time_values = df[time_col].to_numpy()

        # Calculate time differences between consecutive points
        # Note: np.diff returns array of length n-1
        time_diffs = np.diff(time_values)

        # Find all points where the time gap exceeds the threshold
        # Add 1 to indices because diff() reduces array length by 1
        split_indices = np.where(time_diffs > td)[0] + 1

        trajectories_list = []

        # Build segments by splitting at the identified gap points
        start_idx = 0
        for split_idx in split_indices:
            segment_length = split_idx - start_idx

            # Only keep segments that meet the length threshold
            if segment_length > length_threshold:
                temp_df = df.iloc[start_idx:split_idx].copy().reset_index(drop=True)
                trajectories_list.append(temp_df)

            start_idx = split_idx

        # Handle the final segment (from last split to end)
        final_segment_length = len(df) - start_idx
        if final_segment_length > length_threshold:
            temp_df = df.iloc[start_idx:].copy().reset_index(drop=True)
            trajectories_list.append(temp_df)

    else:  # polars DataFrame
        # Extract timestamps for gap analysis
        time_values = df[time_col].to_numpy()

        # Calculate time differences between consecutive points
        time_diffs = np.diff(time_values)

        # Find all points where the time gap exceeds the threshold
        split_indices = np.where(time_diffs > td)[0] + 1

        trajectories_list = []

        # Build segments by splitting at the identified gap points
        start_idx = 0
        for split_idx in split_indices:
            segment_length = int(split_idx) - start_idx

            # Only keep segments that meet the length threshold
            if segment_length > length_threshold:
                temp_df = df[start_idx:int(split_idx)].clone()
                trajectories_list.append(temp_df)

            start_idx = int(split_idx)

        # Handle the final segment (from last split to end)
        final_segment_length = len(df) - start_idx
        if final_segment_length > length_threshold:
            temp_df = df[start_idx:].clone()
            trajectories_list.append(temp_df)

    # Return results based on number of valid segments found
    if not trajectories_list:
        # No valid segments (all too short) - return original
        return df
    elif len(trajectories_list) == 1:
        # Single valid segment - return as DataFrame
        return trajectories_list[0]
    else:
        # Multiple valid segments - return as list
        return trajectories_list
