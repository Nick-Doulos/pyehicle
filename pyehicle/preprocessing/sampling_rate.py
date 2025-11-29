"""
Sampling rate analysis module for pyehicle.

This module provides utilities to analyze the temporal sampling characteristics
of GPS trajectories.
"""

import numpy as np
import pandas as pd
import polars as pl


def get_sampling_rate(
    df: pd.DataFrame | pl.DataFrame,
    time_col: str = 'time'
) -> float:
    """
    Calculate the average sampling rate (time interval) of a GPS trajectory.

    This function computes the mean time difference between consecutive trajectory points,
    providing insight into the temporal resolution of the GPS data. This information is
    useful for:
    - Understanding data quality and resolution
    - Choosing appropriate interpolation parameters
    - Validating trajectory resampling operations
    - Comparing trajectories from different sources

    The function handles various time formats and returns -1.0 for trajectories with
    insufficient data (< 2 points).

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        The DataFrame containing trajectory data with timestamps.
    time_col : str, default='time'
        The column name for time values. The column should contain datetime-like
        values that can be parsed by pandas.to_datetime().

    Returns
    -------
    float
        The average sampling rate in seconds, rounded to 3 decimal places.
        Returns -1.0 if the DataFrame has fewer than 2 rows (cannot calculate rate).

    Raises
    ------
    ValueError
        If the specified time column is not found in the DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> import pyehicle as pye
    >>>
    >>> # Create a trajectory with 5-second intervals
    >>> df = pd.DataFrame({
    ...     'time': pd.date_range('2023-01-01', periods=100, freq='5S'),
    ...     'lat': np.linspace(56.95, 56.96, 100),
    ...     'lon': np.linspace(24.10, 24.11, 100)
    ... })
    >>>
    >>> # Check sampling rate
    >>> rate = pye.preprocessing.get_sampling_rate(df)
    >>> print(f"Sampling rate: {rate} seconds")  # Output: 5.0 seconds
    >>>
    >>> # Use sampling rate to choose interpolation target
    >>> if rate > 10:
    ...     # Sparse data - may need more interpolation
    ...     target_rate = rate / 2
    ... else:
    ...     # Dense data - keep similar rate
    ...     target_rate = rate
    >>>
    >>> # Resample trajectory to target rate
    >>> resampled = pye.preprocessing.by_sampling_rate(
    ...     df,
    ...     target_sampling_rate=target_rate
    ... )

    >>> # Compare sampling rates before and after compression
    >>> original_rate = pye.preprocessing.get_sampling_rate(df)
    >>> compressed = pye.preprocessing.spatio_temporal_compress(df)
    >>> compressed_rate = pye.preprocessing.get_sampling_rate(compressed)
    >>> print(f"Original: {original_rate}s, After compression: {compressed_rate}s")

    Notes
    -----
    **Calculation Method:**
    The function computes: mean(time[i+1] - time[i]) for all consecutive pairs.
    This provides a robust estimate even for trajectories with variable sampling rates.

    **Interpretation:**
    - 1.0s: High-frequency GPS (1 Hz sampling)
    - 5.0s: Standard GPS logging (0.2 Hz)
    - 10-30s: Low-frequency tracking
    - >60s: Sparse trajectory or vehicle with intermittent tracking

    **Limitations:**
    - Assumes time column is in chronological order (not validated)
    - Doesn't detect or handle outliers in sampling rate
    - For highly variable rates, mean may not be representative
    - Returns -1.0 for empty or single-point trajectories

    **Performance:**
    - Time complexity: O(n) where n is the number of points
    - Memory: O(n) for time difference array
    - Fast even for large trajectories (> 1 million points)
    """
    # Validate that time column exists
    if time_col not in df.columns:
        raise ValueError(f"Column '{time_col}' not found in DataFrame.")

    # Early exit for insufficient data
    if len(df) < 2:
        return -1.0

    # Process pandas DataFrame
    if isinstance(df, pd.DataFrame):
        # Check dtype to optimize conversion
        if df[time_col].dtype == 'O':
            # Object dtype (strings) - parse as datetime
            time_data = pd.to_datetime(df[time_col])
        elif df[time_col].dtype != 'datetime64[ns]' and not isinstance(df[time_col].dtype, pd.DatetimeTZDtype):
            # Not datetime - convert it
            time_data = pd.to_datetime(df[time_col])
        else:
            # Already datetime - use directly
            time_data = df[time_col]

        # Calculate time differences between consecutive points
        # .diff() computes time[i] - time[i-1], so first value is NaT
        # Convert to seconds and extract non-null values (skip first NaT)
        time_diffs = time_data.diff().dt.total_seconds().to_numpy()[1:]

    # Process polars DataFrame
    elif isinstance(df, pl.DataFrame):
        # Check dtype to optimize conversion
        col_dtype = df.get_column(time_col).dtype
        if col_dtype == pl.Object or col_dtype == pl.Utf8:
            # String type - parse as datetime
            time_data = df.select(pl.col(time_col).str.strptime(pl.Datetime))[time_col]
        else:
            # Already datetime or numeric
            time_data = df[time_col]

        # Calculate time differences between consecutive points
        # Similar to pandas .diff() but polars-specific
        time_diffs_arr = time_data.diff().to_numpy()[1:]  # Skip first NaT value

        # Convert timedelta to seconds (polars-specific handling)
        time_diffs = time_diffs_arr.astype('timedelta64[s]').astype(float)

    else:
        raise ValueError("df must be either a pandas DataFrame or a polars DataFrame.")

    # Calculate mean sampling rate and round to 3 decimal places
    average_sampling_rate = float(np.mean(time_diffs))
    return round(average_sampling_rate, 3)
