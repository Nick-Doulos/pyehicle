"""
Trajectory interpolation module for pyehicle.

This module provides functions to resample GPS trajectories by interpolating points along
geodesic paths (great-circle routes). Interpolation is useful for:
- Increasing trajectory density for smoother visualization
- Normalizing sampling rates across different data sources
- Preparing trajectories for analysis requiring uniform spacing
- Filling gaps in sparse GPS data

The module implements geodesic interpolation using pyproj's WGS84 ellipsoid calculations,
ensuring accurate distances and bearings across all latitudes. Two interpolation modes:
1. **by_number_of_points()**: Resample to exact point count (spatial uniformity)
2. **by_sampling_rate()**: Resample to temporal interval (temporal uniformity)
"""

import numpy as np
import pandas as pd
import polars as pl
from scipy.interpolate import interp1d
from pyproj import Geod
from pyehicle.preprocessing.sampling_rate import get_sampling_rate
from typing import Union, Tuple

# single Geod instance reused for speed
_GEOD = Geod(ellps="WGS84")


def _interpolate_geo_coord_pyproj(lat1: float, lon1: float,
                                  lat2: float, lon2: float,
                                  fraction: float) -> Tuple[float, float]:
    """
    Interpolate geodesically between (lat1, lon1) and (lat2, lon2) by fraction [0..1].
    Returns (lat_interp, lon_interp).
    """
    # pyproj.Geod.inv expects lon, lat order
    az12, az21, s12 = _GEOD.inv(lon1, lat1, lon2, lat2)
    s = float(s12) * float(fraction)
    lon_i, lat_i, back_az = _GEOD.fwd(lon1, lat1, az12, s)
    return float(lat_i), float(lon_i)


def _vectorized_interpolate_segment(lat_a: np.ndarray, lon_a: np.ndarray,
                                    lat_b: np.ndarray, lon_b: np.ndarray,
                                    fractions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized interpolation for arrays of segment endpoints and fractions.
    - lat_a, lon_a, lat_b, lon_b, fractions are 1D arrays of same length m.
    Returns arrays (lat_interp, lon_interp) of shape (m,).
    """
    # Compute forward azimuth and geodesic distance s12 using vectorized inv
    # pyproj.Geod.inv supports array inputs in lon, lat order
    # It returns arrays: az12, az21, s12
    az12, az21, s12 = _GEOD.inv(lon_a, lat_a, lon_b, lat_b)
    # distance to interpolate
    s_interp = s12 * fractions
    # Use Geod.fwd vectorized: returns lon2, lat2, back_az
    lon_i, lat_i, _ = _GEOD.fwd(lon_a, lat_a, az12, s_interp)
    return np.asarray(lat_i, dtype=float), np.asarray(lon_i, dtype=float)


def by_number_of_points(df: Union[pd.DataFrame, pl.DataFrame],
                        num: int,
                        lat_col: str = "lat",
                        lon_col: str = "lon",
                        time_col: str = "time") -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Resample trajectory to exact number of points using geodesic interpolation.

    This function resamples a GPS trajectory to contain exactly `num` points by interpolating
    along geodesic (great-circle) paths between original points. Points are distributed
    uniformly by distance along the trajectory, and timestamps are interpolated proportionally.

    This is useful for:
    - Normalizing trajectory density across datasets
    - Preparing trajectories for fixed-size neural network inputs
    - Creating smoother visualizations
    - Standardizing trajectories for comparison

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Input trajectory with at least latitude, longitude, and optionally time columns.
        Minimum 2 points required.
    num : int
        Target number of points in the output trajectory. Must be > 2.
        Start and end points are always preserved.
    lat_col : str, default='lat'
        Name of the latitude column (WGS84 decimal degrees).
    lon_col : str, default='lon'
        Name of the longitude column (WGS84 decimal degrees).
    time_col : str, default='time'
        Name of the time column. If present, timestamps are interpolated linearly
        by cumulative distance. If not present, output contains only lat/lon.

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Resampled trajectory with exactly `num` points. Returns same type as input.
        Contains columns: lat_col, lon_col, and time_col (if present in input).
        Points are distributed uniformly by distance along the geodesic path.

    Examples
    --------
    >>> import pandas as pd
    >>> import pyehicle as pye
    >>>
    >>> # Load trajectory with variable spacing
    >>> df = pd.read_csv('trajectory.csv')
    >>> print(f"Original: {len(df)} points")
    >>>
    >>> # Resample to exactly 100 points
    >>> resampled = pye.preprocessing.by_number_of_points(df, num=100)
    >>> print(f"Resampled: {len(resampled)} points")  # Exactly 100
    >>>
    >>> # Increase density for smoother visualization
    >>> dense = pye.preprocessing.by_number_of_points(df, num=500)
    >>> pye.utilities.visualization.single(dense, name='Dense Trajectory')

    Notes
    -----
    **Algorithm:**
    1. Calculate cumulative geodesic distances along the original trajectory
    2. Define `num` target distances uniformly spaced from 0 to total_distance
    3. For each target distance, find containing segment and interpolation fraction
    4. Interpolate coordinates geodesically using pyproj Geod.fwd()
    5. Interpolate timestamps linearly by distance (if time column present)

    **Geodesic Interpolation:**
    - Uses WGS84 ellipsoid (pyproj Geod)
    - Accurate for all distances and latitudes
    - Interpolates along great-circle paths (shortest distance on sphere)
    - Preserves exact start and end points

    **Performance:**
    - Time complexity: O(n + m) where n = input points, m = output points
    - Vectorized operations for efficiency
    - Fast even for large trajectories (>10,000 points)

    **Use Cases:**
    - Standardizing trajectory density for machine learning
    - Creating smooth animations with fixed frame counts
    - Upsampling sparse trajectories (num > len(df))
    - Downsampling dense trajectories (num < len(df))

    See Also
    --------
    by_sampling_rate : Resample by temporal interval
    get_sampling_rate : Calculate current sampling rate
    """
    # Validation
    if num <= 2:
        raise ValueError('"num" must be > 2 (start and end included).')
    if isinstance(df, pl.DataFrame):
        pdf = df.to_pandas()
    else:
        pdf = df.copy()

    if len(pdf) < 2:
        raise ValueError("Input trajectory must contain at least two points")

    lats = pdf[lat_col].to_numpy(dtype=float)
    lons = pdf[lon_col].to_numpy(dtype=float)

    # compute segment distances (vectorized)
    # Geod.inv wants lon, lat ordering
    az12, az21, s12 = _GEOD.inv(lons[:-1], lats[:-1], lons[1:], lats[1:])
    segment_distances = np.asarray(s12, dtype=float)  # meters
    total_distance = float(segment_distances.sum())

    # cumulative distances at segment boundaries: xi[0]=0, xi[1]=d0, xi[2]=d0+d1, ..., xi[len]=total
    xi = np.concatenate(([0.0], np.cumsum(segment_distances)))

    # target distances along the path for the new points
    target_d = np.linspace(0.0, total_distance, num=num, endpoint=True)

    # for each target_d determine which segment it falls into:
    # segment index idx such that xi[idx] <= target_d < xi[idx+1], except last point
    # numpy.searchsorted gives insertion index; subtract 1 to get left index
    idx = np.searchsorted(xi, target_d, side="right") - 1
    # fix boundary cases: for exact total_distance, idx may equal len(segment_distances)
    idx[idx == len(segment_distances)] = len(segment_distances) - 1

    # fraction along that segment
    seg_left = xi[idx]
    seg_len = segment_distances[idx]
    # guard against zero-length segments
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = (target_d - seg_left) / seg_len
    # for points exactly at a node where seg_len == 0, set fraction 0
    fraction = np.nan_to_num(fraction, nan=0.0, posinf=0.0, neginf=0.0)

    # prepare arrays of segment endpoints for vectorized interpolation
    lat_a = lats[idx]
    lon_a = lons[idx]
    lat_b = lats[idx + 1]
    lon_b = lons[idx + 1]

    # vectorized geodesic interpolation per-target
    lat_interp, lon_interp = _vectorized_interpolate_segment(lat_a, lon_a, lat_b, lon_b, fraction)

    # If time column present, interpolate by cumulative distance using linear interpolation
    if time_col in pdf.columns:
        times = pd.to_datetime(pdf[time_col])
        # seconds from start
        t_seconds = (times - times.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
        # xi aligns with nodes; do interp on xi -> t_seconds
        time_interp_seconds = interp1d(xi, t_seconds, kind="linear", fill_value="extrapolate")(target_d)
        times_out = times.iloc[0] + pd.to_timedelta(time_interp_seconds, unit="s")
        out_pdf = pd.DataFrame({time_col: times_out, lat_col: lat_interp, lon_col: lon_interp})
        # ensure first time equals original start exactly (avoid float rounding)
        out_pdf[time_col].iat[0] = times.iloc[0]
    else:
        out_pdf = pd.DataFrame({lat_col: lat_interp, lon_col: lon_interp})

    if isinstance(df, pl.DataFrame):
        return pl.from_pandas(out_pdf)
    return out_pdf


def by_sampling_rate(df: Union[pd.DataFrame, pl.DataFrame],
                     target_sampling_rate: float,
                     lat_col: str = "lat",
                     lon_col: str = "lon",
                     time_col: str = "time") -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Resample trajectory to uniform temporal interval using geodesic interpolation.

    This function resamples a GPS trajectory to a constant time interval (sampling rate)
    by interpolating coordinates at regular temporal intervals. This creates trajectories
    with uniform time spacing, essential for time-series analysis and temporal modeling.

    Coordinates are interpolated geodesically along great-circle paths between original
    points, maintaining geographic accuracy across all latitudes and distances.

    This is useful for:
    - Normalizing temporal resolution across datasets
    - Preparing trajectories for time-series analysis
    - Synchronizing trajectories from different devices
    - Creating animations with consistent frame timing

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Input trajectory with latitude, longitude, and time columns. Minimum 2 points.
    target_sampling_rate : float
        Target time interval in seconds between consecutive points. Must be positive
        and smaller than the current sampling rate. Typical values:
        - 1.0s: High-frequency (1 Hz)
        - 5.0s: Standard GPS logging
        - 10.0s: Low-frequency tracking
    lat_col : str, default='lat'
        Name of the latitude column (WGS84 decimal degrees).
    lon_col : str, default='lon'
        Name of the longitude column (WGS84 decimal degrees).
    time_col : str, default='time'
        Name of the time column. Must be parseable by pandas.to_datetime().

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Resampled trajectory with uniform temporal spacing. Returns same type as input.
        Contains columns: lat_col, lon_col, time_col.
        Points are spaced at exact `target_sampling_rate` intervals.
        Duplicates removed if present.

    Raises
    ------
    ValueError
        - If target_sampling_rate <= 0
        - If target_sampling_rate > current sampling rate (would require downsampling)

    Examples
    --------
    >>> import pandas as pd
    >>> import pyehicle as pye
    >>>
    >>> # Load trajectory with variable sampling rate
    >>> df = pd.read_csv('trajectory.csv')
    >>> current_rate = pye.preprocessing.get_sampling_rate(df)
    >>> print(f"Current sampling rate: {current_rate}s")
    >>>
    >>> # Resample to 5-second intervals
    >>> resampled = pye.preprocessing.by_sampling_rate(df, target_sampling_rate=5.0)
    >>> new_rate = pye.preprocessing.get_sampling_rate(resampled)
    >>> print(f"New sampling rate: {new_rate}s")  # Should be ~5.0
    >>>
    >>> # Synchronize multiple trajectories to same temporal resolution
    >>> traj1 = pye.preprocessing.by_sampling_rate(df1, target_sampling_rate=1.0)
    >>> traj2 = pye.preprocessing.by_sampling_rate(df2, target_sampling_rate=1.0)
    >>> # Now both have 1-second intervals for comparison

    Notes
    -----
    **Algorithm:**
    1. Convert timestamps to seconds from start
    2. Generate target times at uniform intervals: 0, Δt, 2Δt, 3Δt, ...
    3. For each target time, find containing time segment
    4. Calculate interpolation fraction within segment
    5. Interpolate coordinates geodesically at that time
    6. Convert back to datetime format

    **Geodesic Interpolation:**
    - Uses WGS84 ellipsoid for accurate great-circle paths
    - Coordinates interpolated along shortest distance on sphere
    - Maintains geographic accuracy at all latitudes
    - Vectorized operations for performance

    **Temporal Requirements:**
    - `target_sampling_rate` must be smaller than current rate
    - This function increases temporal resolution (more frequent sampling)
    - For decreasing resolution (downsampling), use data selection instead

    **Performance:**
    - Time complexity: O(n + m) where n = input points, m = output points
    - Vectorized numpy operations throughout
    - Fast for typical trajectories (<10,000 points)

    **Use Cases:**
    - Preparing data for LSTM/RNN models (uniform time steps)
    - Creating smooth real-time visualizations
    - Synchronizing multi-sensor data
    - Generating training data with consistent temporal resolution

    See Also
    --------
    by_number_of_points : Resample by fixed point count
    get_sampling_rate : Calculate current sampling rate
    """
    if isinstance(df, pl.DataFrame):
        pdf = df.to_pandas()
    else:
        pdf = df.copy()

    if target_sampling_rate <= 0:
        raise ValueError("target_sampling_rate must be positive")

    if len(pdf) < 2:
        return df.copy()

    # check sampling_rate validity using pyehicle helper (keeps your original behavior)
    sr = get_sampling_rate(df, time_col)
    if target_sampling_rate > sr:
        raise ValueError('"target_sampling_rate" must be less than the sampling rate of the dataframe')

    pdf[time_col] = pd.to_datetime(pdf[time_col])
    times = pdf[time_col].to_numpy(dtype="datetime64[ns]")
    t0 = times[0]
    # seconds from start
    time_seconds = (times - t0).astype("timedelta64[ns]").astype("int64") / 1e9
    time_seconds = time_seconds.astype(float)

    total_time = float(time_seconds[-1] - time_seconds[0])
    if total_time <= 0:
        # degenerate time series
        if isinstance(df, pl.DataFrame):
            return df.clone()
        return df.copy()

    # target times in seconds from start
    target_times = np.arange(0.0, total_time + 1e-9, target_sampling_rate, dtype=float)

    # precompute coordinates arrays
    lats = pdf[lat_col].to_numpy(dtype=float)
    lons = pdf[lon_col].to_numpy(dtype=float)

    # For each target_time we need to find containing segment index and fraction
    # Using vectorized searchsorted:
    idx = np.searchsorted(time_seconds, target_times, side="right") - 1
    # clamp
    idx[idx < 0] = 0
    idx[idx >= (len(time_seconds) - 1)] = len(time_seconds) - 2

    t_left = time_seconds[idx]
    t_right = time_seconds[idx + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = (target_times - t_left) / (t_right - t_left)
    fraction = np.nan_to_num(fraction, nan=0.0, posinf=0.0, neginf=0.0)

    # segment endpoints arrays
    lat_a = lats[idx]
    lon_a = lons[idx]
    lat_b = lats[idx + 1]
    lon_b = lons[idx + 1]

    # interpolate geodesically
    lat_interp, lon_interp = _vectorized_interpolate_segment(lat_a, lon_a, lat_b, lon_b, fraction)

    # timestamps back to pandas times (add to t0)
    times_out = pd.to_datetime(t0) + pd.to_timedelta(target_times, unit="s")

    out_pdf = pd.DataFrame({lat_col: lat_interp, lon_col: lon_interp, time_col: times_out})
    # Remove duplicates if any
    out_pdf = out_pdf.drop_duplicates(subset=[lat_col, lon_col, time_col], keep="first").reset_index(drop=True)

    if isinstance(df, pl.DataFrame):
        return pl.from_pandas(out_pdf)
    return out_pdf
