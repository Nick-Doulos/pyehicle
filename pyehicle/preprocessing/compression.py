"""
Trajectory compression module for pyehicle.

This module provides spatio-temporal compression functionality to reduce the size
of GPS trajectories while preserving their essential shape and characteristics.
It uses DBSCAN-style clustering with KD-trees for efficient neighbor searches.
"""

import warnings
from typing import Optional, Union, Literal

import numpy as np
import pandas as pd
import polars as pl
from pyproj import Geod, Transformer

# Try to import cKDTree for fast spatial queries (10-100x faster than brute force)
try:
    from scipy.spatial import cKDTree as KDTree  # type: ignore
    _have_kdtree = True
except Exception:
    KDTree = None  # type: ignore
    _have_kdtree = False


# ======================== Helper Functions ========================


def _mean_timestamp_ns(arr: np.ndarray) -> np.datetime64:
    """
    Calculate the mean timestamp from an array of datetime64 values.

    This helper computes the average timestamp by converting to nanosecond integers,
    averaging, and converting back. This approach handles datetime arithmetic correctly.

    Parameters
    ----------
    arr : np.ndarray
        Array of datetime64 values.

    Returns
    -------
    np.datetime64
        The mean timestamp as a datetime64[ns] value.
    """
    # Convert to nanosecond integers for averaging
    ints = arr.astype("datetime64[ns]").astype("int64")
    avg = int(np.round(ints.mean()))
    # Convert back to datetime64
    return np.datetime64(avg, "ns")


def _to_ns_array(series_or_array) -> np.ndarray:
    """
    Convert various time representations to a numpy datetime64[ns] array.

    Handles pandas Series, numpy arrays, datetime strings, and other common
    time formats. Ensures consistent datetime64[ns] output for time calculations.

    Parameters
    ----------
    series_or_array : Series, array-like, or datetime-like
        Input time data in any common format.

    Returns
    -------
    np.ndarray
        Array of datetime64[ns] values.
    """
    arr = np.asarray(series_or_array)

    # Handle empty arrays
    if arr.size == 0:
        return np.array([], dtype="datetime64[ns]")

    # If already datetime64, just convert to nanosecond precision
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ns]")

    # Try direct pandas conversion
    try:
        return pd.to_datetime(arr).to_numpy(dtype="datetime64[ns]")
    except Exception:
        # Fallback: wrap in Series first (handles more edge cases)
        return pd.to_datetime(pd.Series(arr)).to_numpy(dtype="datetime64[ns]")


def _aggregate_group(
    df_group: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    time_col: Optional[str],
    agg_other: str
) -> pd.Series:
    """
    Aggregate a cluster of trajectory points into a single representative point.

    This helper function takes a group of spatially/temporally close points and
    combines them based on the specified aggregation strategy. The geographic
    centroid (mean lat/lon) is always used for position, while other columns
    are handled according to the agg_other parameter.

    Parameters
    ----------
    df_group : pd.DataFrame
        DataFrame containing the points in this cluster.
    lat_col : str
        Name of the latitude column.
    lon_col : str
        Name of the longitude column.
    time_col : str or None
        Name of the time column (if present).
    agg_other : str
        Aggregation strategy for non-spatial columns: "first", "last", or "mean".

    Returns
    -------
    pd.Series
        A Series representing the aggregated point with all original columns.

    Notes
    -----
    The geographic centroid is computed as the simple arithmetic mean of lat/lon.
    For very large clusters spanning significant distances, this is an approximation.
    The mean timestamp is computed if time_col is provided.
    """
    # Always compute the geographic centroid (mean of lat/lon)
    vals = {
        lat_col: float(df_group[lat_col].astype(float).mean()),
        lon_col: float(df_group[lon_col].astype(float).mean())
    }

    # Handle timestamp aggregation by computing the mean time
    if (time_col is not None) and (time_col in df_group.columns):
        times = _to_ns_array(df_group[time_col])
        vals[time_col] = pd.to_datetime(str(np.datetime_as_string(_mean_timestamp_ns(times))))

    # Process remaining columns (those that aren't lat, lon, or time)
    skip_cols = {lat_col, lon_col, time_col}
    other_cols = [c for c in df_group.columns if c not in skip_cols]

    # Apply the aggregation strategy to other columns
    if agg_other == "first":
        # Use values from the first point in the cluster
        first_row = df_group.iloc[0]
        for c in other_cols:
            vals[c] = first_row[c]
    elif agg_other == "last":
        # Use values from the last point in the cluster
        last_row = df_group.iloc[-1]
        for c in other_cols:
            vals[c] = last_row[c]
    elif agg_other == "mean":
        # Average numeric columns, use first value for non-numeric
        first_row = df_group.iloc[0]
        for c in other_cols:
            if pd.api.types.is_numeric_dtype(df_group[c].dtype):
                vals[c] = df_group[c].astype(float).mean()
            else:
                vals[c] = first_row[c]
    else:
        # Default to 'first' strategy for unknown values
        first_row = df_group.iloc[0]
        for c in other_cols:
            vals[c] = first_row[c]

    return pd.Series(vals)


# ======================== Main Compression Function ========================


def spatio_temporal_compress(
    df: Union[pd.DataFrame, pl.DataFrame],
    spatial_radius_km: float = 0.1,
    lat_col: str = "lat",
    lon_col: str = "lon",
    time_col: Optional[str] = "time",
    collapse_clusters: bool = True,
    time_threshold_s: Optional[float] = None,
    agg_other: Literal["first", "last", "mean"] = "first",
    min_samples: int = 1,
    drop_noise: bool = False,
    use_aeqd: bool = True,
    final_geodetic_check: bool = False,
    geodetic_check_tol_m: float = 0.0,
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Compress GPS trajectory by clustering nearby points using DBSCAN-style spatial clustering.

    This function reduces the number of points in a trajectory while preserving its essential
    shape by identifying and merging clusters of points that are close together in space
    (and optionally time). It uses a KD-tree for efficient O(n log n) neighbor searches
    and can optionally use an Azimuthal Equidistant (AEQD) projection for sub-meter
    distance accuracy.

    The compression algorithm works in several stages:
    1. Project coordinates to a local metric system (AEQD centered on trajectory centroid)
    2. Build a KD-tree for fast spatial range queries
    3. Find clusters of nearby points using DBSCAN
    4. Optionally filter clusters by temporal proximity
    5. Aggregate or select representatives from each cluster

    Parameters
    ----------
    df : pd.DataFrame or pl.DataFrame
        Input trajectory with latitude, longitude, and optionally time columns.
        Must contain at least lat_col and lon_col columns.
    spatial_radius_km : float, default=0.1
        Clustering radius in kilometers (converted to meters internally).
        Points within this distance are considered spatial neighbors.
        Typical values: 0.01-0.5 km for urban trajectories.
    lat_col : str, default="lat"
        Name of the latitude column (WGS84 decimal degrees).
    lon_col : str, default="lon"
        Name of the longitude column (WGS84 decimal degrees).
    time_col : str or None, default="time"
        Name of the time column. If None, no temporal filtering is applied.
        If provided, must be parseable by pandas.to_datetime().
    collapse_clusters : bool, default=True
        How to represent each cluster in the output:
        - True: Merge all cluster points into a single point (geographic centroid)
        - False: Keep one representative point from each cluster (the one with lowest index)
    time_threshold_s : float or None, default=None
        Temporal clustering threshold in seconds. If provided, points must be within
        this time window AND spatial_radius_km to be clustered together. Useful for
        separating overlapping paths taken at different times.
    agg_other : {"first", "last", "mean"}, default="first"
        How to aggregate non-spatial columns when collapse_clusters=True:
        - "first": Use values from the first point in the cluster
        - "last": Use values from the last point in the cluster
        - "mean": Average numeric columns, use first value for non-numeric
    min_samples : int, default=1
        Minimum number of neighbors (including self) required to form a core point.
        Similar to DBSCAN's min_samples parameter. Higher values filter out
        isolated points more aggressively.
    drop_noise : bool, default=False
        If True, drop points that don't belong to any cluster (noise points).
        If False, keep noise points as singleton clusters.
    use_aeqd : bool, default=True
        If True, use Azimuthal Equidistant projection centered on trajectory centroid
        for accurate distance calculations. If False or if AEQD fails, fallback to
        Web Mercator (EPSG:3857). AEQD is recommended for accurate compression.
    final_geodetic_check : bool, default=False
        If True, verify each cluster using geodesic distances (Geod.inv) after
        the initial projection-based clustering. Slower but provides additional
        accuracy verification. Usually not needed with use_aeqd=True.
    geodetic_check_tol_m : float, default=0.0
        Tolerance in meters for the final geodetic check. Cluster members must
        be within spatial_radius_km + geodetic_check_tol_m of the cluster centroid.
        Only used if final_geodetic_check=True.

    Returns
    -------
    pd.DataFrame or pl.DataFrame
        Compressed trajectory with the same type and column structure as input.
        The order of rows may differ from the input. The number of rows will be
        less than or equal to the input (depending on clustering results).

    Raises
    ------
    ValueError
        If lat_col or lon_col are not present in the DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> import pyehicle as pye
    >>>
    >>> # Load a dense GPS trajectory
    >>> df = pd.read_csv('dense_trajectory.csv')  # 1000 points
    >>> print(f"Original points: {len(df)}")
    >>>
    >>> # Simple compression with 100m radius
    >>> compressed = pye.preprocessing.spatio_temporal_compress(
    ...     df,
    ...     spatial_radius_km=0.1,  # 100 meters
    ...     collapse_clusters=True
    ... )
    >>> print(f"Compressed points: {len(compressed)}")  # ~200 points
    >>>
    >>> # Spatio-temporal compression (separate paths at different times)
    >>> compressed_st = pye.preprocessing.spatio_temporal_compress(
    ...     df,
    ...     spatial_radius_km=0.05,  # 50 meters
    ...     time_threshold_s=300,     # 5 minutes
    ...     collapse_clusters=True
    ... )
    >>>
    >>> # Aggressive compression: drop isolated points, larger radius
    >>> aggressive = pye.preprocessing.spatio_temporal_compress(
    ...     df,
    ...     spatial_radius_km=0.2,  # 200 meters
    ...     min_samples=3,           # Need at least 3 neighbors
    ...     drop_noise=True,         # Remove isolated points
    ...     agg_other="mean"         # Average other columns
    ... )

    Notes
    -----
    **Algorithm Details:**

    The function implements a DBSCAN-like clustering algorithm optimized for GPS data:

    1. **Projection**: Coordinates are projected to a local metric system for accurate
       distance calculations. By default, uses AEQD (Azimuthal Equidistant) projection
       centered on the trajectory's geographic centroid. This ensures distances are
       accurate within a few meters across the entire trajectory.

    2. **Spatial Indexing**: A KD-tree is built on the projected coordinates for
       O(log n) neighbor queries. Without scipy, falls back to O(n²) brute force.

    3. **Clustering**: DBSCAN algorithm identifies clusters of spatially connected points:
       - Core points: Points with ≥ min_samples neighbors within spatial_radius_km
       - Cluster formation: Core points and their neighbors form clusters
       - Noise: Points that don't belong to any cluster (if not dropped)

    4. **Temporal Filtering**: If time_threshold_s is provided, neighbor relationships
       are filtered to only include points within the time window. This prevents
       clustering of points from different trips on the same path.

    5. **Aggregation**: Clusters are represented either by their geographic centroid
       (collapse_clusters=True) or by selecting one representative point.

    **Performance:**
    - With scipy (cKDTree): O(n log n) - fast even for large trajectories
    - Without scipy: O(n²) - acceptable for <10,000 points, slow for larger datasets
    - AEQD projection: ~0.1ms per trajectory (one-time cost)
    - Geodetic check: O(n) extra cost per cluster (usually not needed)

    **Accuracy:**
    - AEQD projection: Sub-meter accuracy for distances up to ~1000 km from center
    - Web Mercator: Acceptable for small spatial_radius (<1 km) at mid-latitudes
    - Geodetic check: Verifies clusters using true geodesic distances (slowest but most accurate)

    **Typical Use Cases:**
    - Reduce storage size of dense GPS logs (e.g., 1 Hz sampling → 0.1 Hz effective)
    - Preprocessing for map-matching (remove redundant points)
    - Noise reduction in stationary periods (e.g., at traffic lights)
    - Simplify trajectory for visualization

    **Warnings:**
    - If scipy is not installed, a warning is issued and O(n²) fallback is used
    - If AEQD projection fails, a warning is issued and Web Mercator is used
    - Very large spatial_radius_km (>5 km) may produce unexpected results
    - Very small min_samples (<1) will be clamped to 1
    """
    # Initialize geodesic calculator for WGS84 ellipsoid
    geod = Geod(ellps="WGS84")
    radius_m = float(spatial_radius_km) * 1000.0  # Convert km to meters

    # ========== Input Validation and Setup ==========

    # Handle empty DataFrame (return early to avoid unnecessary processing)
    if isinstance(df, pd.DataFrame):
        if df.shape[0] == 0:
            return df.copy()
    else:  # polars DataFrame
        if len(df) == 0:
            return df.clone()

    # Convert to pandas for processing (convert back at the end if needed)
    input_was_polars = isinstance(df, pl.DataFrame)
    pdf = df.to_pandas() if input_was_polars else df.copy()

    # Verify required columns exist
    if lat_col not in pdf.columns or lon_col not in pdf.columns:
        raise ValueError("lat_col and lon_col must exist in the DataFrame")

    # Extract coordinate arrays for processing
    lats = pdf[lat_col].to_numpy(dtype=float)
    lons = pdf[lon_col].to_numpy(dtype=float)
    n = len(lats)

    # ========== Temporal Data Setup ==========

    # Convert time column to nanoseconds if temporal filtering is enabled
    if (time_col is not None) and (time_col in pdf.columns) and (time_threshold_s is not None):
        times = _to_ns_array(pdf[time_col])
        time_thresh_ns = np.timedelta64(int(round(time_threshold_s * 1e9)), "ns")
        t_ints = times.astype("datetime64[ns]").astype("int64")
    else:
        # No temporal filtering
        times = None
        time_thresh_ns = None
        t_ints = None

    # ========== Coordinate Projection ==========

    # Project lat/lon to a local metric coordinate system for accurate distance calculations
    # AEQD (Azimuthal Equidistant) centered on trajectory centroid is optimal for this
    use_local_aeqd = bool(use_aeqd)
    transformer = None
    xs = ys = None

    if use_local_aeqd:
        try:
            # Calculate trajectory centroid as projection center
            cen_lat = float(np.mean(lats))
            cen_lon = float(np.mean(lons))

            # Construct AEQD projection string centered at centroid
            # +lat_0, +lon_0: center point
            # +datum=WGS84: use WGS84 ellipsoid
            # +units=m: distances in meters
            aeqd_proj = f"+proj=aeqd +lat_0={cen_lat:.9f} +lon_0={cen_lon:.9f} +datum=WGS84 +units=m +no_defs"

            # Create transformer from WGS84 (EPSG:4326) to AEQD
            # always_xy=True ensures (lon, lat) order for transform()
            transformer = Transformer.from_crs("EPSG:4326", aeqd_proj, always_xy=True)

            # Transform all coordinates to AEQD projected space
            xs, ys = transformer.transform(lons, lats)
        except Exception:
            # AEQD failed (rare), fallback to Web Mercator
            warnings.warn("AEQD projection failed; falling back to EPSG:3857 projection.")
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
            xs, ys = transformer.transform(lons, lats)
    else:
        # User disabled AEQD, use Web Mercator (less accurate at high latitudes)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        xs, ys = transformer.transform(lons, lats)

    # ========== Spatial Neighbor Search ==========

    # Build neighbor lists using KD-tree for O(n log n) performance
    if _have_kdtree:
        # Fast path: scipy's cKDTree (C implementation)
        # Build tree on projected coordinates
        tree = KDTree(np.column_stack([xs, ys]))

        # Find all neighbors within radius_m for each point
        # Returns list of arrays, where neighbors[i] contains indices of points
        # within radius_m of point i (including i itself)
        neighbors = tree.query_ball_point(np.column_stack([xs, ys]), r=radius_m)
    else:
        # Slow path: O(n²) brute force neighbor search
        # Issue warning since this is significantly slower for large datasets
        warnings.warn(
            "scipy.spatial.cKDTree not available: falling back to slower O(n^2) neighbor search. "
            "Install scipy for much better performance."
        )
        neighbors = []
        r2 = radius_m * radius_m  # Use squared distance for faster comparisons

        # For each point, find all neighbors by checking distances
        for i in range(n):
            dx = xs - xs[i]  # X-distance to all points
            dy = ys - ys[i]  # Y-distance to all points
            d2 = dx * dx + dy * dy  # Squared Euclidean distance
            # Find indices where squared distance <= radius²
            neighbors.append(list(np.nonzero(d2 <= r2)[0]))

    # ========== Temporal Filtering (Optional) ==========

    # If time threshold specified, filter neighbor lists by temporal proximity
    if (times is not None) and (time_thresh_ns is not None):
        time_thresh_int = int(time_thresh_ns.astype("timedelta64[ns]").astype("int64"))
        t_ints_int64 = t_ints.astype("int64")

        # For each point, keep only neighbors within time window
        for i in range(n):
            neigh_idx = np.array(neighbors[i], dtype=int)
            if neigh_idx.size == 0:
                continue

            # Calculate absolute time differences (in nanoseconds)
            diffs_ns = np.abs(t_ints_int64[neigh_idx] - t_ints_int64[i])

            # Keep only neighbors within time threshold
            mask = diffs_ns <= time_thresh_int
            neighbors[i] = neigh_idx[mask].tolist()

    # ========== DBSCAN Clustering ==========

    # Run DBSCAN clustering algorithm to identify groups of connected points
    visited = np.zeros(n, dtype=bool)  # Track which points have been visited
    labels = -np.ones(n, dtype=int)     # Cluster labels (-1 = noise/unclustered)
    cluster_id = 0                       # Current cluster ID

    min_samples_thresh = max(1, min_samples)  # Ensure at least 1 sample required

    # Iterate through each point and expand clusters
    for i in range(n):
        if visited[i]:
            continue  # Skip already-visited points

        visited[i] = True
        neigh = np.array(neighbors[i], dtype=int)

        # Check if this is a core point (has enough neighbors)
        if neigh.size < min_samples_thresh:
            continue  # Not a core point, leave as noise (-1)

        # Start a new cluster with this core point
        labels[i] = cluster_id

        # Initialize cluster expansion queue with this point's neighbors
        seed = [int(j) for j in neigh if j != i]

        # Expand cluster by visiting all connected neighbors (breadth-first search)
        while seed:
            j = seed.pop()

            if not visited[j]:
                visited[j] = True
                neigh_j = np.array(neighbors[j], dtype=int)

                # If this neighbor is also a core point, add its neighbors to expansion queue
                if neigh_j.size >= min_samples_thresh:
                    for nb in neigh_j:
                        if not visited[nb]:
                            seed.append(int(nb))

            # Add this point to current cluster if it's not already assigned
            if labels[j] == -1:
                labels[j] = cluster_id

        # Move to next cluster ID
        cluster_id += 1

    # ========== Group Formation ==========

    # Organize points into groups based on cluster labels
    groups = {}
    for idx, lab in enumerate(labels):
        if lab == -1:
            # Noise point (not in any cluster)
            if drop_noise:
                continue  # Skip noise points if drop_noise=True
            else:
                # Keep noise as singleton clusters with unique keys
                key = f"noise_{idx}"
                groups.setdefault(key, []).append(idx)
        else:
            # Regular cluster point
            groups.setdefault(int(lab), []).append(idx)

    # ========== Geodetic Distance Verification (Optional) ==========

    # Optionally verify clusters using true geodesic distances
    # This is slower but provides additional accuracy verification
    # Usually not needed with AEQD projection
    if final_geodetic_check and len(groups) > 0:
        tol = float(geodetic_check_tol_m)
        radius_tol = radius_m + tol  # Allow small tolerance for numerical errors

        new_groups = {}
        for lab_key, idxs in groups.items():
            idxs_arr = np.array(idxs, dtype=int)

            # Get coordinates for this cluster
            group_lats = lats[idxs_arr]
            group_lons = lons[idxs_arr]

            # Calculate cluster centroid in geographic coordinates
            cen_lat = float(group_lats.mean())
            cen_lon = float(group_lons.mean())

            # Use pyproj Geod to calculate geodesic distances from centroid
            # Returns: forward azimuth, back azimuth, distance (meters)
            _, _, dists = geod.inv(
                np.full(len(idxs_arr), cen_lon),  # From centroid lon
                np.full(len(idxs_arr), cen_lat),  # From centroid lat
                group_lons,                        # To each point lon
                group_lats,                        # To each point lat
            )

            # Keep only points within tolerance
            keep_mask = dists <= radius_tol
            kept = idxs_arr[keep_mask]
            removed = idxs_arr[~keep_mask]

            # Reorganize clusters based on geodetic check results
            if kept.size == 0:
                # All points failed check - make them noise if not dropping
                if not drop_noise:
                    for rm in removed:
                        new_groups.setdefault(f"noise_{rm}", []).append(int(rm))
            else:
                # Keep the valid points in this cluster
                new_groups.setdefault(lab_key, []).extend(kept.tolist())
                # Handle removed points as noise
                if removed.size > 0 and not drop_noise:
                    for rm in removed:
                        new_groups.setdefault(f"noise_{rm}", []).append(int(rm))

        groups = new_groups

    # ========== Create Compressed Trajectory ==========

    # Generate final compressed trajectory from clusters
    rows = []
    for lab_key, idxs in groups.items():
        if not idxs:
            continue  # Skip empty groups

        idxs_arr = np.array(idxs, dtype=int)

        if collapse_clusters:
            # Aggregate all points in cluster into a single representative point
            # Uses geographic centroid for lat/lon, mean time, and agg_other strategy for rest
            group_df = pdf.iloc[idxs_arr]
            row = _aggregate_group(group_df, lat_col, lon_col, time_col, agg_other)
            rows.append(row)
        else:
            # Keep one representative point from each cluster (the one with lowest index)
            # This preserves original point data without aggregation
            rep_idx = int(idxs_arr.min())
            rows.append(pdf.iloc[rep_idx])

    # Build result DataFrame from aggregated/selected rows
    result_pdf = pd.DataFrame(rows).reset_index(drop=True) if rows else pd.DataFrame(columns=pdf.columns)

    # ========== Finalization ==========

    # Ensure all original columns are present with proper types
    for c in pdf.columns:
        if c not in result_pdf.columns:
            result_pdf[c] = pd.NA

    # Ensure time column has correct datetime type
    if (time_col is not None) and (time_col in pdf.columns) and (time_col in result_pdf.columns):
        result_pdf[time_col] = pd.to_datetime(result_pdf[time_col])

    # Reorder columns to match input DataFrame
    result_pdf = result_pdf[pdf.columns]

    # Return in the same DataFrame type as the input
    return pl.from_pandas(result_pdf) if input_was_polars else result_pdf
