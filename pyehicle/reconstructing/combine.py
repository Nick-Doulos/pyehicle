"""
Trajectory combination module for pyehicle.

This module provides functionality to intelligently combine multiple topologically
equivalent trajectories (e.g., different bus routes along the same road network,
multiple GPS recordings of the same route) by sorting them based on geographical
proximity using geodesic distances. This is the first step in trajectory reconstruction,
preparing multiple trajectory recordings for refinement and curve interpolation.

Important: This module removes duplicate coordinates and re-assigns timestamps
based on distance traveled. The output may have fewer points than the input and
original timestamps are replaced with synthetic distance-based timestamps.
"""

import pandas as pd
import numpy as np
from pyproj import Geod

# Global Geod instance for WGS84 geodesic distance calculations
# Reusing a single instance is more efficient than creating it repeatedly
geod = Geod(ellps="WGS84")


def trajectory_combiner(
    dataframes,
    lat_col='lat',
    lon_col='lon',
    time_col='time'
):
    """
    Combine multiple topologically equivalent trajectories and sort them by geographical proximity.

    This function takes multiple topologically equivalent trajectory DataFrames (e.g., different
    bus runs along the same route, multiple GPS recordings of the same path) and intelligently
    combines them into a single unified trajectory by sorting points based on nearest-neighbor
    distances. The algorithm starts from the first point and iteratively selects the closest
    unvisited point, creating a spatially continuous trajectory.

    This is particularly useful for:
    - Combining multiple recordings of the same bus/vehicle route
    - Merging GPS traces from different trips along the same path
    - Reconstructing a canonical route from multiple trajectory samples
    - Preparing topologically equivalent trajectories for refinement

    The function uses true geodesic distances (WGS84 ellipsoid) rather than Euclidean
    distances, ensuring accuracy across all latitudes and distances.

    Parameters
    ----------
    dataframes : list of pd.DataFrame
        A list of DataFrames containing topologically equivalent trajectories (e.g., multiple
        GPS recordings of the same route). Each DataFrame must have latitude and longitude
        columns. The DataFrames can have different lengths and may contain additional columns
        (all column types will be preserved).
    lat_col : str, default='lat'
        The column name for latitude values (WGS84 decimal degrees).
    lon_col : str, default='lon'
        The column name for longitude values (WGS84 decimal degrees).
    time_col : str, default='time'
        The column name for timestamp values. Used for timestamp re-assignment
        based on cumulative distance traveled.

    Returns
    -------
    pd.DataFrame
        A single DataFrame containing the combined trajectory, sorted by time after
        spatial reordering and timestamp re-assignment. The output may contain fewer
        points than the input due to duplicate coordinate removal (points at identical
        lat/lon locations are deduplicated, keeping only the first occurrence).
        The index is reset (0, 1, 2, ..., n-1). All original column types are preserved.

    Examples
    --------
    >>> import pandas as pd
    >>> import pyehicle as pye
    >>>
    >>> # Combine multiple bus runs along the same route
    >>> run1 = pd.read_csv('bus_route_66_monday_morning.csv')
    >>> run2 = pd.read_csv('bus_route_66_monday_afternoon.csv')
    >>> run3 = pd.read_csv('bus_route_66_tuesday_morning.csv')
    >>>
    >>> # Combine all runs into a single canonical trajectory
    >>> combined = pye.reconstructing.trajectory_combiner(
    ...     [run1, run2, run3],
    ...     lat_col='lat',
    ...     lon_col='lon',
    ...     time_col='time'
    ... )
    >>> print(f"Combined {len(run1) + len(run2) + len(run3)} points into {len(combined)} points")
    >>> # Note: Point count may be less due to duplicate coordinate removal

    >>> # Combine multiple GPS recordings of the same delivery route
    >>> monday_route = pd.read_csv('delivery_monday.csv')
    >>> tuesday_route = pd.read_csv('delivery_tuesday.csv')
    >>> wednesday_route = pd.read_csv('delivery_wednesday.csv')
    >>>
    >>> # Create canonical route from multiple recordings
    >>> canonical_route = pye.reconstructing.trajectory_combiner(
    ...     [monday_route, tuesday_route, wednesday_route]
    ... )

    Notes
    -----
    **Algorithm:**

    The function implements a multi-step reconstruction algorithm:

    1. **Concatenation**: All input DataFrames are concatenated into one
    2. **Spatial Sorting**: Greedy nearest-neighbor algorithm:
       - Start from the first point (index 0)
       - Calculate geodesic distance from current point to all unvisited points
       - Select the nearest unvisited point as the next point
       - Mark the selected point as visited and repeat
    3. **Duplicate Removal**: Remove points with identical (lat, lon) coordinates,
       keeping only the first occurrence. This eliminates redundant points from
       overlapping segments or stationary GPS readings.
    4. **Timestamp Re-assignment**: Calculate cumulative distances along the spatially-
       sorted path and re-assign timestamps proportional to distance traveled
       (constant speed assumption), preserving the original time range.
    5. **Time Sorting**: Sort by time to maintain the library's assumption of
       time-ordered trajectories. Since timestamps are now monotonically increasing
       along the path, this preserves spatial continuity.

    **Performance:**
    - Time complexity: O(n²) where n is the total number of points
    - Space complexity: O(n) for coordinate arrays and visited tracking
    - For 1000 points: ~1 second
    - For 10,000 points: ~100 seconds (consider preprocessing to reduce points first)

    **Accuracy:**
    - Uses pyproj Geod.inv() for true geodesic distances on WGS84 ellipsoid
    - Accurate to millimeter precision for all distances
    - No projection distortion (unlike Euclidean distance in lat/lon)

    **Important Considerations:**

    - **Point Count Reduction**: The output may have fewer points than the input
      due to duplicate coordinate removal. Points with identical (lat, lon) values
      are deduplicated, keeping only the first occurrence. This affects:
      * Loop trajectories (where paths cross themselves)
      * Stationary GPS readings (e.g., stopped at traffic lights)
      * Overlapping segments from multiple sources

    - **Timestamp Modification**: Original timestamps are discarded and replaced with
      new timestamps calculated from cumulative distances. The new timestamps:
      * Preserve the original time range (min to max)
      * Assume constant speed along the spatially-sorted path
      * Are monotonically increasing, ensuring time-ordered output

    - **Order Dependency**: The algorithm starts from the first point of the first
      DataFrame in the list. Different input orders may produce different results
      if segments overlap spatially.

    - **Greedy Strategy**: Uses nearest-neighbor heuristic, not globally optimal.
      For complex overlapping segments, may not produce the best possible ordering.
      However, this is fast and works well for most GPS trajectories.

    - **Memory Efficiency**: Coordinates are pre-extracted as numpy arrays for
      faster vectorized operations. Uses boolean array for visited tracking
      instead of a set for better iteration performance.

    - **Column Compatibility**: All DataFrames are concatenated, so column names
      and types must be compatible across all input DataFrames. All column types
      are preserved in the output.

    **Use Cases:**
    - **Route Canonicalization**: Combine multiple GPS recordings of the same bus/vehicle
      route to create a single canonical trajectory representative of that route
    - **Multi-Trip Fusion**: Merge GPS traces from different trips along the same path
      (e.g., daily commutes, delivery routes, public transit routes)
    - **Reconstruction Pipeline**: First step before `refine_trajectory()` and
      `curve_interpolation()` for final trajectory refinement

    **Limitations:**
    - Not suitable for very large datasets (>50,000 points) without preprocessing
    - Assumes segments are generally spatially separated (not heavily overlapping)
    - Duplicate coordinates are removed, which may be undesirable for:
      * Loop trajectories where the path legitimately crosses itself
      * Scenarios requiring stationary points (vehicle stopped, waiting)
      * Accurate point-count preservation
    - Timestamps are synthetic (distance-based), losing original timing information
    - Constant speed assumption may not reflect actual vehicle behavior
    - O(n²) complexity makes it slow for large point counts

    See Also
    --------
    refine_trajectory : Second step - enforces road network continuity
    curve_interpolation : Final step - interpolates smooth curves
    by_time : Splits trajectories by temporal gaps
    """
    # Concatenate all input DataFrames into a single DataFrame
    # ignore_index=True ensures we get a clean 0-based index
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Extract coordinates as numpy array for fast vectorized distance calculations
    # This is much faster than accessing DataFrame rows repeatedly
    coordinates = combined_df[[lat_col, lon_col]].to_numpy()
    n = len(coordinates)

    # Early exit for trivial cases (empty or single-point trajectories)
    if n <= 1:
        return combined_df

    # Pre-extract lat/lon arrays for faster geodesic distance calculations
    # pyproj's Geod.inv() can work with arrays, but we iterate manually for
    # the nearest-neighbor search, so separate arrays are more convenient
    lats = coordinates[:, 0]
    lons = coordinates[:, 1]

    # ========== Initialize Nearest-Neighbor Sorting ==========

    # Start from the first point (index 0 in the combined DataFrame)
    sorted_indices = [0]  # List of indices in their sorted order
    current_index = 0      # Index of the current point

    # Track visited points using boolean array (faster than set for dense iteration)
    visited = np.zeros(n, dtype=bool)
    visited[0] = True  # Mark starting point as visited

    # ========== Greedy Nearest-Neighbor Algorithm ==========

    # Continue until all points have been added to the sorted trajectory
    while len(sorted_indices) < n:
        # Get coordinates of current point
        current_lon = lons[current_index]
        current_lat = lats[current_index]

        # Initialize search for nearest unvisited point
        min_dist = float('inf')  # Minimum distance found so far
        nearest_index = -1        # Index of nearest point

        # Search all points to find the nearest unvisited one
        # Note: This is O(n) per iteration, making overall algorithm O(n²)
        for i in range(n):
            if not visited[i]:  # Only consider unvisited points
                # Calculate geodesic distance from current point to point i
                # Geod.inv returns: forward_azimuth, back_azimuth, distance
                # We only need the distance (in meters)
                _, _, dist = geod.inv(current_lon, current_lat, lons[i], lats[i])

                # Update nearest point if this one is closer
                if dist < min_dist:
                    min_dist = dist
                    nearest_index = i

        # Safety check: Break if no unvisited points found (shouldn't happen normally)
        if nearest_index == -1:
            break

        # Mark the nearest point as visited and add it to the sorted trajectory
        visited[nearest_index] = True
        sorted_indices.append(nearest_index)

        # Move to the nearest point for the next iteration
        current_index = nearest_index

    # ========== Reorder DataFrame ==========

    # Use the sorted indices to reorder the combined DataFrame
    # iloc[] preserves all columns and resets the index
    sorted_df = combined_df.iloc[sorted_indices].reset_index(drop=True)

    # ========== Duplicate Removal ==========
    # Remove points with identical (lat, lon) coordinates, keeping first occurrence
    # This eliminates redundant points from overlapping segments but also removes:
    # - Loop self-intersections (trajectory crossing itself)
    # - Stationary GPS readings (e.g., stopped at traffic lights)
    # NOTE: This reduces the point count. Consider removing this line if you need
    # to preserve all original points including duplicates and stationary readings.
    sorted_df = sorted_df.drop_duplicates(subset=[lat_col, lon_col]).reset_index(drop=True)

    # ========== Re-assign Timestamps Based on Spatial Order ==========
    # The trajectory is now spatially sorted, but timestamps may be out of order
    # Re-assign timestamps proportional to cumulative distance traveled
    # This ensures: (1) spatial continuity, (2) monotonic time ordering, (3) realistic timing

    n_points = len(sorted_df)
    if n_points > 1 and time_col in sorted_df.columns:
        # Get original time range from all input trajectories
        original_times = pd.to_datetime(sorted_df[time_col])
        start_time = original_times.min()
        end_time = original_times.max()
        total_time_span = (end_time - start_time).total_seconds()

        # Calculate cumulative distances along the spatially-sorted trajectory
        lats_sorted = sorted_df[lat_col].to_numpy()
        lons_sorted = sorted_df[lon_col].to_numpy()

        # Vectorized geodesic distance calculation for better performance
        # pyproj's Geod.inv() supports array inputs - 2-5x faster than looping
        _, _, segment_dists = geod.inv(
            lons_sorted[:-1], lats_sorted[:-1],  # All points except last
            lons_sorted[1:], lats_sorted[1:]      # All points except first
        )
        # Build cumulative distances: [0, dist0, dist0+dist1, dist0+dist1+dist2, ...]
        cumulative_distances = np.concatenate([[0], np.cumsum(segment_dists)])

        total_distance = cumulative_distances[-1]

        # Assign timestamps proportional to distance traveled (constant speed assumption)
        if total_distance > 0:
            # Calculate time at each point: t = start_time + (distance_fraction * total_time_span)
            distance_fractions = cumulative_distances / total_distance
            new_timestamps = [
                start_time + pd.Timedelta(seconds=frac * total_time_span)
                for frac in distance_fractions
            ]
            sorted_df[time_col] = new_timestamps
        else:
            # All points at same location - distribute time evenly
            if total_time_span > 0:
                time_intervals = pd.date_range(start=start_time, end=end_time, periods=n_points)
                sorted_df[time_col] = time_intervals
            # else: keep original timestamps (all same time anyway)

    # Now sort by time to maintain the library's assumption of time-ordered trajectories
    # Since we just assigned monotonically increasing timestamps based on spatial order,
    # this sort will preserve the spatial continuity
    sorted_df = sorted_df.sort_values(by=time_col).reset_index(drop=True)

    return sorted_df
