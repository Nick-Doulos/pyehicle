"""
Curve interpolation module for pyehicle.

This module provides functionality to improve trajectory realism by interpolating
additional points along road curves. It is the final step in trajectory reconstruction,
taking refined trajectories and adding intermediate points where the vehicle direction
changes (turns, curves, corners).

The interpolation process:
1. Analyzes bearing (direction) changes between consecutive trajectory points
2. Detects curves where bearing changes fall within a specified range
3. Finds the actual road path between the curve endpoints using the road network
4. Interpolates additional points along that path to smooth the curve
5. Distributes timestamps proportionally to distance traveled

This creates trajectories that closely follow actual road geometry, especially at:
- Intersections with gradual turns (e.g., highway exits, roundabouts)
- Curved roads (S-curves, mountain roads, residential streets)
- Any location where GPS sampling is sparse but direction changes significantly

The result is a high-fidelity trajectory suitable for visualization, analysis, and
applications requiring realistic vehicle paths (traffic simulation, route validation).
"""

import numpy as np
import pandas as pd
from pyproj import Geod
from shapely.geometry import LineString, Point
from sklearn.neighbors import BallTree
from pyehicle.utilities.road_network import preprocess_road_segments, filter_road_network_by_bbox

# Global geod for distance
geod = Geod(ellps="WGS84")


def build_balltree(G):
    """
    Builds a BallTree for approximate nearest-neighbor queries based on node coordinates.

    Parameters
    ----------
    G : igraph.Graph
        Road network graph with 'x' (longitude) and 'y' (latitude) node attributes.

    Returns
    -------
    tuple
        (tree, node_indices) where tree is a BallTree and node_indices is an array of node IDs.
    """
    node_coords = np.column_stack((G.vs['y'], G.vs['x']))  # (lat, lon)
    node_coords_rad = np.radians(node_coords)
    tree = BallTree(node_coords_rad)
    node_indices = np.arange(len(G.vs))
    return tree, node_indices


def find_nearest_node(
    G,
    tree,
    node_indices,
    point,
    point_osmid=None,
    k=3,
    max_node_distance=10
):
    """
    Finds the closest node in the same road (OSM ID) as the given point.

    This function ensures that interpolated points are placed on the correct road segment
    by finding network nodes that belong to the same OSM way as the trajectory point.

    Parameters
    ----------
    G : igraph.Graph
        Road network graph.
    tree : BallTree
        Pre-built BallTree for nearest neighbor queries.
    node_indices : np.ndarray
        Array of node indices.
    point : tuple
        (latitude, longitude) of the query point.
    point_osmid : int or list, optional
        OSM ID(s) of the road containing the point.
    k : int, default=3
        Number of nearest neighbors to consider.
    max_node_distance : float, default=25
        Maximum distance in meters to search for nodes.

    Returns
    -------
    int or None
        Index of the nearest node on the same road, or None if not found.
    """
    if point_osmid is None:
        return None

    # Query k nearest network nodes
    point_rad = np.radians([point])  # (lat, lon)
    _, idx_array = tree.query(point_rad, k=k)

    best_node_same_road = None
    best_score_same_road = float('inf')

    candidate_indices = idx_array[0]

    # Check each candidate to find the closest one on the same road
    for i in range(k):
        node_idx = int(node_indices[candidate_indices[i]])
        node_lat = G.vs[node_idx]['y']
        node_lon = G.vs[node_idx]['x']

        _, _, distance_meters = geod.inv(point[1], point[0], node_lon, node_lat)

        # Skip if too far or worse than current best
        if distance_meters > max_node_distance or distance_meters >= best_score_same_road:
            continue

        # Collect OSM IDs of roads connected to this node
        incident_edges = G.incident(node_idx)
        connected_osmids = set()
        for edge_id in incident_edges:
            osmval = G.es[edge_id]['osmid']
            if isinstance(osmval, list):
                connected_osmids.update(osmval)
            else:
                connected_osmids.add(osmval)

        # Update best if this node is on the same road
        if point_osmid in connected_osmids:
            best_score_same_road = distance_meters
            best_node_same_road = node_idx

    return best_node_same_road


def find_shortest_path(G, tree, node_indices, p1, p2, spatial_index, max_node_distance):
    """
    Identifies the path in the graph from p1 to p2.

    Parameters
    ----------
    G : igraph.Graph
        Road network graph.
    tree : BallTree
        Pre-built BallTree for nearest neighbor queries.
    node_indices : np.ndarray
        Array of node indices.
    p1 : Point
        Starting point.
    p2 : Point
        Ending point.
    spatial_index : rtree.Index
        Spatial index for road segments.
    max_node_distance : float
        Maximum distance for node search.

    Returns
    -------
    list or None
        List of edge indices forming the shortest path, or None if not found.
    """
    if spatial_index is None:
        return None

    # Find nearest road segments to each point
    edge_idx_p1 = next(spatial_index.nearest((p1.x, p1.y, p1.x, p1.y), 1))
    edge_idx_p2 = next(spatial_index.nearest((p2.x, p2.y, p2.x, p2.y), 1))

    p1_osmid = G.es[edge_idx_p1]['osmid']
    p2_osmid = G.es[edge_idx_p2]['osmid']

    # Find nearest nodes on the same OSM ID
    p1_node = find_nearest_node(G, tree, node_indices, (p1.y, p1.x), p1_osmid, max_node_distance=max_node_distance)
    if p1_node is None:
        return None
    p2_node = find_nearest_node(G, tree, node_indices, (p2.y, p2.x), p2_osmid, max_node_distance=max_node_distance)
    if p2_node is None or p1_node == p2_node:
        return None

    epath = G.get_shortest_paths(p1_node, to=p2_node, weights='length', output='epath')
    return epath[0] if epath and epath[0] else None


def reconstruct_path_line_from_nodes(G, path):
    """
    Reconstructs a LineString from a given list of edge indices.

    Parameters
    ----------
    G : igraph.Graph
        Road network graph.
    path : list
        List of edge indices.

    Returns
    -------
    LineString or None
        Reconstructed path geometry.
    """
    if not path:
        return None

    p1_node = G.es[path[0]].source
    p2_node = G.es[path[-1]].target

    vpath = G.get_shortest_paths(p1_node, to=p2_node, weights='length', output='vpath')
    if not vpath or not vpath[0]:
        return None

    node_path = vpath[0]
    coords = [(G.vs[n]['x'], G.vs[n]['y']) for n in node_path]

    # Remove consecutive duplicates
    filtered_coords = [coords[0]]
    for c in coords[1:]:
        if c != filtered_coords[-1]:
            filtered_coords.append(c)

    return LineString(filtered_coords) if len(filtered_coords) > 1 else None


def apply_minimum_gap(points, times, t1, t2, min_gap=1.0):
    """
    Apply minimum time gap between interpolated points to prevent timestamp collisions.

    This function prevents duplicate or excessively close timestamps in interpolated
    trajectories by:
    1. Nudging boundary timestamps away from segment endpoints (t1, t2)
    2. Removing consecutive duplicate timestamps
    3. Validating that at least 2 unique points remain

    This is critical for curve interpolation where multiple points may be added between
    two trajectory points, and temporal ordering must be maintained.

    Parameters
    ----------
    points : list of shapely.geometry.Point
        List of interpolated Point geometries along the curve.
    times : list of float
        List of timestamps (Unix epoch seconds) corresponding to each point.
        Same length as `points`.
    t1 : float
        Start timestamp of the curve segment (Unix epoch seconds). Boundary to avoid.
    t2 : float
        End timestamp of the curve segment (Unix epoch seconds). Boundary to avoid.
    min_gap : float, default=1.0
        Minimum time gap in seconds between consecutive timestamps. Default of 1 second
        provides reasonable temporal spacing for most GPS data.

    Returns
    -------
    points : list of shapely.geometry.Point or None
        Filtered list of points with valid timestamps, or None if fewer than 2 remain.
    times : list of float or None
        Filtered list of timestamps with min_gap enforced, or None if invalid.

    Notes
    -----
    **Boundary Nudging:**
    If the first or last timestamp exactly matches t1 or t2 (within 1e-9 tolerance),
    it's nudged by min_gap seconds to prevent exact duplicates with the segment endpoints.
    - First timestamp: nudged forward (t1 + min_gap)
    - Last timestamp: nudged backward (t2 - min_gap)

    **Duplicate Removal:**
    After nudging, consecutive duplicate timestamps are removed. This can occur when
    path reconstruction produces points at the same location or when nudging creates
    collisions.

    **Validation:**
    The function returns (None, None) if fewer than 2 unique points remain after
    filtering. This prevents invalid trajectories with insufficient data.

    **Use Case:**
    This function is called after `interpolate_points()` generates intermediate points
    along a curve. It ensures the temporal spacing is valid and prevents timestamp
    conflicts when merging interpolated points back into the main trajectory.

    Examples
    --------
    >>> from shapely.geometry import Point
    >>> import numpy as np
    >>>
    >>> # Interpolated points along a curve
    >>> points = [Point(24.10, 56.95), Point(24.105, 56.951), Point(24.11, 56.952)]
    >>> times = [100.0, 105.0, 105.0]  # Duplicate timestamp
    >>> t1, t2 = 100.0, 110.0
    >>>
    >>> # Apply minimum gap and remove duplicates
    >>> filtered_points, filtered_times = apply_minimum_gap(points, times, t1, t2, min_gap=1.0)
    >>>
    >>> print(f"Original: {len(points)} points")
    >>> print(f"Filtered: {len(filtered_points)} points")
    >>> print(f"Times: {filtered_times}")  # [101.0, 105.0] - first nudged, duplicate removed
    """
    # Convert times list to numpy array for vectorized operations
    times_arr = np.array(times, dtype=float)

    # ========== Boundary Nudging ==========
    # Adjust boundary times to avoid exact collisions with segment endpoints (t1, t2)
    # If first timestamp exactly matches t1, nudge it forward by min_gap
    if abs(times_arr[0] - t1) < 1e-9:
        times_arr[0] = t1 + min_gap  # Shift away from start boundary

    # If last timestamp exactly matches t2, nudge it backward by min_gap
    if abs(times_arr[-1] - t2) < 1e-9:
        times_arr[-1] = t2 - min_gap  # Shift away from end boundary

    # ========== Remove Consecutive Duplicate Timestamps ==========
    # Create boolean mask: True for first occurrence, False for consecutive duplicates
    # np.concatenate([True], ...) ensures the first element is always kept
    unique_mask = np.concatenate(([True], times_arr[1:] != times_arr[:-1]))

    # Filter points and times using the unique mask
    unique_pts = [points[i] for i in range(len(points)) if unique_mask[i]]
    unique_times = times_arr[unique_mask].tolist()

    # ========== Validation ==========
    # Ensure at least 2 unique points remain (minimum for a valid trajectory segment)
    if len(unique_pts) < 2:
        return None, None  # Invalid - not enough points

    # Return filtered points and times
    return unique_pts, unique_times


def interpolate_points(
    p1,
    p2,
    t1,
    t2,
    G,
    tree,
    node_indices,
    spatial_index,
    max_node_distance
):
    """
    Finds the shortest path between p1 and p2 and generates intermediate points/timestamps.

    Parameters
    ----------
    p1 : Point
        Starting point.
    p2 : Point
        Ending point.
    t1 : float
        Start timestamp.
    t2 : float
        End timestamp.
    G : igraph.Graph
        Road network graph.
    tree : BallTree
        Pre-built BallTree.
    node_indices : np.ndarray
        Node indices array.
    spatial_index : rtree.Index
        Spatial index for segments.
    max_node_distance : float
        Maximum node search distance.

    Returns
    -------
    tuple
        (points, times) lists of interpolated Points and timestamps, or (None, None).
    """
    path_data = find_shortest_path(G, tree, node_indices, p1, p2, spatial_index, max_node_distance)
    if not path_data:
        return None, None

    path_line = reconstruct_path_line_from_nodes(G, path_data)
    if not path_line:
        return None, None

    total_length = path_line.length
    if not total_length:
        return None, None

    # Extract coordinates from the path
    coords = list(path_line.coords)
    num_coords = len(coords)
    if num_coords < 2:
        return None, None

    # Calculate cumulative distances along the path
    lons = np.array([c[0] for c in coords])
    lats = np.array([c[1] for c in coords])

    # Vectorized geodesic distance calculation for better performance
    # pyproj's Geod.inv() supports array inputs - 2-5x faster than looping
    _, _, segment_dists = geod.inv(
        lons[:-1], lats[:-1],  # All points except last
        lons[1:], lats[1:]      # All points except first
    )
    # Build cumulative distances: [0, dist0, dist0+dist1, dist0+dist1+dist2, ...]
    dists = np.concatenate([[0], np.cumsum(segment_dists)])

    total_dist = dists[-1]
    if not total_dist:
        return None, None

    # Interpolate timestamps proportional to distance along path
    time_diff = t2 - t1
    fractions = dists / total_dist
    times = t1 + fractions * time_diff

    # Convert coordinates to Point objects
    points = [Point(coords[i]) for i in range(num_coords)]

    # Apply minimum gap and remove duplicates
    new_pts, new_times = apply_minimum_gap(points, times.tolist(), t1, t2, min_gap=1.0)
    return (new_pts, new_times) if new_pts else (None, None)


def compute_bearing_pyproj(lat1, lon1, lat2, lon2):
    """
    Compute the forward azimuth (bearing) from point 1 to point 2 using geodesic calculation.

    Bearing represents the compass direction of travel from one point to another, measured
    in degrees clockwise from true north. This function uses pyproj's geodesic inverse
    calculation for accuracy across all latitudes and distances.

    Parameters
    ----------
    lat1 : float
        Latitude of the starting point in WGS84 decimal degrees.
    lon1 : float
        Longitude of the starting point in WGS84 decimal degrees.
    lat2 : float
        Latitude of the ending point in WGS84 decimal degrees.
    lon2 : float
        Longitude of the ending point in WGS84 decimal degrees.

    Returns
    -------
    float
        Forward azimuth (bearing) in degrees, normalized to [0, 360).
        - 0° = North
        - 90° = East
        - 180° = South
        - 270° = West

    Notes
    -----
    **Geodesic vs Euclidean:**
    This function uses geodesic (great-circle) calculation on the WGS84 ellipsoid, which is
    accurate for all distances and latitudes. Simple Euclidean calculation (atan2) would fail
    at high latitudes or long distances.

    **Bearing Convention:**
    Returns bearing in the range [0, 360) degrees:
    - Always positive (no negative angles)
    - Clockwise from true north (standard compass convention)

    **Use Case:**
    This function is used to calculate trajectory bearings for curve detection. Large bearing
    changes (e.g., 90°) indicate sharp turns, while small changes (e.g., 10°) indicate gradual curves.

    Examples
    --------
    >>> # Calculate bearing from Riga to Stockholm (approximately NNW)
    >>> bearing = compute_bearing_pyproj(56.95, 24.10, 59.33, 18.07)
    >>> print(f"Bearing: {bearing:.1f}°")  # ~330° (NNW)
    >>>
    >>> # Calculate bearing change at a turn
    >>> p1 = (56.95, 24.10)  # Before turn
    >>> p2 = (56.952, 24.11)  # At turn
    >>> p3 = (56.951, 24.12)  # After turn
    >>>
    >>> bearing1 = compute_bearing_pyproj(*p1, *p2)  # Bearing into turn
    >>> bearing2 = compute_bearing_pyproj(*p2, *p3)  # Bearing out of turn
    >>> change = abs(bearing2 - bearing1)
    >>> change = change if change <= 180 else 360 - change  # Handle wrap-around
    >>> print(f"Bearing change: {change:.1f}°")  # Indicates curve severity
    """
    # Calculate forward azimuth using pyproj geodesic inverse
    # geod.inv(lon1, lat1, lon2, lat2) returns (fwd_az, back_az, distance)
    # Note: pyproj uses (lon, lat) order, not (lat, lon)
    fwd_az, _, _ = geod.inv(lon1, lat1, lon2, lat2)

    # Normalize bearing to [0, 360) range
    # pyproj returns azimuths in (-180, 180], so add 360 and take modulo
    return (fwd_az + 360) % 360


def calculate_bearings_pyproj(df, lat_col='lat', lon_col='lon'):
    """
    Calculate bearings (forward azimuths) between all consecutive points in a trajectory.

    This function computes the direction of travel at each trajectory point by calculating
    the bearing to the next point. The result is an array of bearings representing the
    vehicle's heading at each location.

    Bearings are used for curve detection: large bearing changes between consecutive points
    indicate curves, turns, or corners that may require additional interpolation.

    Parameters
    ----------
    df : pd.DataFrame
        Trajectory DataFrame with columns for latitude and longitude. Points should be
        in chronological order.
    lat_col : str, default='lat'
        Column name for latitude values (WGS84 decimal degrees).
    lon_col : str, default='lon'
        Column name for longitude values (WGS84 decimal degrees).

    Returns
    -------
    np.ndarray
        Array of bearings in degrees [0, 360), same length as the input DataFrame.
        - bearings[i] = direction from point i to point i+1
        - bearings[-1] = NaN (no next point to calculate bearing to)
        - bearings[i] = NaN if points i and i+1 are identical (zero distance)

    Notes
    -----
    **Bearing Interpretation:**
    - 0° = Traveling North
    - 90° = Traveling East
    - 180° = Traveling South
    - 270° = Traveling West

    **NaN Values:**
    The last element is always NaN because there's no following point. Additionally,
    if consecutive points are identical (duplicate coordinates), the bearing is set to NaN.

    **Use Case:**
    This function is called by `curve_interpolation()` to analyze the trajectory. Bearing
    changes (differences between consecutive bearings) identify locations where the vehicle
    direction changes, indicating curves that may need interpolation.

    **Performance:**
    - Time complexity: O(n) where n is the number of trajectory points
    - Uses geodesic calculation for each consecutive pair

    Examples
    --------
    >>> import pandas as pd
    >>> import pyehicle as pye
    >>>
    >>> # Create trajectory with a 90° turn
    >>> df = pd.DataFrame({
    ...     'lat': [56.95, 56.96, 56.96, 56.97],  # North, then North, then East, then East
    ...     'lon': [24.10, 24.10, 24.11, 24.12]
    ... })
    >>>
    >>> # Calculate bearings
    >>> bearings = pye.reconstructing.curve_interpolation.calculate_bearings_pyproj(df)
    >>> print(bearings)
    >>> # [0.0, 90.0, 90.0, nan]  # North → East turn at index 1
    >>>
    >>> # Calculate bearing changes for curve detection
    >>> bearing_changes = pd.Series(bearings).diff().abs()
    >>> bearing_changes = np.where(bearing_changes <= 180, bearing_changes, 360 - bearing_changes)
    >>> print(f"Bearing changes: {bearing_changes}")
    >>> # Large change at index 1 indicates turn
    """
    # Extract coordinate arrays for efficient iteration
    lats = df[lat_col].to_numpy()
    lons = df[lon_col].to_numpy()
    n = len(df)

    # Initialize bearings array with NaN (last element will remain NaN)
    bearings = np.full(n, np.nan, dtype=float)

    # Calculate bearing for each consecutive pair of points
    for i in range(n - 1):
        lat1, lon1 = lats[i], lons[i]      # Current point
        lat2, lon2 = lats[i + 1], lons[i + 1]  # Next point

        # Skip if points are identical (zero distance → undefined bearing)
        # This can occur with duplicate GPS readings or after deduplication edge cases
        if lat1 == lat2 and lon1 == lon2:
            continue  # Leave bearings[i] as NaN

        # Calculate geodesic bearing from current point to next point
        bearings[i] = compute_bearing_pyproj(lat1, lon1, lat2, lon2)

    return bearings


def curve_interpolation(df, road_network, lower_threshold=20, upper_threshold=80, max_node_distance=10, time_col='time', lat_col='lat', lon_col='lon'):
    """
    Improve trajectory realism by interpolating points along road curves based on bearing changes.

    This is the final step in trajectory reconstruction. It takes a refined trajectory and adds
    intermediate points at locations where the vehicle direction changes (curves, turns, corners).
    By detecting bearing changes and filling in the actual road geometry, it creates high-fidelity
    trajectories that closely follow real-world road paths.

    The function analyzes bearing (direction) changes between consecutive points. When a bearing
    change falls within the specified threshold range, it indicates a curve that would benefit from
    additional points. The function then:
    1. Finds the road network path between the curve endpoints
    2. Interpolates multiple points along that path
    3. Assigns timestamps proportionally to distance traveled

    This is essential for:
    - Visualizing realistic vehicle paths (trajectories that look like actual driving)
    - Traffic analysis requiring accurate road following (lane-level positioning)
    - Route validation and comparison (matching GPS traces to expected routes)
    - Simulation and animation (smooth, realistic vehicle movement)

    **Algorithm Overview:**
    1. **Bearing Analysis**: Calculate bearing (direction) between each consecutive point pair
    2. **Bearing Change Detection**: Compute absolute change in bearing (accounts for 360° wrap)
    3. **Curve Identification**: Flag points where bearing change ∈ [lower_threshold, upper_threshold]
    4. **Path Interpolation**: For each curve, find road network path and interpolate points
    5. **Temporal Distribution**: Assign timestamps proportional to distance along interpolated path

    **Why Bearing Thresholds?**
    - **Too small (<20°)**: Captures noise and minor GPS drift, not real curves
    - **Sweet spot (20-80°)**: Gradual turns, highway exits, curved roads, roundabouts
    - **Too large (>80°)**: Sharp 90° turns already handled by `refine_trajectory()`

    Parameters
    ----------
    df : pd.DataFrame
        Input trajectory DataFrame with columns for latitude, longitude, and timestamp.
        Should be output from `refine_trajectory()` for best results. Minimum columns:
        lat_col, lon_col, time_col.
    road_network : igraph.Graph
        Road network graph loaded from OSM data. Must have:
        - Node attributes: 'x' (longitude), 'y' (latitude)
        - Edge attributes: 'osmid' (OpenStreetMap way ID), 'geometry' (LineString)
        Typically loaded via `utilities.road_network.load_road_network()`.
    lower_threshold : float, default=20
        Minimum bearing change in degrees to trigger curve interpolation. Bearing changes
        smaller than this are considered straight segments (no interpolation needed).
        Typical values:
        - 10°: Very sensitive, captures subtle curves (may add noise)
        - 20°: Recommended default, captures most real curves
        - 30°: Less sensitive, only significant curves
    upper_threshold : float, default=80
        Maximum bearing change in degrees to trigger curve interpolation. Bearing changes
        larger than this are considered sharp turns (already handled by `refine_trajectory()`
        or too abrupt for curve interpolation).
        Typical values:
        - 60°: Captures gradual to moderate curves
        - 80°: Recommended default, avoids sharp 90° turns
        - 90°: Includes sharper turns (may overlap with refinement)
    max_node_distance : float, default=10
        Maximum distance in meters for snapping trajectory points to road nodes during
        interpolation. Points farther than this will skip interpolation. Typical values:
        - 10-15m: High-accuracy GPS (recommended)
        - 20-30m: Standard GPS accuracy
        - 40-50m: Low-accuracy GPS or sparse road networks
    time_col : str, default='time'
        Name of the timestamp column in df. Can be Unix timestamps (numeric) or formatted
        datetime strings. Supports various formats.
    lat_col : str, default='lat'
        Name of the latitude column in df (WGS84 decimal degrees).
    lon_col : str, default='lon'
        Name of the longitude column in df (WGS84 decimal degrees).

    Returns
    -------
    pd.DataFrame
        Enhanced trajectory with interpolated points at curves. Contains the same columns
        as the input (lat_col, lon_col, time_col) with:
        - Original trajectory points preserved
        - Intermediate points added at detected curves
        - Timestamps distributed proportionally to distance
        - Points sorted chronologically
        - Duplicates removed
        Output format: Timestamps as strings ('%Y-%m-%d %H:%M:%S')

    Examples
    --------
    >>> import pandas as pd
    >>> import pyehicle as pye
    >>>
    >>> # Load refined trajectory (output from refine_trajectory)
    >>> df = pd.read_csv('refined_trajectory.csv')
    >>> print(f"Refined trajectory: {len(df)} points")
    >>>
    >>> # Load road network
    >>> G = pye.utilities.road_network.load_road_network('city_roads.graphml')
    >>>
    >>> # Interpolate curves with default thresholds (20-80°)
    >>> final = pye.reconstructing.curve_interpolation(
    ...     df,
    ...     road_network=G,
    ...     lower_threshold=20,
    ...     upper_threshold=80,
    ...     max_node_distance=10
    ... )
    >>>
    >>> print(f"Final trajectory: {len(final)} points")
    >>> print(f"Added {len(final) - len(df)} points at curves")
    >>>
    >>> # Visualize before/after
    >>> pye.utilities.visualization.multiple(
    ...     [df, final],
    ...     names=['Refined', 'With Curves'],
    ...     show_in_browser=True
    ... )

    >>> # Full reconstruction pipeline (all steps)
    >>> import pyehicle as pye
    >>>
    >>> # 1. Preprocessing
    >>> df = pd.read_csv('raw_gps.csv')
    >>> compressed = pye.preprocessing.spatio_temporal_compress(df)
    >>> matched = pye.preprocessing.leuven(compressed, city='Riga', country='Latvia')
    >>>
    >>> # 2. Segmentation and combination
    >>> segments = pye.preprocessing.by_time(matched, time_threshold=300)
    >>> combined = pye.reconstructing.trajectory_combiner(segments)
    >>>
    >>> # 3. Refinement (road network continuity)
    >>> G = pye.utilities.road_network.load_road_network('riga_roads.graphml')
    >>> refined = pye.reconstructing.refine_trajectory(combined, G, max_node_distance=10)
    >>>
    >>> # 4. Curve interpolation (THIS FUNCTION - final step)
    >>> final = pye.reconstructing.curve_interpolation(
    ...     refined, G,
    ...     lower_threshold=20,
    ...     upper_threshold=80,
    ...     max_node_distance=10
    ... )
    >>>
    >>> # Save final high-fidelity trajectory
    >>> final.to_csv('reconstructed_trajectory.csv', index=False)
    >>> print(f"✓ Reconstruction complete: {len(final)} points")

    Notes
    -----
    **Algorithm Details:**
    1. Calculate bearings between consecutive points using geodesic forward azimuth
    2. Compute bearing changes: abs(bearing[i] - bearing[i-1]), handling 360° wrap-around
    3. For each point where lower_threshold ≤ bearing_change ≤ upper_threshold:
       - Find shortest path through road network from previous point to current point
       - Reconstruct LineString geometry from the path
       - Calculate cumulative distances along the path
       - Interpolate timestamps proportionally: t = t1 + (d/total_d) * (t2-t1)
       - Replace the two original points with the interpolated path points
    4. Deduplicate coordinates and sort by timestamp

    **Bearing Change Calculation:**
    Bearing changes handle the circular nature of angles (360° = 0°):
    ```python
    change = abs(bearing2 - bearing1)
    change = change if change <= 180 else 360 - change
    ```
    This ensures: 350° → 10° = 20° change (not 340°)

    **Temporal Interpolation:**
    Timestamps are distributed proportionally to distance along the interpolated path,
    assuming constant speed:
    - t[i] = t1 + (cumulative_distance[i] / total_distance) * (t2 - t1)
    - This maintains realistic timing and ensures chronological order

    **Road Network Filtering:**
    The function filters the road network to the trajectory's bounding box (0.5° buffer)
    for performance, similar to `refine_trajectory()`.

    **Performance:**
    - Time complexity: O(n * m * k) where:
      - n = trajectory points
      - m = avg points per interpolated curve
      - k = avg edges in shortest path
    - For 1000-point trajectory with 20 curves: 10-60 seconds
    - Dominated by shortest path computation (Dijkstra's algorithm)

    **Quality Tuning:**
    - **More aggressive** (lower_threshold=5, upper_threshold=60):
      - Captures more curves
      - More interpolated points
      - Smoother trajectories
      - Slower processing
      - Risk of over-interpolation
    - **Less aggressive** (lower_threshold=15, upper_threshold=40):
      - Captures only significant curves
      - Fewer interpolated points
      - Faster processing
      - May miss subtle curves

    **Limitations:**
    - Requires road network with valid geometry and OSM IDs
    - May fail for disconnected roads or missing road segments
    - Does not handle U-turns well (>180° bearing changes)
    - Assumes GPS points are reasonably map-matched (best after `refine_trajectory()`)
    - Does not consider speed, acceleration, or vehicle dynamics

    **Use Cases:**
    - Final step in trajectory reconstruction pipeline
    - Preparing trajectories for visualization and presentation
    - Generating realistic vehicle paths for traffic simulation
    - Route validation and GPS trace comparison
    - Creating training data for machine learning (realistic vehicle behavior)

    **When NOT to Use:**
    - Raw GPS data (use `refine_trajectory()` first)
    - Trajectories with very high sampling rate (already smooth)
    - Applications not requiring curve realism (simple analysis)

    See Also
    --------
    trajectory_combiner : First step - combine and sort trajectory segments
    refine_trajectory : Second step - enforce road network continuity
    calculate_bearings_pyproj : Calculate trajectory bearings for curve detection
    utilities.road_network.load_road_network : Load OSM road network graph
    """
    # Work on a copy to avoid SettingWithCopyWarning
    df = df.copy()

    road_network = filter_road_network_by_bbox(road_network, df, buffer=0.5)

    # Clean and prepare trajectory data
    df = df.drop_duplicates(subset=[lat_col, lon_col]).sort_values(by=time_col).reset_index(drop=True)

    # Convert timestamps to Unix time (seconds since epoch)
    if pd.api.types.is_numeric_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], unit='s')
    else:
        df[time_col] = pd.to_datetime(df[time_col], format='%Y-%m-%d %H:%M:%S')
    df[time_col] = df[time_col].astype('int64') / 1e9

    # Build spatial index and BallTree for efficient road network queries
    _, spatial_index = preprocess_road_segments(road_network)
    tree, node_indices = build_balltree(road_network)

    # Calculate bearings and bearing changes
    bearings = calculate_bearings_pyproj(df, lat_col=lat_col, lon_col=lon_col)
    df['bearing'] = bearings

    # Calculate absolute bearing change, accounting for circular nature of angles
    bearing_diff = df['bearing'].diff().abs()
    df['bearing_change'] = np.where(bearing_diff <= 180, bearing_diff, 360 - bearing_diff)

    # Extract coordinate and time arrays for processing
    lons = df[lon_col].to_numpy()
    lats = df[lat_col].to_numpy()
    times = df[time_col].to_numpy()
    bearing_changes = df['bearing_change'].to_numpy()

    # Convert coordinates to Shapely Point objects
    points = [Point(lon, lat) for lon, lat in zip(lons, lats)]

    interpolated_points = []
    interpolated_timestamps = []
    last_point = None

    num_points = len(points)
    for i in range(num_points):
        # Always add the first point
        if i == 0:
            interpolated_points.append(points[i])
            interpolated_timestamps.append(times[i])
            last_point = points[i]
            continue

        prev_point = points[i - 1]
        curr_point = points[i]
        prev_time = times[i - 1]
        curr_time = times[i]

        # Check bearing change threshold
        bc = bearing_changes[i]
        if not np.isnan(bc) and lower_threshold <= bc <= upper_threshold:
            # Attempt interpolation
            ipts, itms = interpolate_points(
                prev_point, curr_point,
                prev_time, curr_time,
                road_network,
                tree,
                node_indices,
                spatial_index,
                max_node_distance=max_node_distance
            )
            if ipts and itms and len(ipts) > 2:
                # Remove the previously added point to avoid duplication
                interpolated_points.pop()
                interpolated_timestamps.pop()

                # Insert new
                for p, t in zip(ipts, itms):
                    interpolated_points.append(p)
                    interpolated_timestamps.append(t)
                    last_point = p
            else:
                interpolated_points.append(curr_point)
                interpolated_timestamps.append(curr_time)
                last_point = curr_point
        else:
            interpolated_points.append(curr_point)
            interpolated_timestamps.append(curr_time)
            last_point = curr_point

    # Ensure the final trajectory point is included
    if points and points[-1] != last_point:
        interpolated_points.append(points[-1])
        interpolated_timestamps.append(times[-1])

    # Extract coordinates from Point objects
    final_lats = np.array([p.y for p in interpolated_points])
    final_lons = np.array([p.x for p in interpolated_points])

    # Build output DataFrame
    final_df = pd.DataFrame({
        lat_col: final_lats,
        lon_col: final_lons,
        time_col: interpolated_timestamps
    })

    # Convert timestamps back to human-readable format
    final_df[time_col] = pd.to_datetime(final_df[time_col], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')

    # Clean up and sort final trajectory
    final_df = final_df.drop_duplicates(subset=[lat_col, lon_col], keep='first').sort_values(by=time_col).reset_index(drop=True)
    return final_df
